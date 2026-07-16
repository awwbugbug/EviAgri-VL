import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from static_qlora_config import load_training_config
from static_qlora_data import AssistantOnlyVisionCollator, JsonlDataset, preflight_dataset
from static_qlora_model import build_qlora_model


@dataclass(frozen=True)
class ModePaths:
    train_jsonl: Path
    val_jsonl: Path
    output_dir: Path
    full_preflight: Path


def _stable_rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def deterministic_smoke_subset(
    rows: list[dict[str, Any]], positive: int, null: int, seed: int
) -> list[dict[str, Any]]:
    positives = [row for row in rows if row.get("task_type") == "pest_evidence_grounding"]
    nulls = [row for row in rows if row.get("task_type") == "prompt_conflict_null_evidence"]
    if len(positives) < positive or len(nulls) < null:
        raise ValueError(
            f"smoke subset requires {positive} positive/{null} null, found {len(positives)}/{len(nulls)}"
        )
    selected = sorted(positives, key=lambda row: _stable_rank(seed, row["id"]))[:positive]
    selected += sorted(nulls, key=lambda row: _stable_rank(seed, row["id"]))[:null]
    return sorted(selected, key=lambda row: _stable_rank(seed + 1, row["id"]))


def resolve_mode_paths(config: dict[str, Any], mode: str) -> ModePaths:
    if mode not in {"smoke", "formal"}:
        raise ValueError(f"unsupported training mode: {mode}")
    mixed = Path(config["mixed_data_root"])
    experiment = Path(config["experiment_root"])
    return ModePaths(
        train_jsonl=mixed / "train.jsonl",
        val_jsonl=mixed / "val.jsonl",
        output_dir=experiment / mode,
        full_preflight=experiment / "preparation" / "full_preflight.json",
    )


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty training output: {path}")


def training_argument_values(config: dict[str, Any], mode: str) -> dict[str, Any]:
    training = config["training"]
    values: dict[str, Any] = {
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "num_train_epochs": training["num_train_epochs"],
        "learning_rate": training["learning_rate"],
        "lr_scheduler_type": training["lr_scheduler_type"],
        "warmup_ratio": training["warmup_ratio"],
        "weight_decay": training["weight_decay"],
        "max_grad_norm": training["max_grad_norm"],
        "optim": training["optim"],
        "bf16": training["bf16"],
        "tf32": training["tf32"],
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": training["logging_steps"],
        "logging_first_step": True,
        "logging_nan_inf_filter": False,
        "eval_strategy": "steps",
        "eval_steps": training["eval_steps"],
        "save_strategy": "steps",
        "save_steps": training["save_steps"],
        "save_total_limit": training["save_total_limit"],
        "dataloader_num_workers": training["dataloader_num_workers"],
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "label_names": ["labels"],
        "report_to": [],
        "seed": config["seed"],
        "data_seed": config.get("data_seed", config["seed"]),
    }
    if mode == "smoke":
        values.update(
            {
                "max_steps": config["smoke"]["max_steps"],
                "gradient_accumulation_steps": config["smoke"]["gradient_accumulation_steps"],
                "logging_steps": 1,
                "eval_strategy": "no",
                "save_strategy": "no",
                "dataloader_num_workers": 0,
            }
        )
    elif mode != "formal":
        raise ValueError(f"unsupported training mode: {mode}")
    return values


def extract_logged_losses(history: list[dict[str, Any]]) -> list[float]:
    return [float(row["loss"]) for row in history if "loss" in row]


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_full_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"full train/val preflight has not passed: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("splits", {}).get("train", {}).get("samples") != 20478:
        raise RuntimeError("full preflight train count is not 20,478")
    if report.get("splits", {}).get("val", {}).get("samples") != 3052:
        raise RuntimeError("full preflight val count is not 3,052")
    return report


def run_training(config: dict[str, Any], mode: str, resume_from_checkpoint: Path | None = None) -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import AutoProcessor, Trainer, TrainingArguments

    paths = resolve_mode_paths(config, mode)
    full_preflight = _load_full_preflight(paths.full_preflight)
    if resume_from_checkpoint is None:
        ensure_empty_output(paths.output_dir)
    elif not resume_from_checkpoint.is_dir():
        raise ValueError(f"resume checkpoint does not exist: {resume_from_checkpoint}")
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        processor = AutoProcessor.from_pretrained(
            config["model_path"],
            min_pixels=config["vision"]["min_pixels"],
            max_pixels=config["vision"]["max_pixels"],
            use_fast=False,
            local_files_only=True,
        )
        collator = AssistantOnlyVisionCollator(processor, config["training"]["max_length"])
        train_source = JsonlDataset(paths.train_jsonl)
        eval_dataset = JsonlDataset(paths.val_jsonl)
        if mode == "smoke":
            train_dataset = deterministic_smoke_subset(
                train_source.records,
                positive=config["smoke"]["positive"],
                null=config["smoke"]["null"],
                seed=config["seed"],
            )
            mode_preflight = preflight_dataset(
                train_dataset, collator, config["training"]["max_length"]
            )
            trainer_eval_dataset = None
        else:
            train_dataset = train_source
            mode_preflight = full_preflight["splits"]["train"]
            trainer_eval_dataset = eval_dataset

        model, trainable, targets = build_qlora_model(config)
        argument_values = training_argument_values(config, mode)
        argument_values["output_dir"] = str(paths.output_dir)
        argument_values["run_name"] = f"static_qlora_v1_{mode}"
        arguments = TrainingArguments(**argument_values)
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=trainer_eval_dataset,
            data_collator=collator,
            processing_class=processor,
        )
        torch.cuda.reset_peak_memory_stats()
        train_result = trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
        adapter_dir = paths.output_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(paths.output_dir / "processor")
        trainer.save_state()
        losses = extract_logged_losses(trainer.state.log_history)
        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        summary = {
            "mode": mode,
            "completed": True,
            "elapsed_seconds": time.time() - started,
            "train_metrics": train_result.metrics,
            "losses": losses,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "trainable_parameter_count": len(trainable),
            "lora_target_count": len(targets),
            "preflight": mode_preflight,
            "adapter_dir": str(adapter_dir),
        }
        _write_json_atomic(paths.output_dir / "config.snapshot.json", config)
        _write_json_atomic(paths.output_dir / "environment.json", environment)
        _write_json_atomic(paths.output_dir / "trainable_parameters.json", trainable)
        _write_json_atomic(paths.output_dir / "lora_targets.json", targets)
        _write_json_atomic(paths.output_dir / "run_summary.json", summary)
        return summary
    except Exception as error:
        failure = {
            "mode": mode,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        _write_json_atomic(paths.output_dir / "failure.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Static QLoRA v1")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    summary = run_training(config, args.mode, args.resume_from_checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
