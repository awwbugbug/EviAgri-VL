"""Immutable smoke/formal trainer for Task 9D Static QLoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from task9d_config import load_task9d_config
from task9d_data import AssistantOnlyV2Collator, Task9dDataset
from task9d_model import build_task9d_model


def run_directory(experiment_root: str | Path, variant: str, seed: int) -> Path:
    if variant not in {"A", "B", "C"} or seed not in {17, 29, 43}:
        raise ValueError("Task 9D run must use frozen variant and seed")
    return Path(experiment_root) / "runs" / variant / f"seed_{seed}"


def ensure_new_output(path: str | Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite Task 9D output: {path}")


def task9d_training_arguments(*, seed: int, mode: str) -> dict[str, Any]:
    if seed not in {17, 29, 43} or mode not in {"smoke", "formal"}:
        raise ValueError("invalid frozen Task 9D seed or mode")
    values: dict[str, Any] = {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 192,
        "learning_rate": 0.0001,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 8,
        "logging_first_step": True,
        "logging_nan_inf_filter": False,
        "eval_strategy": "steps",
        "eval_steps": 64,
        "save_strategy": "steps",
        "save_steps": 192,
        "save_total_limit": 1,
        "load_best_model_at_end": False,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "label_names": ["labels"],
        "report_to": [],
        "seed": seed,
        "data_seed": seed,
    }
    if mode == "smoke":
        values.update({
            "max_steps": 3, "gradient_accumulation_steps": 1,
            "eval_strategy": "no", "save_strategy": "no",
            "logging_steps": 1, "dataloader_num_workers": 0,
        })
    return values


def final_checkpoint_rationale(step: int) -> dict[str, Any]:
    if step != 192:
        raise ValueError("formal Task 9D checkpoint must be fixed step 192")
    return {"selected_step": 192, "rule": "fixed_final_step_no_dev_selection", "early_stopping": False}


def validate_finite_history(history: Iterable[dict[str, Any]]) -> None:
    for row in history:
        for key in ("loss", "eval_loss", "grad_norm"):
            if key in row and not math.isfinite(float(row[key])):
                raise FloatingPointError(f"non-finite {key} at step {row.get('step')}")


def adapter_hash_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def role_exposure_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["role"]) for row in rows).items()))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_smoke_indices(envelopes: list[dict[str, Any]]) -> list[int]:
    selected: dict[str, int] = {}
    for index, row in enumerate(envelopes):
        selected.setdefault(str(row["role"]), index)
    if not {"positive", "semantic_negative"}.issubset(selected):
        raise ValueError("smoke input lacks required positive/semantic roles")
    return [selected[role] for role in sorted(selected)]


def run_training(config: dict[str, Any], variant: str, seed: int, mode: str) -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    from torch.utils.data import Subset
    from transformers import AutoProcessor, Trainer, TrainingArguments

    experiment_root = Path(config["experiment_root"])
    formal_dir = run_directory(experiment_root, variant, seed)
    output = formal_dir if mode == "formal" else experiment_root / "smoke" / variant / f"seed_{seed}"
    ensure_new_output(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        prepared = Path(config["prepared_root"])
        train_path = prepared / "variants" / variant / "train_schedule.jsonl"
        val_path = prepared / "variants" / variant / "val.jsonl"
        image_root = prepared
        train = Task9dDataset(train_path, image_root)
        val = Task9dDataset(val_path, image_root)
        exposure = role_exposure_counts(train.records)
        if mode == "formal" and len(train) != 1536:
            raise ValueError(f"formal Task 9D schedule must contain 1536 rows, found {len(train)}")
        train_data = train if mode == "formal" else Subset(train, _select_smoke_indices(train.records))
        val_data = val if mode == "formal" else None
        processor = AutoProcessor.from_pretrained(
            config["model_path"], min_pixels=200704, max_pixels=401408,
            use_fast=False, local_files_only=True,
        )
        collator = AssistantOnlyV2Collator(processor, max_length=1024)
        model, trainable, targets = build_task9d_model(config)
        arguments = task9d_training_arguments(seed=seed, mode=mode)
        arguments.update({"output_dir": str(output), "run_name": f"task9d_{variant}_{seed}_{mode}"})
        trainer = Trainer(
            model=model, args=TrainingArguments(**arguments), train_dataset=train_data,
            eval_dataset=val_data, data_collator=collator, processing_class=processor,
        )
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        validate_finite_history(trainer.state.log_history)
        adapter_dir = output / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter = adapter_dir / "adapter_model.safetensors"
        hash_report = adapter_hash_report(adapter)
        _write_json(output / "adapter.sha256.json", hash_report)
        processor.save_pretrained(output / "processor")
        trainer.save_state()
        summary = {
            "version": "task9d-run-summary-v1", "completed": True, "variant": variant,
            "seed": seed, "mode": mode, "elapsed_seconds": time.time() - started,
            "actual_training_rows": len(train_data), "role_exposure_counts": exposure,
            "optimizer_steps": int(trainer.state.global_step), "train_metrics": result.metrics,
            "checkpoint_selection": final_checkpoint_rationale(192) if mode == "formal" else "smoke_no_checkpoint",
            "adapter": hash_report, "trainable_parameters": trainable, "lora_targets": targets,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__, "accelerate": accelerate.__version__,
                            "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__},
            "log_history": trainer.state.log_history,
        }
        expected_steps = 3 if mode == "smoke" else 192
        if summary["optimizer_steps"] != expected_steps:
            raise RuntimeError(f"unexpected optimizer step count: {summary['optimizer_steps']}")
        if mode == "smoke" and summary["peak_vram_reserved_bytes"] >= 40 * 1024**3:
            raise RuntimeError("smoke peak VRAM exceeded 40 GiB gate")
        _write_json(output / "config.snapshot.json", config)
        _write_json(output / "run_summary.json", summary)
        _write_json(output / "status.json", {"state": "completed", "variant": variant, "seed": seed, "mode": mode})
        return summary
    except Exception as exc:
        _write_json(output / "failure.json", {
            "state": "failed", "variant": variant, "seed": seed, "mode": mode,
            "error": str(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    config = load_task9d_config(args.config)
    print(json.dumps(run_training(config, args.variant, args.seed, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
