"""Train one independent Task 10C C2 seed and save four adapter checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from task10c_c2_contract import (
    C2_EXPOSURES,
    C2_STEPS,
    c2_training_arguments,
    checkpoint_path,
    validate_checkpoint_summary,
    verify_c2_protocol,
)
from task10c_training import (
    DiagnosisOnlyCollator,
    REDUCTION_CONTRACT,
    Task10CDataset,
    Task10CTrainer,
)
from task9d_model import build_task9d_model, reject_unsafe_trainables
from train_task10c_smoke import SEEDS
from train_task9d import adapter_hash_report, ensure_new_output, validate_finite_history


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_completion(root: Path, names: list[str]) -> None:
    (root / "completion.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def training_directory(experiment_root: str | Path, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Task 10C C2 requires frozen seed: {seed}")
    return Path(experiment_root) / "training" / f"seed_{seed}"


def save_c2_checkpoint(
    *,
    model,
    state,
    training_root: str | Path,
    seed: int,
    step: int,
    trainable_parameters: list[str],
    protocol_manifest_sha256: str,
) -> dict[str, Any]:
    training_root = Path(training_root)
    output = checkpoint_path(training_root, step)
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite C2 checkpoint: {output}")
    temporary.mkdir(parents=True)
    try:
        adapter_dir = temporary / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        raw_report = adapter_hash_report(adapter_file)
        adapter_report = {
            **raw_report,
            "path": str(output / "adapter" / "adapter_model.safetensors"),
        }
        history = [
            dict(row)
            for row in getattr(state, "log_history", [])
            if int(row.get("step", step)) <= step
        ]
        summary = {
            "version": "task10c-c2-checkpoint-v1",
            "completed": True,
            "seed": seed,
            "optimizer_steps": step,
            "actual_exposures": C2_EXPOSURES[step],
            "loss_reduction": REDUCTION_CONTRACT,
            "log_history": history,
            "trainable_parameters": list(trainable_parameters),
            "adapter": adapter_report,
            "protocol_manifest_sha256": protocol_manifest_sha256,
            "checkpoint_role": "learning_curve_observation_not_selection",
            "authorize_reuse": False,
        }
        validate_checkpoint_summary(summary, seed=seed, step=step)
        _write_json(temporary / "adapter.sha256.json", adapter_report)
        _write_json(temporary / "checkpoint_summary.json", summary)
        state.save_to_json(str(temporary / "trainer_state.json"))
        _write_json(temporary / "status.json", {
            "state": "completed", "seed": seed, "optimizer_steps": step,
        })
        _write_completion(temporary, [
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
            "adapter.sha256.json",
            "checkpoint_summary.json",
            "trainer_state.json",
            "status.json",
        ])
        temporary.rename(output)
        return summary
    except Exception as exc:
        _write_json(temporary / "failure.json", {
            "state": "failed",
            "seed": seed,
            "optimizer_steps": step,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


class C2CheckpointCallback(TrainerCallback):
    def __init__(
        self,
        training_root: str | Path,
        *,
        seed: int,
        trainable_parameters: list[str],
        protocol_manifest_sha256: str,
    ):
        self.training_root = Path(training_root)
        self.seed = seed
        self.trainable_parameters = list(trainable_parameters)
        self.protocol_manifest_sha256 = protocol_manifest_sha256
        self.saved: dict[int, dict[str, Any]] = {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        del args, kwargs
        step = int(state.global_step)
        if step in C2_STEPS and step not in self.saved:
            self.saved[step] = save_c2_checkpoint(
                model=model,
                state=state,
                training_root=self.training_root,
                seed=self.seed,
                step=step,
                trainable_parameters=self.trainable_parameters,
                protocol_manifest_sha256=self.protocol_manifest_sha256,
            )
        return control


def validate_c2_run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("completed") is not True:
        raise ValueError("C2 training summary is incomplete")
    seed = summary.get("seed")
    if seed not in SEEDS:
        raise ValueError("C2 training summary seed mismatch")
    if int(summary.get("optimizer_steps", -1)) != 64:
        raise ValueError("C2 training must finish exactly 64 optimizer steps")
    if int(summary.get("actual_exposures", -1)) != 512:
        raise ValueError("C2 training must contain exactly 512 exposures")
    if int(summary.get("training_rows", -1)) != 64:
        raise ValueError("C2 training must use the same 64-row smoke-train")
    if summary.get("continued_from_c1") is not False:
        raise ValueError("C2 training must not be continued from C1")
    if summary.get("loss_reduction") != REDUCTION_CONTRACT:
        raise ValueError("C2 training loss reduction mismatch")
    checkpoints = summary.get("checkpoints", {})
    if {int(key) for key in checkpoints} != set(C2_STEPS):
        raise ValueError("C2 training checkpoint set mismatch")
    validate_finite_history(summary.get("log_history", []))
    try:
        reject_unsafe_trainables(summary.get("trainable_parameters", []))
    except ValueError as exc:
        raise ValueError(f"unsafe trainable parameters in C2 training: {exc}") from exc
    for step in C2_STEPS:
        checkpoint = checkpoints[str(step)]
        if int(checkpoint.get("optimizer_steps", -1)) != step:
            raise ValueError(f"C2 checkpoint {step} summary mismatch")
        if int(checkpoint.get("actual_exposures", -1)) != C2_EXPOSURES[step]:
            raise ValueError(f"C2 checkpoint {step} exposures mismatch")
        adapter = checkpoint.get("adapter", {})
        if len(str(adapter.get("sha256", ""))) != 64 or int(adapter.get("bytes", 0)) <= 0:
            raise ValueError(f"C2 checkpoint {step} adapter report mismatch")
    return {"passed": True, "seed": seed}


def run_c2_training(
    *,
    protocol_root: str | Path,
    model_path: str | Path,
    experiment_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import AutoProcessor, TrainingArguments, set_seed

    protocol_root, model_path = Path(protocol_root), Path(model_path)
    output = training_directory(experiment_root, seed)
    gate = verify_c2_protocol(protocol_root)
    config = json.loads((protocol_root / "config.snapshot.json").read_text(encoding="utf-8"))
    if Path(config["model_path"]).resolve() != model_path.resolve():
        raise ValueError("C2 model path differs from signed protocol")
    ensure_new_output(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "status.json", {
        "state": "running", "seed": seed, "stage": "training", "expected_steps": 64,
    })
    started = time.time()
    try:
        dataset = Task10CDataset(protocol_root / "smoke_train.jsonl")
        if len(dataset) != 64:
            raise ValueError("C2 must train on the same 64-row smoke-train")
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=200704,
            max_pixels=401408,
            use_fast=False,
            local_files_only=True,
        )
        collator = DiagnosisOnlyCollator(processor, max_length=1024)
        set_seed(seed)
        model, trainable, targets = build_task9d_model({"model_path": str(model_path)})
        reject_unsafe_trainables(trainable)
        argument_values = c2_training_arguments(seed)
        reduction = argument_values.pop("reduction_contract")
        argument_values.update({
            "output_dir": str(output / "trainer_runtime"),
            "run_name": f"task10c_c2_seed_{seed}",
        })
        callback = C2CheckpointCallback(
            output,
            seed=seed,
            trainable_parameters=trainable,
            protocol_manifest_sha256=gate["manifest_sha256"],
        )
        trainer = Task10CTrainer(
            model=model,
            args=TrainingArguments(**argument_values),
            train_dataset=dataset,
            data_collator=collator,
            processing_class=processor,
            callbacks=[callback],
        )
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        validate_finite_history(trainer.state.log_history)
        if int(trainer.state.global_step) != 64:
            raise ValueError("C2 must finish exactly 64 optimizer steps")
        if set(callback.saved) != set(C2_STEPS):
            raise ValueError("C2 checkpoint set mismatch")
        trainer.state.save_to_json(str(output / "trainer_state.json"))
        processor.save_pretrained(output / "processor")
        summary = {
            "version": "task10c-c2-training-summary-v1",
            "completed": True,
            "seed": seed,
            "optimizer_steps": int(trainer.state.global_step),
            "actual_exposures": int(trainer.state.global_step) * 8,
            "training_rows": len(dataset),
            "continued_from_c1": False,
            "elapsed_seconds": time.time() - started,
            "train_metrics": result.metrics,
            "loss_reduction": reduction,
            "checkpoint_selection": "fixed_step_64_no_selection",
            "checkpoints": {str(step): callback.saved[step] for step in C2_STEPS},
            "trainable_parameters": trainable,
            "lora_targets": targets,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
                "peft": peft.__version__,
                "bitsandbytes": bitsandbytes.__version__,
            },
            "log_history": trainer.state.log_history,
            "protocol_gate": gate,
            "task8_locked_content_read": False,
            "authorize_next_experiment": False,
        }
        validate_c2_run_summary(summary)
        _write_json(output / "config.snapshot.json", {
            "version": "task10c-c2-training-config-v1",
            "seed": seed,
            "model_path": str(model_path),
            "protocol_root": str(protocol_root),
            "training_file": "smoke_train.jsonl",
            "training_arguments": argument_values,
            "loss_reduction": reduction,
            "checkpoint_steps": list(C2_STEPS),
        })
        _write_json(output / "run_summary.json", summary)
        _write_json(output / "status.json", {
            "state": "completed", "seed": seed, "stage": "training", "optimizer_steps": 64,
        })
        _write_completion(output, [
            "config.snapshot.json", "run_summary.json", "trainer_state.json", "status.json",
        ])
        return summary
    except Exception as exc:
        _write_json(output / "failure.json", {
            "state": "failed",
            "seed": seed,
            "stage": "training",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        })
        _write_json(output / "status.json", {
            "state": "failed", "seed": seed, "stage": "training",
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Task 10C C2 64-step seed")
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    print(json.dumps(run_c2_training(
        protocol_root=args.protocol_root,
        model_path=args.model_path,
        experiment_root=args.experiment_root,
        seed=args.seed,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
