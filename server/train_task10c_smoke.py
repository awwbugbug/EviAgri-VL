"""Run one frozen eight-step Task 10C diagnosis-only QLoRA smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import traceback
from pathlib import Path
from typing import Any

from task10c_contract import EXPECTED_MANIFEST_SHA256
from task10c_training import (
    REDUCTION_CONTRACT,
    DiagnosisOnlyCollator,
    Task10CDataset,
    Task10CTrainer,
    smoke_training_arguments,
)
from task9d_model import build_task9d_model, reject_unsafe_trainables
from train_task9d import adapter_hash_report, ensure_new_output, validate_finite_history


SEEDS = (17, 29, 43)


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


def _verify_completion(root: Path) -> None:
    completion = root / "completion.sha256"
    if not completion.is_file():
        raise ValueError(f"missing completion SHA256: {completion}")
    for line in completion.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = root / name
        if not path.is_file() or _sha256(path) != expected.lower():
            raise ValueError(f"completion SHA256 mismatch: {name}")


def run_directory(root: str | Path, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Task 10C smoke requires frozen seed: {seed}")
    return Path(root) / "training" / f"seed_{seed}"


def verify_protocol_gate(protocol_root: str | Path) -> dict[str, Any]:
    root = Path(protocol_root)
    _verify_completion(root)
    report = json.loads((root / "preflight_report.json").read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("Task 10C protocol gate is not passed")
    if report.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Task 10C protocol manifest SHA mismatch")
    if report.get("rows_by_split") != {"dev": 80, "train": 192, "val": 48}:
        raise ValueError("Task 10C protocol split mismatch")
    if report.get("smoke_train_count") != 64 or report.get("smoke_dev_count") != 16:
        raise ValueError("Task 10C smoke quota mismatch")
    if report.get("source_overlap") != 0 or report.get("component_overlap") != 0:
        raise ValueError("Task 10C protocol overlap gate failed")
    return report


def validate_smoke_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("completed") is not True:
        raise ValueError("Task 10C smoke summary is incomplete")
    if summary.get("seed") not in SEEDS:
        raise ValueError("Task 10C smoke summary has invalid seed")
    if int(summary.get("optimizer_steps", -1)) != 8:
        raise ValueError("Task 10C smoke optimizer steps must equal 8")
    if int(summary.get("actual_exposures", -1)) != 64:
        raise ValueError("Task 10C smoke exposures must equal 64")
    if summary.get("loss_reduction") != REDUCTION_CONTRACT:
        raise ValueError("Task 10C smoke loss reduction mismatch")
    validate_finite_history(summary.get("log_history", []))
    trainables = summary.get("trainable_parameters", [])
    try:
        reject_unsafe_trainables(trainables)
    except ValueError as exc:
        raise ValueError(f"unsafe trainable parameters in Task 10C smoke: {exc}") from exc
    adapter = summary.get("adapter", {})
    if not isinstance(adapter.get("sha256"), str) or len(adapter["sha256"]) != 64:
        raise ValueError("Task 10C smoke adapter SHA is invalid")
    for row in summary.get("log_history", []):
        for key in ("loss", "grad_norm", "eval_loss"):
            if key in row and not math.isfinite(float(row[key])):
                raise FloatingPointError(f"non-finite {key} at step {row.get('step')}")
    return {"passed": True, "seed": summary["seed"]}


def _write_completion(output: Path, names: list[str]) -> None:
    (output / "completion.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def run_smoke_training(
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
    output = run_directory(experiment_root, seed)
    gate = verify_protocol_gate(protocol_root)
    config = json.loads((protocol_root / "config.snapshot.json").read_text(encoding="utf-8"))
    if Path(config["model_path"]).resolve() != model_path.resolve():
        raise ValueError("Task 10C model path differs from signed protocol")
    ensure_new_output(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "status.json", {"state": "running", "seed": seed, "stage": "training"})
    started = time.time()
    try:
        dataset = Task10CDataset(protocol_root / "smoke_train.jsonl")
        if len(dataset) != 64:
            raise ValueError(f"Task 10C smoke train must contain 64 rows, found {len(dataset)}")
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
        argument_values = smoke_training_arguments(seed)
        reduction = argument_values.pop("reduction_contract")
        argument_values.update({
            "output_dir": str(output),
            "run_name": f"task10c_smoke_seed_{seed}",
        })
        trainer = Task10CTrainer(
            model=model,
            args=TrainingArguments(**argument_values),
            train_dataset=dataset,
            data_collator=collator,
            processing_class=processor,
        )
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        validate_finite_history(trainer.state.log_history)
        optimizer_steps = int(trainer.state.global_step)
        actual_exposures = optimizer_steps * 8
        adapter_dir = output / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter = adapter_dir / "adapter_model.safetensors"
        adapter_report = adapter_hash_report(adapter)
        _write_json(output / "adapter.sha256.json", adapter_report)
        processor.save_pretrained(output / "processor")
        trainer.save_state()
        summary = {
            "version": "task10c-smoke-training-summary-v1",
            "completed": True,
            "seed": seed,
            "elapsed_seconds": time.time() - started,
            "actual_training_rows": len(dataset),
            "actual_exposures": actual_exposures,
            "optimizer_steps": optimizer_steps,
            "train_metrics": result.metrics,
            "loss_reduction": reduction,
            "checkpoint_selection": "smoke_final_step_not_reusable_for_c2",
            "adapter": adapter_report,
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
            "authorize_c2_reuse": False,
        }
        validate_smoke_summary(summary)
        _write_json(output / "config.snapshot.json", {
            "version": "task10c-smoke-training-config-v1",
            "seed": seed,
            "model_path": str(model_path),
            "protocol_root": str(protocol_root),
            "training_arguments": argument_values,
            "loss_reduction": reduction,
        })
        _write_json(output / "run_summary.json", summary)
        _write_json(output / "status.json", {
            "state": "completed", "seed": seed, "stage": "training",
            "optimizer_steps": optimizer_steps,
        })
        names = [
            "adapter.sha256.json", "config.snapshot.json", "run_summary.json",
            "trainer_state.json", "status.json",
        ]
        _write_completion(output, names)
        return summary
    except Exception as exc:
        _write_json(output / "failure.json", {
            "state": "failed", "seed": seed, "stage": "training",
            "error": str(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        })
        _write_json(output / "status.json", {"state": "failed", "seed": seed, "stage": "training"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Task 10C eight-step smoke")
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke_training(
        protocol_root=args.protocol_root,
        model_path=args.model_path,
        experiment_root=args.experiment_root,
        seed=args.seed,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
