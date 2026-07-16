"""Train one frozen arm/seed of the Task 9D v2.2 micro validation."""

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
from typing import Any

from task9d_config import load_task9d_config
from task9d_data import Task9dDataset
from task9d_model import build_task9d_model
from task9d_v22_training import (
    REDUCTION_CONTRACT,
    V22LossCollator,
    V22PerExampleMeanTrainer,
    v22_training_arguments,
)
from train_task9d import adapter_hash_report, ensure_new_output, validate_finite_history


ARMS = {"Control", "TaxMask"}
SEEDS = {17, 29, 43}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_directory(experiment_root: str | Path, arm: str, seed: int) -> Path:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("v2.2 run must use frozen arm and seed")
    return Path(experiment_root) / "runs" / arm / f"seed_{seed}"


def validate_loss_audit_gate(audit_path: str | Path, schedule_path: str | Path) -> dict[str, Any]:
    audit_path, schedule_path = Path(audit_path), Path(schedule_path)
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("v2.2 loss reduction audit did not pass")
    if report.get("reduction") != REDUCTION_CONTRACT:
        raise ValueError("v2.2 loss reduction contract mismatch")
    if report.get("task8_locked_set_read") is not False or report.get("training_started") is not False:
        raise ValueError("v2.2 loss audit provenance is unsafe")
    schedule_hash = _sha256(schedule_path)
    if report.get("input_sha256", {}).get("train_schedule") != schedule_hash:
        raise ValueError("v2.2 loss audit schedule hash mismatch")
    arms = report.get("arms", {})
    if set(arms) != ARMS:
        raise ValueError("v2.2 loss audit arm set mismatch")
    control, taxmask = arms["Control"], arms["TaxMask"]
    if set(control) != set(taxmask):
        raise ValueError("v2.2 loss audit role set mismatch")
    for role in sorted(control):
        left, right = control[role], taxmask[role]
        if int(left["samples"]) != int(right["samples"]):
            raise ValueError(f"v2.2 role sample count mismatch for {role}")
        if abs(float(left["mean_example_loss_weight"]) - float(right["mean_example_loss_weight"])) > 1e-12:
            raise ValueError(f"v2.2 example loss weight mismatch for {role}")
        if abs(
            float(left["normalized_total_gradient_weight"])
            - float(right["normalized_total_gradient_weight"])
        ) > 1e-12:
            raise ValueError(f"v2.2 gradient weight mismatch for {role}")
    return {
        "passed": True,
        "audit_sha256": _sha256(audit_path),
        "schedule_sha256": schedule_hash,
        "reduction": REDUCTION_CONTRACT,
    }


def run_training(
    *,
    base_config_path: Path,
    protocol_root: Path,
    image_root: Path,
    experiment_root: Path,
    audit_path: Path,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    from transformers import AutoProcessor, AutoTokenizer, TrainingArguments, set_seed

    output = run_directory(experiment_root, arm, seed)
    schedule_path = protocol_root / "train_schedule.jsonl"
    audit_gate = validate_loss_audit_gate(audit_path, schedule_path)
    ensure_new_output(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        config = load_task9d_config(base_config_path)
        train = Task9dDataset(schedule_path, image_root)
        val = Task9dDataset(protocol_root / "val.jsonl", image_root)
        roles = Counter(str(row["role"]) for row in train.records)
        if len(train) != 512 or len(val) != 96:
            raise ValueError(f"v2.2 frozen row count mismatch: train={len(train)} val={len(val)}")
        if set(roles) != {"positive", "semantic_negative", "visual_counterfactual"}:
            raise ValueError(f"v2.2 frozen role set mismatch: {dict(roles)}")
        processor = AutoProcessor.from_pretrained(
            config["model_path"], min_pixels=200704, max_pixels=401408,
            use_fast=False, local_files_only=True,
        )
        fast = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True, local_files_only=True)
        collator = V22LossCollator(
            processor, 1024, arm=arm, fast_tokenizer=fast,
        )
        set_seed(seed)
        model, trainable, targets = build_task9d_model(config)
        argument_values = v22_training_arguments(seed=seed)
        reduction = argument_values.pop("reduction_contract")
        argument_values.update({
            "output_dir": str(output),
            "run_name": f"task9d_v22_{arm}_{seed}",
        })
        trainer = V22PerExampleMeanTrainer(
            model=model,
            args=TrainingArguments(**argument_values),
            train_dataset=train,
            eval_dataset=val,
            data_collator=collator,
            processing_class=processor,
        )
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        validate_finite_history(trainer.state.log_history)
        if int(trainer.state.global_step) != 64:
            raise RuntimeError(f"unexpected v2.2 optimizer steps: {trainer.state.global_step}")
        adapter_dir = output / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter = adapter_dir / "adapter_model.safetensors"
        adapter_report = adapter_hash_report(adapter)
        _write_json(output / "adapter.sha256.json", adapter_report)
        processor.save_pretrained(output / "processor")
        trainer.save_state()
        summary = {
            "version": "task9d-v22-run-summary-v1",
            "completed": True,
            "arm": arm,
            "seed": seed,
            "elapsed_seconds": time.time() - started,
            "actual_training_rows": len(train),
            "role_exposure_counts": dict(sorted(roles.items())),
            "optimizer_steps": int(trainer.state.global_step),
            "train_metrics": result.metrics,
            "checkpoint_selection": {
                "selected_step": 64,
                "rule": "fixed_final_step_no_dev_selection",
                "early_stopping": False,
            },
            "loss_reduction": reduction,
            "loss_audit_gate": audit_gate,
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
            "task8_locked_set_read": False,
        }
        _write_json(output / "config.snapshot.json", {
            **config,
            "v22": {
                "arm": arm,
                "seed": seed,
                "protocol_root": str(protocol_root),
                "image_root": str(image_root),
                "experiment_root": str(experiment_root),
                "loss_audit": str(audit_path),
                "loss_reduction": reduction,
            },
        })
        _write_json(output / "run_summary.json", summary)
        _write_json(output / "status.json", {"state": "completed", "arm": arm, "seed": seed})
        completion_names = ["adapter.sha256.json", "config.snapshot.json", "run_summary.json", "status.json"]
        (output / "completion.sha256").write_text(
            "".join(f"{_sha256(output / name)}  {name}\n" for name in completion_names),
            encoding="utf-8",
        )
        return summary
    except Exception as exc:
        _write_json(output / "failure.json", {
            "state": "failed", "arm": arm, "seed": seed,
            "error": str(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--loss-audit", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=sorted(SEEDS), required=True)
    args = parser.parse_args()
    result = run_training(
        base_config_path=args.base_config,
        protocol_root=args.protocol_root,
        image_root=args.image_root,
        experiment_root=args.experiment_root,
        audit_path=args.loss_audit,
        arm=args.arm,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
