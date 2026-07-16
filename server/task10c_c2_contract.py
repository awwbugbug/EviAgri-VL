"""Frozen contract for the Task 10C C2 64-step learning curve."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from task10c_contract import CLASS_IDS
from task10c_training import REDUCTION_CONTRACT, smoke_training_arguments
from task9d_model import reject_unsafe_trainables
from train_task10c_smoke import SEEDS, _verify_completion, verify_protocol_gate
from train_task9d import validate_finite_history


C2_STEPS = (8, 16, 32, 64)
C2_EXPOSURES = {step: step * 8 for step in C2_STEPS}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def c2_training_arguments(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"Task 10C C2 requires frozen seed: {seed}")
    values = smoke_training_arguments(seed)
    values.update({
        "max_steps": 64,
        "gradient_accumulation_steps": 8,
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 1,
    })
    return values


def verify_c2_protocol(protocol_root: str | Path) -> dict[str, Any]:
    root = Path(protocol_root)
    _verify_completion(root)
    train = _read_jsonl(root / "smoke_train.jsonl")
    dev = _read_jsonl(root / "smoke_dev.jsonl")
    if len(train) != 64:
        raise ValueError(
            f"C2 requires the same 64-row smoke-train, found {len(train)}"
        )
    if len(dev) != 16:
        raise ValueError(f"C2 requires the same 16-row smoke-dev, found {len(dev)}")
    gate = verify_protocol_gate(root)
    train_counts = Counter(int(row["class_id"]) for row in train)
    dev_counts = Counter(int(row["class_id"]) for row in dev)
    if train_counts != Counter({class_id: 4 for class_id in CLASS_IDS}):
        raise ValueError("C2 same 64-row smoke-train class quota mismatch")
    if dev_counts != Counter({class_id: 1 for class_id in CLASS_IDS}):
        raise ValueError("C2 same 16-row smoke-dev class quota mismatch")
    if len({str(row["id"]) for row in train}) != 64:
        raise ValueError("C2 smoke-train IDs must be unique")
    if len({str(row["source_image_sha256"]) for row in train}) != 64:
        raise ValueError("C2 smoke-train source SHA values must be unique")
    train_sha = {str(row["source_image_sha256"]) for row in train}
    dev_sha = {str(row["source_image_sha256"]) for row in dev}
    if train_sha & dev_sha:
        raise ValueError("C2 source SHA overlap between smoke-train and smoke-dev")
    train_components = {str(row["near_duplicate_component_id"]) for row in train}
    dev_components = {str(row["near_duplicate_component_id"]) for row in dev}
    if train_components & dev_components:
        raise ValueError("C2 component overlap between smoke-train and smoke-dev")
    return {
        **gate,
        "training_file": "smoke_train.jsonl",
        "training_rows": len(train),
        "smoke_dev_rows": len(dev),
    }


def checkpoint_path(training_root: str | Path, step: int) -> Path:
    if step not in C2_STEPS:
        raise ValueError(f"invalid C2 checkpoint step: {step}")
    return Path(training_root) / "checkpoints" / f"step_{step:03d}"


def validate_checkpoint_summary(
    summary: dict[str, Any],
    *,
    seed: int,
    step: int,
) -> dict[str, Any]:
    if seed not in SEEDS or summary.get("seed") != seed:
        raise ValueError("C2 checkpoint seed mismatch")
    if step not in C2_STEPS or int(summary.get("optimizer_steps", -1)) != step:
        raise ValueError("C2 checkpoint step mismatch")
    if summary.get("completed") is not True:
        raise ValueError("C2 checkpoint is incomplete")
    if int(summary.get("actual_exposures", -1)) != C2_EXPOSURES[step]:
        raise ValueError("C2 checkpoint exposures mismatch")
    if summary.get("loss_reduction") != REDUCTION_CONTRACT:
        raise ValueError("C2 checkpoint loss reduction mismatch")
    if summary.get("authorize_reuse") is not False:
        raise ValueError("C2 checkpoint must not authorize reuse")
    validate_finite_history(summary.get("log_history", []))
    try:
        reject_unsafe_trainables(summary.get("trainable_parameters", []))
    except ValueError as exc:
        raise ValueError(f"unsafe trainable parameters in C2 checkpoint: {exc}") from exc
    adapter = summary.get("adapter", {})
    if not _SHA256.fullmatch(str(adapter.get("sha256", ""))):
        raise ValueError("C2 checkpoint adapter SHA256 is invalid")
    if int(adapter.get("bytes", 0)) <= 0:
        raise ValueError("C2 checkpoint adapter byte count is invalid")
    return {"passed": True, "seed": seed, "step": step}
