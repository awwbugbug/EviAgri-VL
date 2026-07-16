"""Fail-closed configuration validation for Task 9D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FROZEN = {
    "version": "task9d-static-qlora-v1",
    "seeds": [17, 29, 43],
    "variants": ["A", "B", "C"],
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
    "quantization": {"type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
    "vision": {"min_pixels": 200704, "max_pixels": 401408},
    "training": {
        "max_length": 1024,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 192,
        "learning_rate": 0.0001,
        "early_stopping": False,
        "eval_steps": [64, 128, 192],
    },
    "decoding": {"do_sample": False, "max_new_tokens": 512},
}


def load_task9d_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load Task 9D config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Task 9D config must be an object")
    allowed = set(FROZEN) | {"model_path", "prepared_root", "experiment_root"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown Task 9D config keys: {sorted(unknown)}")
    for key, expected in FROZEN.items():
        if value.get(key) != expected:
            raise ValueError(f"frozen Task 9D config mismatch at {key}")
    model_path = value.get("model_path")
    if not isinstance(model_path, str) or not model_path.endswith("Qwen2___5-VL-3B-Instruct"):
        raise ValueError("frozen Task 9D backbone must be Qwen2.5-VL-3B-Instruct")
    return value
