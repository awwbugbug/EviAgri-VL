import json
from pathlib import Path
from typing import Any


TOP_LEVEL_KEYS = (
    "seed",
    "model_path",
    "source_data_root",
    "mixed_data_root",
    "experiment_root",
    "lora",
    "quantization",
    "vision",
    "data",
    "training",
)

NESTED_KEYS = {
    "lora": ("r", "alpha", "dropout"),
    "quantization": ("type", "double_quant", "compute_dtype"),
    "vision": ("min_pixels", "max_pixels"),
    "data": ("train_positive", "train_null"),
    "training": (
        "max_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
    ),
}


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], section: str) -> None:
    for key in keys:
        if key not in mapping:
            prefix = f"{section}." if section else ""
            raise ValueError(f"missing required config key: {prefix}{key}")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def load_training_config(path: Path) -> dict[str, Any]:
    """Load and validate the shared Static QLoRA v1 configuration."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load training config {path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError("training config must be a JSON object")
    _require_keys(config, TOP_LEVEL_KEYS, "")
    for section, keys in NESTED_KEYS.items():
        if not isinstance(config[section], dict):
            raise ValueError(f"config section {section} must be an object")
        _require_keys(config[section], keys, section)

    for key in ("model_path", "source_data_root", "mixed_data_root", "experiment_root"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty path string")

    _require_positive_int(config["seed"], "seed")
    _require_positive_int(config["lora"]["r"], "lora.r")
    _require_positive_int(config["lora"]["alpha"], "lora.alpha")
    dropout = config["lora"]["dropout"]
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("lora.dropout must be in [0, 1)")

    if config["quantization"] != {
        "type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    }:
        raise ValueError("quantization must use NF4, double quantization, and bfloat16 compute")

    for key in ("min_pixels", "max_pixels"):
        _require_positive_int(config["vision"][key], f"vision.{key}")
    if config["vision"]["min_pixels"] > config["vision"]["max_pixels"]:
        raise ValueError("vision.min_pixels must not exceed vision.max_pixels")

    positive = config["data"]["train_positive"]
    null = config["data"]["train_null"]
    _require_positive_int(positive, "data.train_positive")
    _require_positive_int(null, "data.train_null")
    if positive != 2 * null:
        raise ValueError("training data must preserve the approved positive:null ratio of 2:1")

    training = config["training"]
    for key in (
        "max_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
    ):
        _require_positive_int(training[key], f"training.{key}")
    for key in ("learning_rate", "max_grad_norm"):
        if not isinstance(training[key], (int, float)) or training[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("warmup_ratio", "weight_decay"):
        if not isinstance(training[key], (int, float)) or not 0 <= training[key] < 1:
            raise ValueError(f"training.{key} must be in [0, 1)")

    return config
