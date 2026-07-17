import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10c_contract import CLASS_IDS, EXPECTED_MANIFEST_SHA256
from task10c_c2_contract import (
    C2_EXPOSURES,
    C2_STEPS,
    c2_training_arguments,
    checkpoint_path,
    validate_checkpoint_summary,
    verify_c2_protocol,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _row(class_id: int, index: int, split: str) -> dict:
    return {
        "id": f"{split}-{class_id:03d}-{index}",
        "class_id": class_id,
        "class_band": "head" if class_id in {16, 22, 24, 45, 50, 101} else (
            "medium" if class_id in {10, 68, 71, 87, 99} else "tail"
        ),
        "source_image_sha256": f"{class_id:03d}{index:061d}"[-64:],
        "near_duplicate_component_id": f"{split}-component-{class_id}-{index}",
        "model": {"messages": []},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _signed_protocol(tmp_path: Path, *, train_per_class: int = 4, overlap: bool = False) -> Path:
    root = tmp_path / "protocol"
    root.mkdir(parents=True)
    train = [_row(class_id, index, "train") for class_id in CLASS_IDS for index in range(train_per_class)]
    dev = [_row(class_id, 99, "dev") for class_id in CLASS_IDS]
    if overlap:
        dev[0]["near_duplicate_component_id"] = train[0]["near_duplicate_component_id"]
    _write_jsonl(root / "smoke_train.jsonl", train)
    _write_jsonl(root / "smoke_dev.jsonl", dev)
    files = {
        "preflight_report.json": {
            "passed": True,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "rows_by_split": {"dev": 80, "train": 192, "val": 48},
            "smoke_train_count": len(train),
            "smoke_dev_count": len(dev),
            "source_overlap": 0,
            "component_overlap": 0,
        },
        "config.snapshot.json": {"model_path": "/models/Qwen2___5-VL-3B-Instruct"},
        "status.json": {"state": "completed"},
    }
    for name, value in files.items():
        _write_json(root / name, value)
    names = list(files) + ["smoke_train.jsonl", "smoke_dev.jsonl"]
    (root / "completion.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    return root


def _checkpoint_summary(seed: int = 17, step: int = 8) -> dict:
    return {
        "completed": True,
        "seed": seed,
        "optimizer_steps": step,
        "actual_exposures": step * 8,
        "loss_reduction": "per_example_active_token_mean_then_batch_mean",
        "log_history": [{"step": step, "loss": 1.0, "grad_norm": 0.5}],
        "trainable_parameters": [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
        ],
        "adapter": {"sha256": "a" * 64, "bytes": 123},
        "authorize_reuse": False,
    }


def test_c2_contract_uses_same_smoke_train_and_exact_schedule(tmp_path):
    protocol = _signed_protocol(tmp_path)
    report = verify_c2_protocol(protocol)

    assert report["training_file"] == "smoke_train.jsonl"
    assert report["training_rows"] == 64
    assert C2_STEPS == (8, 16, 32, 64)
    assert C2_EXPOSURES == {8: 64, 16: 128, 32: 256, 64: 512}
    args = c2_training_arguments(17)
    assert args["max_steps"] == 64
    assert args["gradient_accumulation_steps"] == 8
    assert args["learning_rate"] == pytest.approx(1e-4)
    assert args["lr_scheduler_type"] == "cosine"
    assert args["warmup_ratio"] == pytest.approx(0.03)
    assert args["save_strategy"] == "no"


def test_c2_protocol_rejects_full_train_or_component_overlap(tmp_path):
    full = _signed_protocol(tmp_path / "full", train_per_class=12)
    with pytest.raises(ValueError, match="same 64-row smoke-train"):
        verify_c2_protocol(full)

    overlap = _signed_protocol(tmp_path / "overlap", overlap=True)
    with pytest.raises(ValueError, match="component overlap"):
        verify_c2_protocol(overlap)


def test_checkpoint_path_and_summary_are_exact(tmp_path):
    assert checkpoint_path(tmp_path, 8) == tmp_path / "checkpoints" / "step_008"
    with pytest.raises(ValueError, match="checkpoint step"):
        checkpoint_path(tmp_path, 12)

    assert validate_checkpoint_summary(_checkpoint_summary(), seed=17, step=8)["passed"] is True
    wrong = _checkpoint_summary(step=8)
    wrong["actual_exposures"] = 63
    with pytest.raises(ValueError, match="exposures"):
        validate_checkpoint_summary(wrong, seed=17, step=8)


def test_checkpoint_summary_rejects_nonfinite_unsafe_or_reusable():
    nonfinite = _checkpoint_summary()
    nonfinite["log_history"] = [{"step": 8, "loss": float("nan")}]
    with pytest.raises(FloatingPointError, match="non-finite"):
        validate_checkpoint_summary(nonfinite, seed=17, step=8)

    unsafe = _checkpoint_summary()
    unsafe["trainable_parameters"] = ["visual.merger.lora_A.default.weight"]
    with pytest.raises(ValueError, match="unsafe trainable"):
        validate_checkpoint_summary(unsafe, seed=17, step=8)

    reusable = _checkpoint_summary()
    reusable["authorize_reuse"] = True
    with pytest.raises(ValueError, match="reuse"):
        validate_checkpoint_summary(reusable, seed=17, step=8)
