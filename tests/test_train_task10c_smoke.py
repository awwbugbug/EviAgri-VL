import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10c_contract import EXPECTED_MANIFEST_SHA256
from train_task10c_smoke import (
    run_directory,
    validate_smoke_summary,
    verify_protocol_gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _signed_protocol(tmp_path: Path) -> Path:
    root = tmp_path / "protocol"
    root.mkdir()
    files = {
        "preflight_report.json": {
            "passed": True,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "rows_by_split": {"dev": 80, "train": 192, "val": 48},
            "smoke_train_count": 64,
            "smoke_dev_count": 16,
            "source_overlap": 0,
            "component_overlap": 0,
        },
        "config.snapshot.json": {"model_path": "/models/Qwen2___5-VL-3B-Instruct"},
        "model_files.sha256.json": [{"name": "model.safetensors", "sha256": "a" * 64}],
        "status.json": {"state": "completed"},
    }
    for name, value in files.items():
        _write_json(root / name, value)
    (root / "smoke_train.jsonl").write_text("{}\n" * 64, encoding="utf-8")
    (root / "smoke_dev.jsonl").write_text("{}\n" * 16, encoding="utf-8")
    names = list(files) + ["smoke_train.jsonl", "smoke_dev.jsonl"]
    (root / "completion.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    return root


def _summary(seed=29, steps=8, exposures=64):
    return {
        "seed": seed,
        "optimizer_steps": steps,
        "actual_exposures": exposures,
        "loss_reduction": "per_example_active_token_mean_then_batch_mean",
        "trainable_parameters": [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
        ],
        "log_history": [{"loss": 1.0, "grad_norm": 0.5, "step": 1}],
        "adapter": {"sha256": "f" * 64, "bytes": 123},
        "completed": True,
    }


def test_run_directory_and_seed_are_frozen(tmp_path):
    assert run_directory(tmp_path, 17) == tmp_path / "training" / "seed_17"
    with pytest.raises(ValueError, match="frozen seed"):
        run_directory(tmp_path, 31)


def test_protocol_gate_rehashes_inputs_and_checks_exact_contract(tmp_path):
    protocol = _signed_protocol(tmp_path)
    gate = verify_protocol_gate(protocol)

    assert gate["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert gate["smoke_train_count"] == 64

    (protocol / "smoke_train.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completion SHA256"):
        verify_protocol_gate(protocol)


def test_validate_summary_requires_exact_steps_exposures_and_safe_trainables():
    assert validate_smoke_summary(_summary())["passed"] is True

    wrong_steps = _summary(steps=7)
    with pytest.raises(ValueError, match="optimizer steps"):
        validate_smoke_summary(wrong_steps)

    wrong_exposures = _summary(exposures=63)
    with pytest.raises(ValueError, match="exposures"):
        validate_smoke_summary(wrong_exposures)

    unsafe = _summary()
    unsafe["trainable_parameters"] = ["visual.merger.lora_A.default.weight"]
    with pytest.raises(ValueError, match="unsafe trainable"):
        validate_smoke_summary(unsafe)


def test_validate_summary_rejects_nonfinite_history_and_wrong_reduction():
    nonfinite = _summary()
    nonfinite["log_history"] = [{"loss": float("nan"), "step": 1}]
    with pytest.raises(FloatingPointError, match="non-finite"):
        validate_smoke_summary(nonfinite)

    wrong = _summary()
    wrong["loss_reduction"] = "token_global_mean"
    with pytest.raises(ValueError, match="loss reduction"):
        validate_smoke_summary(wrong)
