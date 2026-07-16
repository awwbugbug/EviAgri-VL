import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from train_task9d_v22 import run_directory, validate_loss_audit_gate


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(schedule):
    roles = {}
    for role, share in (("positive", 0.334), ("semantic_negative", 0.334), ("visual_counterfactual", 0.332)):
        roles[role] = {
            "samples": 1, "active_tokens": {"min": 1, "max": 1, "mean": 1, "sum": 1},
            "mean_example_loss_weight": 1.0,
            "mean_optimizer_window_weight": 0.125,
            "total_gradient_weight": 0.125,
            "normalized_total_gradient_weight": share,
        }
    return {
        "passed": True,
        "reduction": "per_example_active_token_mean_then_batch_mean",
        "task8_locked_set_read": False,
        "training_started": False,
        "input_sha256": {"train_schedule": _sha(schedule)},
        "arms": {"Control": roles, "TaxMask": json.loads(json.dumps(roles))},
    }


def test_training_gate_rehashes_schedule_and_requires_equal_role_loss_mass(tmp_path):
    schedule = tmp_path / "train_schedule.jsonl"
    schedule.write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "loss_reduction_audit.json"
    audit.write_text(json.dumps(_report(schedule)), encoding="utf-8")
    result = validate_loss_audit_gate(audit, schedule)
    assert result["passed"] is True
    assert result["audit_sha256"] == _sha(audit)

    report = _report(schedule)
    report["arms"]["TaxMask"]["positive"]["normalized_total_gradient_weight"] = 0.3
    audit.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="gradient weight mismatch"):
        validate_loss_audit_gate(audit, schedule)


def test_training_gate_blocks_failed_or_stale_audit(tmp_path):
    schedule = tmp_path / "train_schedule.jsonl"
    schedule.write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "loss_reduction_audit.json"
    report = _report(schedule)
    report["passed"] = False
    audit.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        validate_loss_audit_gate(audit, schedule)

    report = _report(schedule)
    report["input_sha256"]["train_schedule"] = "0" * 64
    audit.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule hash mismatch"):
        validate_loss_audit_gate(audit, schedule)


def test_v22_run_directory_is_new_arm_seed_matrix(tmp_path):
    assert run_directory(tmp_path, "Control", 17) == tmp_path / "runs/Control/seed_17"
    assert run_directory(tmp_path, "TaxMask", 43) == tmp_path / "runs/TaxMask/seed_43"
    with pytest.raises(ValueError, match="frozen arm and seed"):
        run_directory(tmp_path, "Other", 17)
