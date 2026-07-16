import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task10c_smoke import (
    condition_metrics,
    decide_c1_engineering,
)
from task10c_contract import CLASS_IDS


def _valid_report(seed: int) -> dict:
    return {
        "seed": seed,
        "optimizer_steps": 8,
        "actual_exposures": 64,
        "prediction_count": 64,
        "condition_counts": {
            "image_train_prompt": 16,
            "image_unseen_prompt": 16,
            "no_image_train_prompt": 16,
            "no_image_unseen_prompt": 16,
        },
        "completion_verified": True,
        "adapter_reload_verified": True,
        "observations": {
            "image_train_prompt": {"macro_f1": 0.0},
            "image_unseen_prompt": {"macro_f1": 0.0},
            "no_image_train_prompt": {"macro_f1": 0.0},
            "no_image_unseen_prompt": {"macro_f1": 0.0},
        },
    }


def test_condition_metrics_uses_all_sixteen_classes_and_strict_validity():
    predictions = []
    for class_id in CLASS_IDS:
        predictions.append({
            "class_id": class_id,
            "parsed": {
                "syntax_valid": True,
                "schema_valid": True,
                "pest_id": f"IP{class_id:03d}" if class_id == CLASS_IDS[0] else "IP009",
                "error": None,
            },
        })

    metrics = condition_metrics(predictions)

    assert metrics["count"] == 16
    assert metrics["syntax_validity"] == 1.0
    assert metrics["schema_validity"] == 1.0
    assert metrics["accuracy"] == 1 / 16
    assert metrics["unique_predicted_ids"] == 1
    assert 0 < metrics["macro_f1"] < 1 / 16


def test_condition_metrics_counts_invalid_json_as_incorrect():
    predictions = [{
        "class_id": class_id,
        "parsed": {
            "syntax_valid": False,
            "schema_valid": False,
            "pest_id": None,
            "error": "invalid_json",
        },
    } for class_id in CLASS_IDS]

    metrics = condition_metrics(predictions)

    assert metrics["syntax_validity"] == 0.0
    assert metrics["schema_validity"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_c1_requires_all_three_exact_runs_and_never_authorizes_execution():
    reports = {seed: _valid_report(seed) for seed in (17, 29, 43)}

    decision = decide_c1_engineering(reports)

    assert decision["status"] == "PASS_C1_ENGINEERING"
    assert decision["authorize_task10c_c2_execution"] is False
    assert decision["requires_user_approval_for_c2"] is True

    reports[29]["prediction_count"] = 63
    failed = decide_c1_engineering(reports)
    assert failed["status"] == "FAIL_C1_ENGINEERING"
    assert any("prediction count" in reason for reason in failed["reasons"])


def test_smoke_performance_is_observed_but_never_used_as_engineering_gate():
    reports = {seed: _valid_report(seed) for seed in (17, 29, 43)}
    for report in reports.values():
        for metrics in report["observations"].values():
            metrics["macro_f1"] = 0.0

    decision = decide_c1_engineering(reports)

    assert decision["status"] == "PASS_C1_ENGINEERING"
    assert not any("macro" in reason.lower() for reason in decision["reasons"])


def test_c1_fails_closed_on_missing_seed_or_integrity_flag():
    reports = {seed: _valid_report(seed) for seed in (17, 29)}
    missing = decide_c1_engineering(reports)
    assert missing["status"] == "FAIL_C1_ENGINEERING"
    assert any("seed set" in reason for reason in missing["reasons"])

    reports[43] = _valid_report(43)
    reports[17]["adapter_reload_verified"] = False
    failed = decide_c1_engineering(reports)
    assert any("integrity" in reason for reason in failed["reasons"])
