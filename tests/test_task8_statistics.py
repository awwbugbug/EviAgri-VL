import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

import json

from evaluate_task8 import (
    bootstrap_ci,
    evaluate_experiment,
    exact_mcnemar,
    paired_bootstrap_delta,
)


def mean_value(rows):
    return sum(row["value"] for row in rows) / len(rows)


def test_bootstrap_ci_is_deterministic_and_family_level():
    rows = [
        {"family_id": family, "condition": condition, "value": value}
        for family, value in (("a", 0.0), ("b", 1.0), ("c", 1.0))
        for condition in ("original_correct", "blank_image")
    ]

    first = bootstrap_ci(rows, mean_value, repetitions=1000, seed=20260715)
    second = bootstrap_ci(list(reversed(rows)), mean_value, repetitions=1000, seed=20260715)

    assert first == second
    assert first["repetitions"] == 1000
    assert first["estimate"] == 2 / 3
    assert first["low"] <= first["estimate"] <= first["high"]


def test_paired_bootstrap_reports_model_minus_baseline_delta():
    baseline = [
        {"family_id": family, "condition": condition, "value": 0.0}
        for family in ("a", "b", "c")
        for condition in ("original_correct", "blank_image")
    ]
    model = [{**row, "value": 1.0} for row in baseline]

    result = paired_bootstrap_delta(
        baseline, model, mean_value, repetitions=1000, seed=20260715
    )

    assert result["delta_direction"] == "model_minus_baseline"
    assert result["estimate"] == 1.0
    assert result["low"] == 1.0
    assert result["high"] == 1.0


def test_paired_bootstrap_rejects_unpaired_families():
    baseline = [{"family_id": "a", "value": 0.0}]
    model = [{"family_id": "b", "value": 1.0}]

    try:
        paired_bootstrap_delta(baseline, model, mean_value)
    except ValueError as error:
        assert "family" in str(error)
    else:
        raise AssertionError("expected unpaired-family refusal")


def test_exact_mcnemar_uses_two_sided_binomial_tail():
    result = exact_mcnemar(
        baseline_correct=[True, False, False, False],
        model_correct=[False, True, True, True],
    )

    assert result["baseline_only_correct"] == 1
    assert result["model_only_correct"] == 3
    assert result["discordant"] == 4
    assert result["p_value"] == 0.625


def test_exact_mcnemar_no_discordance_is_one():
    result = exact_mcnemar([True, False], [True, False])

    assert result["discordant"] == 0
    assert result["p_value"] == 1.0


def _prediction(pest_id):
    return json.dumps(
        {
            "evidence_present": True,
            "evidence_bbox": [0, 0, 10, 10],
            "visible_attributes": [],
            "diagnosis": {"pest_id": pest_id, "pest_name": f"p{pest_id}"},
            "reliability": "supported",
        }
    )


def test_experiment_summary_registers_family_cis_and_paired_b1_b2_tests():
    groups = {"B1": [], "B2": []}
    for family, truth in (("f1", 1), ("f2", 2)):
        for group in groups:
            predicted = truth if group == "B2" or family == "f1" else 99
            groups[group].append(
                {
                    "job_id": f"{group}:{family}",
                    "audit_id": family,
                    "family_id": family,
                    "group": group,
                    "condition": "original_correct",
                    "query_pest_id": truth,
                    "actual_image_pest_id": truth,
                    "expected_evidence_present": True,
                    "gt_bbox": [0, 0, 10, 10],
                    "prediction": _prediction(predicted),
                }
            )

    result = evaluate_experiment(groups, repetitions=1000, seed=20260715)

    assert result["confidence_intervals"]["B2"]["positive.diagnosis_accuracy"]["repetitions"] == 1000
    paired = result["paired_b1_b2"]["original_correct.diagnosis_correct"]
    assert paired["mcnemar"]["model_only_correct"] == 1
    assert paired["bootstrap_delta"]["delta_direction"] == "model_minus_baseline"
    assert result["decision"]["value"] == "inconclusive"
