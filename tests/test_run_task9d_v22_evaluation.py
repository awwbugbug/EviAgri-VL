import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from run_task9d_v22_evaluation import aggregate_group_metrics, paired_evidence_statistics


def _row(family, positive, prediction):
    return {
        "id": f"{family}-{positive}", "family_id": family,
        "positive": positive, "primary_positive": positive,
        "role": "positive" if positive else "semantic_negative",
        "evidence_prediction": prediction,
        "positive_correct": bool(positive and prediction is True),
        "null_fp": bool(not positive and prediction is not False),
    }


def _metrics(ba, tpr, fpr, rows):
    return {
        "evidence": {"balanced_accuracy": ba, "positive_tpr": tpr, "overall_null_fpr": fpr,
                     "semantic_null_fpr": fpr, "visual_null_fpr": fpr},
        "json": {"syntax_validity": 1.0, "schema_validity": 1.0},
        "evidence_semantic_consistency": 1.0, "evidence_task_compliance": 1.0,
        "prompt_gap": 0.0, "row_outcomes": rows,
    }


def test_paired_statistics_resample_families_and_measure_evidence_delta():
    control_rows = [_row("f1", True, False), _row("f1", False, False)]
    taxmask_rows = [_row("f1", True, True), _row("f1", False, False)]
    result = paired_evidence_statistics(
        _metrics(0.5, 0.0, 0.0, control_rows),
        _metrics(1.0, 1.0, 0.0, taxmask_rows),
        repetitions=20, seed=17,
    )
    assert result["balanced_accuracy_delta_ci"]["estimate"] == 0.5
    assert result["bootstrap"]["unit"] == "family_id"
    assert result["mcnemar"]["model_only_correct"] == 1


def test_group_aggregation_reports_mean_std_and_worst_seed():
    values = {
        "17": _metrics(0.6, 0.4, 0.2, []),
        "29": _metrics(0.7, 0.5, 0.1, []),
        "43": _metrics(0.8, 0.6, 0.0, []),
    }
    result = aggregate_group_metrics(values)
    assert result["balanced_accuracy"]["mean"] == 0.7
    assert result["balanced_accuracy"]["worst"] == 0.6
    assert result["overall_null_fpr"]["worst"] == 0.2
    assert result["seeds"] == 3
