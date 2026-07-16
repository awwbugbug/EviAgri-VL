import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task9d_v22 import decide_h1, evaluate_evidence_predictions


def _raw(present):
    value = {
        "evidence_present": present,
        "evidence_region": [1, 2, 3, 4] if present else None,
        "visible_attributes": [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": 1 if present else None,
            "pest_name": "pest" if present else None,
            "species": None,
            "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }
    return json.dumps(value, separators=(",", ":"))


def _fixture():
    manifest = []
    for family in ("f1", "f2"):
        manifest.extend([
            {"id": f"{family}-p", "family_id": family, "role": "positive", "condition": "original",
             "prompt_view": "canonical", "query_class_id": 1, "gt_bbox": [1, 2, 3, 4]},
            {"id": f"{family}-n", "family_id": family, "role": "semantic_negative", "condition": "semantic_null",
             "prompt_view": "canonical", "query_class_id": 2, "gt_bbox": None},
            {"id": f"{family}-b", "family_id": family, "role": "visual_counterfactual", "condition": "blank",
             "prompt_view": "canonical", "query_class_id": 1, "gt_bbox": None},
        ])
    predictions = [
        {"id": row["id"], "raw_text": _raw(row["role"] == "positive")}
        for row in manifest
    ]
    return manifest, predictions


def test_evidence_metrics_use_canonical_positive_and_penalize_invalid_null():
    manifest, predictions = _fixture()
    metrics = evaluate_evidence_predictions(manifest, predictions)
    assert metrics["evidence"]["positive_tpr"] == 1.0
    assert metrics["evidence"]["overall_null_fpr"] == 0.0
    assert metrics["evidence"]["balanced_accuracy"] == 1.0
    assert metrics["json"]["schema_validity"] == 1.0

    predictions[1]["raw_text"] = "not json"
    metrics = evaluate_evidence_predictions(manifest, predictions)
    assert metrics["evidence"]["overall_null_fpr"] == 0.25
    assert metrics["json"]["schema_validity"] == 5 / 6


def _seed_metrics(ba, tpr, fpr, schema=1.0, compliance=1.0, prompt_gap=0.0):
    return {
        "evidence": {
            "balanced_accuracy": ba, "positive_tpr": tpr,
            "overall_null_fpr": fpr, "semantic_null_fpr": fpr,
            "visual_null_fpr": fpr,
        },
        "by_condition": {"blank": {"null_fpr": fpr}, "blur": {"null_fpr": fpr}},
        "json": {"syntax_validity": schema, "schema_validity": schema},
        "evidence_semantic_consistency": compliance,
        "evidence_task_compliance": compliance,
        "prompt_gap": prompt_gap,
    }


def test_h1_decision_applies_all_preregistered_gates():
    controls = {str(seed): _seed_metrics(0.60, 0.30, 0.10) for seed in (17, 29, 43)}
    taxmask = {str(seed): _seed_metrics(0.72, 0.45, 0.01) for seed in (17, 29, 43)}
    passed = decide_h1(
        controls, taxmask,
        pooled_balanced_accuracy_delta_ci={"estimate": 0.12, "low": 0.04, "high": 0.20},
    )
    assert passed["passed"] is True

    taxmask = {str(seed): _seed_metrics(0.72, 0.45, 0.14) for seed in (17, 29, 43)}
    failed = decide_h1(
        controls, taxmask,
        pooled_balanced_accuracy_delta_ci={"estimate": 0.12, "low": 0.04, "high": 0.20},
    )
    assert failed["passed"] is False
    assert any("null FPR regression" in reason for reason in failed["reasons"])
