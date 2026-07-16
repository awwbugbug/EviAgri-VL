import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task10_pairs import evaluate_family_pairs


NULL_CONDITIONS = (
    "semantic_null",
    "source_visual_null",
    "blank",
    "blur",
    "shuffle",
)


def _value(present, pest_id=12):
    value = {
        "evidence_present": present,
        "evidence_region": [1, 2, 30, 40] if present else None,
        "visible_attributes": [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": pest_id if present else None,
            "pest_name": "aphid" if present else None,
            "species": None,
            "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }
    return json.dumps(value, separators=(",", ":"))


def _fixture(families=("f1",)):
    rows = []
    predictions = {}
    for family in families:
        original_id = f"{family}-original"
        rows.append({
            "id": original_id,
            "family_id": family,
            "role": "positive",
            "condition": "original",
            "prompt_view": "canonical",
            "query_class_id": 12,
        })
        predictions[original_id] = {"raw_text": _value(True)}
        for condition in NULL_CONDITIONS:
            identifier = f"{family}-{condition}"
            rows.append({
                "id": identifier,
                "family_id": family,
                "role": "semantic_negative" if condition == "semantic_null" else "visual_counterfactual",
                "condition": condition,
                "prompt_view": "canonical",
                "query_class_id": 12,
            })
            predictions[identifier] = {"raw_text": _value(False)}
    return rows, predictions


def test_strict_family_success_requires_original_and_all_nulls():
    rows, predictions = _fixture()

    report = evaluate_family_pairs(rows, predictions)
    assert report["strict_family_success"] == 1.0
    assert report["original_positive_tpr"] == 1.0

    predictions["f1-shuffle"] = {"raw_text": _value(True)}
    report = evaluate_family_pairs(rows, predictions)

    assert report["strict_family_success"] == 0.0
    assert report["by_condition"]["shuffle"]["pair_success"] == 0.0
    assert report["by_condition"]["shuffle"]["null_fpr"] == 1.0


def test_original_requires_correct_canonical_diagnosis_id():
    rows, predictions = _fixture()
    predictions["f1-original"] = {"raw_text": _value(True, pest_id=99)}

    report = evaluate_family_pairs(rows, predictions)

    assert report["original_positive_tpr"] == 0.0
    assert report["strict_family_success"] == 0.0


def test_invalid_json_is_failure_and_never_silently_excluded():
    rows, predictions = _fixture()
    predictions["f1-blank"] = {"raw_text": "not-json"}

    report = evaluate_family_pairs(rows, predictions)

    assert report["by_condition"]["blank"]["null_fpr"] == 1.0
    assert report["invalid_prediction_count"] == 1
    assert report["evaluated_prediction_count"] == 6


def test_missing_condition_blocks_report():
    rows, predictions = _fixture()
    rows = [row for row in rows if row["condition"] != "blank"]
    predictions.pop("f1-blank")

    with pytest.raises(ValueError, match="condition set"):
        evaluate_family_pairs(rows, predictions)


def test_prediction_id_mismatch_or_duplicate_manifest_blocks_report():
    rows, predictions = _fixture()
    predictions.pop("f1-blur")
    with pytest.raises(ValueError, match="prediction ID mismatch"):
        evaluate_family_pairs(rows, predictions)

    rows, predictions = _fixture()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate manifest ID"):
        evaluate_family_pairs(rows, predictions)


def test_contradiction_is_reversal_not_any_pair_failure():
    rows, predictions = _fixture(families=("f1", "f2"))
    predictions["f1-original"] = {"raw_text": _value(False)}
    predictions["f1-blank"] = {"raw_text": _value(True)}

    report = evaluate_family_pairs(rows, predictions)

    assert report["contradiction_rate"] == pytest.approx(0.5)
    assert report["by_condition"]["shuffle"]["pair_success"] == pytest.approx(0.5)
    assert report["by_condition"]["blank"]["concrete_diagnosis_drop"] == pytest.approx(0.0)
