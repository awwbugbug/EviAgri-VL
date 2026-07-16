import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task8 import compute_group_metrics


def prediction(
    *, present: bool, bbox, pest_id: int | None, pest_name: str = "pest"
) -> str:
    value = {
        "evidence_present": present,
        "evidence_bbox": bbox,
        "visible_attributes": [],
        "diagnosis": (
            {"pest_id": pest_id, "pest_name": pest_name}
            if pest_id is not None
            else "uncertain"
        ),
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }
    return json.dumps(value, separators=(",", ":"))


def row(
    condition: str,
    output: str,
    *,
    family: str = "f1",
    query_id: int = 7,
    actual_id: int = 7,
    gt_bbox=None,
) -> dict:
    return {
        "job_id": f"B1:{family}:{condition}",
        "audit_id": f"{family}:{condition}",
        "family_id": family,
        "group": "B1",
        "condition": condition,
        "query_pest_id": query_id,
        "actual_image_pest_id": actual_id,
        "expected_evidence_present": condition == "original_correct",
        "gt_bbox": gt_bbox,
        "prediction": output,
    }


def test_null_rows_never_enter_localization_metrics():
    rows = [
        row(
            "original_correct",
            prediction(present=True, bbox=[0, 0, 10, 10], pest_id=7),
            gt_bbox=[0, 0, 10, 10],
        ),
        row("blank_image", prediction(present=False, bbox=None, pest_id=None)),
    ]

    result = compute_group_metrics(rows)

    assert result["positive"]["samples"] == 1
    assert result["positive"]["mean_iou"] == 1.0
    assert result["counterfactual"]["samples"] == 1
    assert "mean_iou" not in result["counterfactual"]


def test_missing_positive_bbox_contributes_zero_iou_and_is_not_supported():
    result = compute_group_metrics(
        [
            row(
                "original_correct",
                prediction(present=True, bbox=None, pest_id=7),
                gt_bbox=[0, 0, 10, 10],
            )
        ]
    )

    assert result["positive"]["diagnosis_accuracy"] == 1.0
    assert result["positive"]["mean_iou"] == 0.0
    assert result["positive"]["supported_diagnosis_rate"] == 0.0


def test_false_box_on_null_is_counted_separately():
    result = compute_group_metrics(
        [row("blank_image", prediction(present=False, bbox=[1, 1, 4, 4], pest_id=None))]
    )

    assert result["counterfactual"]["predicted_box_on_null_rate"] == 1.0
    assert result["counterfactual"]["false_localization_rate"] == 0.0
    assert result["counterfactual"]["refusal_accuracy"] == 0.0
    assert result["counterfactual"]["ebhr"] == 1.0
    assert result["counterfactual"]["prompt_compliance_error"] == 1.0


def test_wrong_query_echo_raises_prompt_error_and_ebhr():
    result = compute_group_metrics(
        [
            row(
                "original_wrong_query",
                prediction(present=False, bbox=None, pest_id=9),
                query_id=9,
                actual_id=7,
            )
        ]
    )

    assert result["counterfactual"]["null_fpr"] == 1.0
    assert result["counterfactual"]["ebhr"] == 1.0
    assert result["counterfactual"]["prompt_compliance_error"] == 1.0


def test_supported_diagnosis_requires_correct_class_and_iou_at_least_half():
    rows = [
        row(
            "original_correct",
            prediction(present=True, bbox=[0, 0, 10, 10], pest_id=7),
            family="good",
            gt_bbox=[0, 0, 10, 10],
        ),
        row(
            "original_correct",
            prediction(present=True, bbox=[20, 20, 30, 30], pest_id=7),
            family="bad-box",
            gt_bbox=[0, 0, 10, 10],
        ),
        row(
            "original_correct",
            prediction(present=True, bbox=[0, 0, 10, 10], pest_id=8),
            family="bad-class",
            gt_bbox=[0, 0, 10, 10],
        ),
    ]

    result = compute_group_metrics(rows)

    assert result["positive"]["supported_diagnosis_rate"] == 1 / 3
    assert [item["supported"] for item in result["row_outcomes"]] == [True, False, False]


def test_plain_string_diagnosis_is_a_schema_and_compliance_failure():
    invalid = json.dumps(
        {
            "evidence_present": True,
            "evidence_bbox": [0, 0, 10, 10],
            "visible_attributes": ["body"],
            "diagnosis": "mole cricket",
            "reliability": "high",
        }
    )

    result = compute_group_metrics(
        [row("original_correct", invalid, gt_bbox=[0, 0, 10, 10])]
    )

    assert result["overall"]["schema_valid_rate"] == 0.0
    assert result["row_outcomes"][0]["prompt_compliant"] is False
