import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from decide_task9d import decide_task9d
from evaluate_task9d import evaluate_predictions, paired_statistics


def _output(present, pest_id=None, box=None):
    return json.dumps({
        "evidence_present": present,
        "evidence_region": box,
        "visible_attributes": [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": pest_id if present else None,
            "pest_name": f"p{pest_id}" if present else None,
            "species": None, "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }, separators=(",", ":"))


def _row(identifier, family, role, condition, truth, output, prompt="canonical"):
    return ({
        "id": identifier, "family_id": family, "role": role, "condition": condition,
        "query_class_id": truth, "gt_bbox": [0, 0, 10, 10] if role == "positive" else None,
        "prompt_view": prompt,
    }, {"id": identifier, "raw_text": output})


def test_metrics_separate_positive_null_conditions_bands_and_json_levels():
    pairs = [
        _row("p1", "f1", "positive", "original", 1, _output(True, 1, [0, 0, 10, 10])),
        _row("p2", "f2", "positive", "original", 2, _output(True, 9, [20, 20, 30, 30])),
        _row("s1", "f3", "semantic_negative", "semantic_null", 3, _output(False)),
        _row("b1", "f4", "visual_counterfactual", "blank", 4, _output(False)),
        _row("u1", "f5", "visual_counterfactual", "blur", 5, _output(True, 5, [1, 1, 2, 2])),
        _row("h1", "f6", "visual_counterfactual", "shuffle", 6, "not-json"),
    ]
    manifest, predictions = zip(*pairs)
    result = evaluate_predictions(list(manifest), list(predictions), {1: "head", 2: "tail"})
    assert result["positive"]["accuracy"] == 0.5
    assert result["positive"]["macro_f1"] == 0.5
    assert result["positive"]["mean_iou"] == 0.5
    assert result["positive"]["pointing_game"] == 0.5
    assert result["bands"]["head"]["accuracy"] == 1.0
    assert result["bands"]["tail"]["accuracy"] == 0.0
    assert result["null"]["semantic_null_fpr"] == 0.0
    assert result["null"]["visual_null_fpr"] == 1 / 3
    assert result["by_condition"]["blank"]["refusal_rate"] == 1.0
    assert result["by_condition"]["blur"]["concrete_diagnosis_rate"] == 1.0
    assert result["json"]["syntax_validity"] == 5 / 6
    assert result["json"]["schema_validity"] == 5 / 6
    assert result["json"]["semantic_consistency"] == 5 / 6
    assert result["json"]["task_compliance"] == 4 / 6


def test_prompt_gap_and_family_paired_image_dependency():
    pairs = []
    for view, correct in (("native", True), ("canonical", False), ("unseen_alpha", True)):
        pairs.append(_row(f"{view}-p", f"{view}-f", "positive", "original", 1,
                          _output(True, 1 if correct else 9, [0, 0, 10, 10]), prompt=view))
    for condition in ("blank", "blur", "shuffle"):
        pairs.append(_row(f"pair-{condition}", "paired", "visual_counterfactual", condition, 1, _output(False)))
    pairs.append(_row("pair-original", "paired", "positive", "original", 1,
                      _output(True, 1, [0, 0, 10, 10]), prompt="canonical"))
    manifest, predictions = zip(*pairs)
    result = evaluate_predictions(list(manifest), list(predictions), {1: "head"})
    assert result["prompt_views"]["native"]["accuracy"] == 1.0
    assert result["prompt_gap"] == 0.5
    assert result["paired_image_dependency"]["blank"]["specific_diagnosis_drop"] == 1.0


def test_prompt_gap_averages_numbered_native_training_views():
    pairs = []
    for view, correct in (
        ("canonical", False),
        ("native_0", True),
        ("native_1", True),
        ("native_2", False),
    ):
        pairs.append(_row(
            f"{view}-p", f"{view}-f", "positive", "original", 1,
            _output(True, 1 if correct else 9, [0, 0, 10, 10]), prompt=view,
        ))
    manifest, predictions = zip(*pairs)
    result = evaluate_predictions(list(manifest), list(predictions), {1: "head"})
    assert result["positive"]["samples"] == 1
    assert result["positive"]["accuracy"] == 0.0
    assert result["training_prompt_accuracy"] == 2 / 3
    assert result["neutral_prompt_accuracy"] == 0.0
    assert result["prompt_gap"] == 2 / 3


def _seed_metrics(acc=.70, macro=.60, null=.04, supported=.50, gap=.02, compliance=1.0):
    return {
        "positive": {"accuracy": acc, "macro_f1": macro, "supported_diagnosis_rate": supported},
        "null": {"overall_null_fpr": null},
        "by_condition": {"blank": {"null_fpr": null}, "blur": {"null_fpr": null}},
        "prompt_gap": gap,
        "json": {"schema_validity": 1.0, "semantic_consistency": compliance, "task_compliance": compliance},
        "shortcut_gate_passed": True,
        "macro_f1_delta_ci": {"high": macro - .55},
        "null_fpr_delta_ci": {"high": -0.01},
    }


def test_decision_applies_any_seed_elimination_and_lexicographic_selection():
    base = _seed_metrics(acc=.70, macro=.55, null=.20, supported=.40)
    variants = {
        "A": [_seed_metrics(acc=.66), _seed_metrics(), _seed_metrics()],
        "B": [_seed_metrics(macro=.62, null=.04), _seed_metrics(macro=.61, null=.03), _seed_metrics(macro=.60, null=.05)],
        "C": [_seed_metrics(macro=.63, null=.12), _seed_metrics(macro=.64), _seed_metrics(macro=.62)],
    }
    decision = decide_task9d(base, variants)
    assert decision["variants"]["A"]["passed"] is False
    assert "accuracy" in " ".join(decision["variants"]["A"]["reasons"]).lower()
    assert decision["variants"]["C"]["passed"] is False
    assert decision["selected_protocol"] == "B"
    assert decision["authorize_9e_recommendation"] is True
    assert decision["variants"]["B"]["aggregate"]["seeds"] == 3


def test_paired_statistics_use_1000_family_bootstraps_and_mcnemar():
    base = {"row_outcomes": [
        {"id": "p1", "family_id": "f1", "positive": True, "query_class_id": 1,
         "diagnosis_id": 9, "diagnosis_correct": False, "iou": 0.0},
        {"id": "n1", "family_id": "f1", "positive": False, "specific_diagnosis": True},
        {"id": "p2", "family_id": "f2", "positive": True, "query_class_id": 2,
         "diagnosis_id": 2, "diagnosis_correct": True, "iou": 1.0},
        {"id": "n2", "family_id": "f2", "positive": False, "specific_diagnosis": True},
    ]}
    model = {"row_outcomes": [
        {"id": "p1", "family_id": "f1", "positive": True, "query_class_id": 1,
         "diagnosis_id": 1, "diagnosis_correct": True, "iou": 1.0},
        {"id": "n1", "family_id": "f1", "positive": False, "specific_diagnosis": False},
        {"id": "p2", "family_id": "f2", "positive": True, "query_class_id": 2,
         "diagnosis_id": 2, "diagnosis_correct": True, "iou": 1.0},
        {"id": "n2", "family_id": "f2", "positive": False, "specific_diagnosis": False},
    ]}
    report = paired_statistics(base, model, repetitions=1000, seed=20260716)
    assert report["macro_f1_delta_ci"]["repetitions"] == 1000
    assert report["null_fpr_delta_ci"]["estimate"] == -1.0
    assert report["accuracy_mcnemar"]["model_only_correct"] == 1
