"""Condition-aware, leakage-resistant metrics for Task 9D."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_static_qlora import _macro_f1, bbox_iou, pointing_game
from evaluate_task8 import exact_mcnemar, paired_bootstrap_delta
from task9b_protocol import DIAGNOSIS_KEYS, OUTPUT_KEYS, validate_target_semantics


def _divide(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _schema_valid(value: Any) -> bool:
    if not isinstance(value, dict) or tuple(value) != OUTPUT_KEYS:
        return False
    if not isinstance(value.get("evidence_present"), bool):
        return False
    if not isinstance(value.get("visible_attributes"), list):
        return False
    diagnosis = value.get("diagnosis")
    return isinstance(diagnosis, dict) and tuple(diagnosis) == DIAGNOSIS_KEYS


def _parse(raw: str) -> dict[str, Any]:
    outcome = {"syntax_valid": False, "schema_valid": False, "semantic_consistent": False,
               "value": None, "error": None}
    try:
        value = json.loads(raw)
        outcome["syntax_valid"] = isinstance(value, dict)
        if not outcome["syntax_valid"]:
            raise ValueError("top-level JSON is not an object")
        outcome["value"] = value
        outcome["schema_valid"] = _schema_valid(value)
        if not outcome["schema_valid"]:
            raise ValueError("schema or key order invalid")
        validate_target_semantics(value)
        outcome["semantic_consistent"] = True
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        outcome["error"] = str(exc)
    return outcome


def _specific(value: Any) -> bool:
    return bool(isinstance(value, dict) and isinstance(value.get("diagnosis"), dict)
                and value["diagnosis"].get("pest_id") is not None)


def _refusal(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    diagnosis = value.get("diagnosis")
    return bool(value.get("evidence_present") is False and value.get("evidence_region") is None
                and isinstance(diagnosis, dict) and diagnosis.get("status") in {"uncertain", "abstain"}
                and all(diagnosis.get(key) is None for key in ("pest_id", "pest_name", "species", "stage")))


def _outcomes(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    truth = {str(row["id"]): row for row in manifest}
    predicted = {str(row["id"]): row for row in predictions}
    if len(truth) != len(manifest) or len(predicted) != len(predictions) or set(truth) != set(predicted):
        raise ValueError("Task 9D evaluation ID mismatch or duplicate")
    outcomes = []
    for identifier in sorted(truth):
        row = truth[identifier]
        parsed = _parse(str(predicted[identifier].get("raw_text", "")))
        value = parsed["value"] if parsed["semantic_consistent"] else None
        role = str(row["role"])
        positive = role == "positive"
        primary_positive = positive and str(row.get("prompt_view") or "canonical") == "canonical"
        diagnosis = value.get("diagnosis") if value else None
        diagnosis_id = diagnosis.get("pest_id") if isinstance(diagnosis, dict) else None
        correct = bool(positive and diagnosis_id == int(row["query_class_id"]))
        region = value.get("evidence_region") if value else None
        gt = row.get("gt_bbox")
        iou = bbox_iou(region, gt) if positive else None
        pointing = pointing_game(region, gt) if positive else None
        supported = bool(correct and value.get("evidence_present") is True and iou is not None and iou >= .5)
        refusal = _refusal(value)
        task_compliant = bool(parsed["semantic_consistent"] and (
            (positive and value.get("evidence_present") is True and _specific(value))
            or (not positive and refusal)
        ))
        outcomes.append({
            **{key: row.get(key) for key in ("id", "family_id", "role", "condition", "prompt_view", "query_class_id")},
            **parsed, "positive": positive, "diagnosis_id": diagnosis_id, "diagnosis_correct": correct,
            "primary_positive": primary_positive,
            "specific_diagnosis": _specific(value), "refusal": refusal, "task_compliant": task_compliant,
            "iou": iou, "pointing": pointing, "supported": supported,
            "predicted_box": bool(value is not None and value.get("evidence_region") is not None),
        })
    return outcomes


def _positive_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [int(row["query_class_id"]) for row in rows]
    predicted = [row["diagnosis_id"] for row in rows]
    return {
        "samples": len(rows),
        "accuracy": _divide(sum(row["diagnosis_correct"] for row in rows), len(rows)),
        "macro_f1": _macro_f1(truth, predicted),
        "mean_iou": _divide(sum(float(row["iou"] or 0) for row in rows), len(rows)),
        "pointing_game": _divide(sum(float(row["pointing"] or 0) for row in rows), len(rows)),
        "supported_diagnosis_rate": _divide(sum(row["supported"] for row in rows), len(rows)),
    }


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "null_fpr": _divide(sum(row["specific_diagnosis"] for row in rows), len(rows)),
        "refusal_rate": _divide(sum(row["refusal"] for row in rows), len(rows)),
        "concrete_diagnosis_rate": _divide(sum(row["specific_diagnosis"] for row in rows), len(rows)),
        "false_localization_rate": _divide(sum(row["predicted_box"] for row in rows), len(rows)),
    }


def evaluate_predictions(
    manifest: list[dict[str, Any]], predictions: list[dict[str, Any]], class_bands: dict[int, str]
) -> dict[str, Any]:
    outcomes = _outcomes(manifest, predictions)
    all_positives = [row for row in outcomes if row["positive"]]
    positives = [row for row in all_positives if row["primary_positive"]]
    nulls = [row for row in outcomes if not row["positive"]]
    semantic = [row for row in nulls if row["role"] == "semantic_negative"]
    visual = [row for row in nulls if row["role"] == "visual_counterfactual"]
    grouped_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nulls:
        grouped_condition[str(row["condition"])].append(row)
    bands = {}
    for band in ("head", "medium", "tail"):
        items = [row for row in positives if class_bands.get(int(row["query_class_id"])) == band]
        if items:
            bands[band] = _positive_metrics(items)
    views: dict[str, Any] = {}
    grouped_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_positives:
        grouped_view[str(row.get("prompt_view") or "canonical")].append(row)
    for view, rows in sorted(grouped_view.items()):
        views[view] = _positive_metrics(rows)
    native_rows = [
        row for row in all_positives
        if str(row.get("prompt_view") or "") == "native"
        or str(row.get("prompt_view") or "").startswith("native_")
    ]
    canonical = views.get("canonical", {"accuracy": 0.0})["accuracy"]
    native = _positive_metrics(native_rows)["accuracy"] if native_rows else canonical
    paired: dict[str, Any] = {}
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        family_rows[str(row["family_id"])].append(row)
    for condition in ("blank", "blur", "shuffle"):
        drops = []
        for rows in family_rows.values():
            originals = [row for row in rows if row["condition"] == "original" and row["primary_positive"]]
            changed = [row for row in rows if row["condition"] == condition]
            if originals and changed:
                drops.append(float(originals[0]["specific_diagnosis"]) - float(changed[0]["specific_diagnosis"]))
        if drops:
            paired[condition] = {"families": len(drops), "specific_diagnosis_drop": sum(drops) / len(drops)}
    return {
        "positive": _positive_metrics(positives),
        "bands": bands,
        "null": {
            "samples": len(nulls),
            "overall_null_fpr": _divide(sum(row["specific_diagnosis"] for row in nulls), len(nulls)),
            "semantic_null_fpr": _divide(sum(row["specific_diagnosis"] for row in semantic), len(semantic)),
            "visual_null_fpr": _divide(sum(row["specific_diagnosis"] for row in visual), len(visual)),
            "concrete_diagnosis_under_null": _divide(sum(row["specific_diagnosis"] for row in nulls), len(nulls)),
            "false_localization_rate": _divide(sum(row["predicted_box"] for row in nulls), len(nulls)),
        },
        "by_condition": {condition: _condition_metrics(rows) for condition, rows in sorted(grouped_condition.items())},
        "json": {
            "syntax_validity": _divide(sum(row["syntax_valid"] for row in outcomes), len(outcomes)),
            "schema_validity": _divide(sum(row["schema_valid"] for row in outcomes), len(outcomes)),
            "semantic_consistency": _divide(sum(row["semantic_consistent"] for row in outcomes), len(outcomes)),
            "task_compliance": _divide(sum(row["task_compliant"] for row in outcomes), len(outcomes)),
        },
        "prompt_views": views,
        "training_prompt_accuracy": native,
        "neutral_prompt_accuracy": canonical,
        "prompt_gap": abs(float(native) - float(canonical)),
        "paired_image_dependency": paired,
        "row_outcomes": outcomes,
    }


def paired_statistics(
    base_metrics: dict[str, Any], model_metrics: dict[str, Any], *,
    repetitions: int = 1000, seed: int = 20260716,
) -> dict[str, Any]:
    base_rows = {str(row["id"]): row for row in base_metrics["row_outcomes"]}
    model_rows = {str(row["id"]): row for row in model_metrics["row_outcomes"]}
    if set(base_rows) != set(model_rows):
        raise ValueError("paired Task 9D statistics require identical evaluation IDs")
    positive_ids = sorted(
        identifier for identifier, row in base_rows.items()
        if row.get("primary_positive", row["positive"])
    )
    null_ids = sorted(identifier for identifier, row in base_rows.items() if not row["positive"])
    base_positive = [base_rows[identifier] for identifier in positive_ids]
    model_positive = [model_rows[identifier] for identifier in positive_ids]
    base_null = [base_rows[identifier] for identifier in null_ids]
    model_null = [model_rows[identifier] for identifier in null_ids]
    macro = lambda rows: _macro_f1(
        [int(row["query_class_id"]) for row in rows], [row["diagnosis_id"] for row in rows]
    )
    mean_accuracy = lambda rows: _divide(sum(row["diagnosis_correct"] for row in rows), len(rows))
    mean_null_fpr = lambda rows: _divide(sum(row["specific_diagnosis"] for row in rows), len(rows))
    mean_iou = lambda rows: _divide(sum(float(row["iou"] or 0) for row in rows), len(rows))
    return {
        "bootstrap": {"repetitions": repetitions, "seed": seed, "unit": "family_id"},
        "accuracy_delta_ci": paired_bootstrap_delta(
            base_positive, model_positive, mean_accuracy, repetitions=repetitions, seed=seed
        ),
        "macro_f1_delta_ci": paired_bootstrap_delta(
            base_positive, model_positive, macro, repetitions=repetitions, seed=seed
        ),
        "null_fpr_delta_ci": paired_bootstrap_delta(
            base_null, model_null, mean_null_fpr, repetitions=repetitions, seed=seed
        ),
        "localization_delta_ci": paired_bootstrap_delta(
            base_positive, model_positive, mean_iou, repetitions=repetitions, seed=seed
        ),
        "accuracy_mcnemar": exact_mcnemar(
            [bool(base_rows[identifier]["diagnosis_correct"]) for identifier in positive_ids],
            [bool(model_rows[identifier]["diagnosis_correct"]) for identifier in positive_ids],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--class-bands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bands = {int(key): value for key, value in json.loads(args.class_bands.read_text(encoding="utf-8")).items()}
    result = evaluate_predictions(read(args.manifest), read(args.predictions), bands)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
