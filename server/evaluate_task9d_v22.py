"""Evidence-level evaluation and frozen H1 decision for Task 9D v2.2."""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any

from evaluate_static_qlora import bbox_iou, pointing_game
from evaluate_task9d import _parse


def _rate(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _evidence_semantic(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("evidence_present"), bool):
        return False
    diagnosis = value.get("diagnosis")
    if not isinstance(diagnosis, dict):
        return False
    if value["evidence_present"] is True:
        region = value.get("evidence_region")
        return bool(
            isinstance(region, list) and len(region) == 4
            and diagnosis.get("status") == "supported"
            and value.get("reliability") == "supported"
        )
    return bool(
        value.get("evidence_region") is None
        and diagnosis.get("status") in {"abstain", "uncertain"}
        and all(diagnosis.get(key) is None for key in ("pest_id", "pest_name", "species", "stage"))
        and value.get("reliability") == "insufficient_visual_evidence"
    )


def _outcomes(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]):
    truth = {str(row["id"]): row for row in manifest}
    predicted = {str(row["id"]): row for row in predictions}
    if len(truth) != len(manifest) or len(predicted) != len(predictions) or set(truth) != set(predicted):
        raise ValueError("v2.2 evaluation ID mismatch or duplicate")
    outcomes = []
    for identifier in sorted(truth):
        row = truth[identifier]
        parsed = _parse(str(predicted[identifier].get("raw_text", "")))
        value = parsed["value"] if parsed["schema_valid"] else None
        prediction = value.get("evidence_present") if value is not None else None
        positive = str(row["role"]) == "positive"
        primary_positive = bool(
            positive and str(row.get("condition")) == "original"
            and str(row.get("prompt_view")) == "canonical"
        )
        semantic = bool(parsed["schema_valid"] and _evidence_semantic(value))
        task_compliant = bool(semantic and ((positive and prediction is True) or (not positive and prediction is False)))
        null_fp = bool(not positive and prediction is not False)
        region = value.get("evidence_region") if value else None
        gt = row.get("gt_bbox")
        iou = bbox_iou(region, gt) if primary_positive else None
        pointing = pointing_game(region, gt) if primary_positive else None
        outcomes.append({
            "id": identifier,
            "family_id": str(row["family_id"]),
            "role": str(row["role"]),
            "condition": str(row.get("condition")),
            "prompt_view": str(row.get("prompt_view")),
            "positive": positive,
            "primary_positive": primary_positive,
            "syntax_valid": bool(parsed["syntax_valid"]),
            "schema_valid": bool(parsed["schema_valid"]),
            "evidence_prediction": prediction,
            "evidence_semantic": semantic,
            "evidence_task_compliant": task_compliant,
            "positive_correct": bool(primary_positive and prediction is True),
            "null_fp": null_fp,
            "predicted_box": bool(region is not None),
            "iou": iou,
            "pointing": pointing,
        })
    return outcomes


def _evidence_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    positives = [row for row in rows if row["primary_positive"]]
    nulls = [row for row in rows if not row["positive"]]
    semantic = [row for row in nulls if row["role"] == "semantic_negative"]
    visual = [row for row in nulls if row["role"] == "visual_counterfactual"]
    tpr = _rate(row["positive_correct"] for row in positives)
    fpr = _rate(row["null_fp"] for row in nulls)
    return {
        "positive_samples": len(positives),
        "null_samples": len(nulls),
        "positive_tpr": tpr,
        "overall_null_fpr": fpr,
        "semantic_null_fpr": _rate(row["null_fp"] for row in semantic),
        "visual_null_fpr": _rate(row["null_fp"] for row in visual),
        "balanced_accuracy": (tpr + 1.0 - fpr) / 2.0,
    }


def evaluate_evidence_predictions(
    manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    outcomes = _outcomes(manifest, predictions)
    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    views: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        if not row["positive"]:
            conditions[row["condition"]].append(row)
        if row["positive"] and row["condition"] == "original":
            views[row["prompt_view"]].append(row)
    view_tpr = {
        view: _rate(row["evidence_prediction"] is True for row in rows)
        for view, rows in sorted(views.items())
    }
    native_values = [view_tpr[key] for key in ("native_0", "native_1", "native_2") if key in view_tpr]
    canonical = view_tpr.get("canonical", 0.0)
    native = mean(native_values) if native_values else canonical
    primary = [row for row in outcomes if row["primary_positive"]]
    nulls = [row for row in outcomes if not row["positive"]]
    return {
        "evidence": _evidence_metrics(outcomes),
        "by_condition": {
            condition: {
                "samples": len(rows),
                "null_fpr": _rate(row["null_fp"] for row in rows),
                "false_localization_rate": _rate(row["predicted_box"] for row in rows),
            }
            for condition, rows in sorted(conditions.items())
        },
        "json": {
            "syntax_validity": _rate(row["syntax_valid"] for row in outcomes),
            "schema_validity": _rate(row["schema_valid"] for row in outcomes),
        },
        "evidence_semantic_consistency": _rate(row["evidence_semantic"] for row in outcomes),
        "evidence_task_compliance": _rate(row["evidence_task_compliant"] for row in outcomes),
        "localization": {
            "positive_mean_iou": _rate(float(row["iou"] or 0.0) for row in primary),
            "positive_pointing_game": _rate(float(row["pointing"] or 0.0) for row in primary),
            "null_false_localization_rate": _rate(row["predicted_box"] for row in nulls),
        },
        "prompt_view_positive_tpr": view_tpr,
        "training_prompt_positive_tpr": native,
        "neutral_prompt_positive_tpr": canonical,
        "prompt_gap": abs(native - canonical),
        "row_outcomes": outcomes,
    }


def decide_h1(
    controls: dict[str, dict[str, Any]],
    taxmask: dict[str, dict[str, Any]],
    *,
    pooled_balanced_accuracy_delta_ci: dict[str, float],
) -> dict[str, Any]:
    if set(controls) != {"17", "29", "43"} or set(taxmask) != set(controls):
        raise ValueError("H1 decision requires paired seeds 17/29/43")
    seeds = ("17", "29", "43")
    ba_delta = [
        taxmask[seed]["evidence"]["balanced_accuracy"]
        - controls[seed]["evidence"]["balanced_accuracy"] for seed in seeds
    ]
    tpr_delta = [
        taxmask[seed]["evidence"]["positive_tpr"]
        - controls[seed]["evidence"]["positive_tpr"] for seed in seeds
    ]
    reasons: list[str] = []
    if mean(ba_delta) < 0.08:
        reasons.append("mean balanced-accuracy improvement below 8pp")
    if float(pooled_balanced_accuracy_delta_ci["low"]) <= 0.0:
        reasons.append("paired bootstrap 95% CI lower bound is not above zero")
    if sum(delta > 0 for delta in ba_delta) < 2:
        reasons.append("fewer than two seeds improve balanced accuracy")
    if min(ba_delta) < -0.03:
        reasons.append("worst seed balanced accuracy regresses by more than 3pp")
    if mean(tpr_delta) < 0.10:
        reasons.append("mean positive evidence TPR improvement below 10pp")
    for metric in ("overall_null_fpr", "semantic_null_fpr", "visual_null_fpr"):
        regression = mean(taxmask[s]["evidence"][metric] for s in seeds) - mean(
            controls[s]["evidence"][metric] for s in seeds
        )
        if regression > 0.03:
            reasons.append(f"null FPR regression exceeds 3pp for {metric}")
    for condition in ("blank", "blur"):
        if mean(taxmask[s]["by_condition"][condition]["null_fpr"] for s in seeds) >= 0.10:
            reasons.append(f"{condition} Null FPR is not below 10%")
    if min(taxmask[s]["json"]["syntax_validity"] for s in seeds) < 0.99:
        reasons.append("syntax validity below 99%")
    if min(taxmask[s]["json"]["schema_validity"] for s in seeds) < 0.99:
        reasons.append("schema validity below 99%")
    if min(taxmask[s]["evidence_semantic_consistency"] for s in seeds) < 0.99:
        reasons.append("evidence semantic consistency below 99%")
    if min(taxmask[s]["evidence_task_compliance"] for s in seeds) < 0.99:
        reasons.append("evidence task compliance below 99%")
    if max(taxmask[s]["prompt_gap"] for s in seeds) >= 0.05:
        reasons.append("prompt gap is not below 5pp")
    return {
        "hypothesis": "taxonomy value loss materially impairs evidence learning",
        "passed": not reasons,
        "reasons": reasons,
        "balanced_accuracy_delta_by_seed": dict(zip(seeds, ba_delta)),
        "mean_balanced_accuracy_delta": mean(ba_delta),
        "positive_tpr_delta_by_seed": dict(zip(seeds, tpr_delta)),
        "mean_positive_tpr_delta": mean(tpr_delta),
        "pooled_balanced_accuracy_delta_ci": pooled_balanced_accuracy_delta_ci,
        "authorize_larger_factorization_ablation": not reasons,
        "authorize_task9e": False,
    }
