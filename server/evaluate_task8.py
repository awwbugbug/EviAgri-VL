from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from evaluate_static_qlora import (
    _diagnosis_id,
    _macro_f1,
    _safe_divide,
    _valid_bbox,
    bbox_iou,
    parse_structured_json,
    pointing_game,
)


POSITIVE_CONDITION = "original_correct"


def parse_task8_prediction(text: str) -> dict[str, Any]:
    value = parse_structured_json(text)
    if not isinstance(value["evidence_present"], bool):
        raise ValueError("evidence_present must be boolean")
    box = value["evidence_bbox"]
    if box is not None and not _valid_bbox(box):
        raise ValueError("evidence_bbox must be a valid box or null")
    attributes = value["visible_attributes"]
    if not isinstance(attributes, list) or not all(
        isinstance(attribute, str) for attribute in attributes
    ):
        raise ValueError("visible_attributes must be an array of strings")
    diagnosis = value["diagnosis"]
    valid_diagnosis = diagnosis == "uncertain" or (
        isinstance(diagnosis, dict)
        and tuple(diagnosis) == ("pest_id", "pest_name")
        and isinstance(diagnosis["pest_id"], int)
        and not isinstance(diagnosis["pest_id"], bool)
        and isinstance(diagnosis["pest_name"], str)
        and bool(diagnosis["pest_name"].strip())
    )
    if not valid_diagnosis:
        raise ValueError("diagnosis must be a pest_id/pest_name object or uncertain")
    if not isinstance(value["reliability"], str) or not value["reliability"].strip():
        raise ValueError("reliability must be a non-empty string")
    return value


def _family_map(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_id = row.get("family_id")
        if family_id is None:
            raise ValueError("bootstrap row is missing family_id")
        grouped[str(family_id)].append(row)
    for family_rows in grouped.values():
        family_rows.sort(
            key=lambda row: (
                str(row.get("condition", "")),
                str(row.get("audit_id", "")),
                str(row.get("job_id", "")),
            )
        )
    if not grouped:
        raise ValueError("bootstrap requires at least one family")
    return dict(grouped)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    repetitions: int = 1000,
    seed: int = 20260715,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    grouped = _family_map(rows)
    family_ids = sorted(grouped)
    canonical = [row for family_id in family_ids for row in grouped[family_id]]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repetitions):
        sampled: list[dict[str, Any]] = []
        for _ in family_ids:
            sampled.extend(grouped[rng.choice(family_ids)])
        samples.append(float(metric(sampled)))
    return {
        "estimate": float(metric(canonical)),
        "low": _percentile(samples, 0.025),
        "high": _percentile(samples, 0.975),
        "confidence": 0.95,
        "repetitions": repetitions,
        "seed": seed,
        "resampling_unit": "family_id",
    }


def paired_bootstrap_delta(
    baseline_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    repetitions: int = 1000,
    seed: int = 20260715,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    baseline = _family_map(baseline_rows)
    model = _family_map(model_rows)
    if set(baseline) != set(model):
        raise ValueError("paired bootstrap requires identical family sets")
    family_ids = sorted(baseline)
    canonical_baseline = [row for family_id in family_ids for row in baseline[family_id]]
    canonical_model = [row for family_id in family_ids for row in model[family_id]]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(repetitions):
        sampled_baseline: list[dict[str, Any]] = []
        sampled_model: list[dict[str, Any]] = []
        for _ in family_ids:
            family_id = rng.choice(family_ids)
            sampled_baseline.extend(baseline[family_id])
            sampled_model.extend(model[family_id])
        deltas.append(float(metric(sampled_model) - metric(sampled_baseline)))
    return {
        "estimate": float(metric(canonical_model) - metric(canonical_baseline)),
        "low": _percentile(deltas, 0.025),
        "high": _percentile(deltas, 0.975),
        "confidence": 0.95,
        "repetitions": repetitions,
        "seed": seed,
        "resampling_unit": "family_id",
        "delta_direction": "model_minus_baseline",
    }


def exact_mcnemar(
    baseline_correct: list[bool], model_correct: list[bool]
) -> dict[str, Any]:
    if len(baseline_correct) != len(model_correct):
        raise ValueError("McNemar inputs must be paired and equal length")
    baseline_only = sum(
        bool(baseline) and not bool(model)
        for baseline, model in zip(baseline_correct, model_correct)
    )
    model_only = sum(
        not bool(baseline) and bool(model)
        for baseline, model in zip(baseline_correct, model_correct)
    )
    discordant = baseline_only + model_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(baseline_only, model_only)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(tail + 1))
            / (2**discordant),
        )
    return {
        "baseline_only_correct": baseline_only,
        "model_only_correct": model_only,
        "discordant": discordant,
        "p_value": p_value,
        "test": "exact_two_sided_mcnemar",
    }


def _parse_prediction(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return parse_task8_prediction(str(row["prediction"])), None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return None, str(error)


def _row_outcome(row: dict[str, Any]) -> dict[str, Any]:
    parsed, parse_error = _parse_prediction(row)
    positive = row.get("condition") == POSITIVE_CONDITION
    diagnosis = parsed.get("diagnosis") if parsed else None
    diagnosis_id = _diagnosis_id(diagnosis)
    predicted_present = parsed is not None and parsed.get("evidence_present") is True
    predicted_bbox = parsed.get("evidence_bbox") if parsed else None
    diagnosis_correct = bool(
        positive
        and diagnosis_id is not None
        and diagnosis_id == row.get("actual_image_pest_id")
    )
    iou = bbox_iou(predicted_bbox, row.get("gt_bbox")) if positive else None
    pointing = pointing_game(predicted_bbox, row.get("gt_bbox")) if positive else None
    supported = bool(
        positive
        and diagnosis_correct
        and predicted_present
        and _valid_bbox(predicted_bbox)
        and iou is not None
        and iou >= 0.5
    )
    specific_diagnosis = isinstance(diagnosis, dict)
    refusal = bool(
        parsed is not None
        and parsed.get("evidence_present") is False
        and parsed.get("evidence_bbox") is None
        and parsed.get("diagnosis") == "uncertain"
        and parsed.get("reliability") == "insufficient_visual_evidence"
    )
    wrong_query_echo = bool(
        row.get("group") != "B0"
        and row.get("condition") == "original_wrong_query"
        and specific_diagnosis
        and diagnosis_id == row.get("query_pest_id")
        and row.get("query_pest_id") != row.get("actual_image_pest_id")
    )
    internally_consistent = bool(
        parsed is not None
        and (
            (
                parsed.get("evidence_present") is True
                and _valid_bbox(parsed.get("evidence_bbox"))
                and isinstance(parsed.get("diagnosis"), dict)
            )
            or (
                parsed.get("evidence_present") is False
                and parsed.get("evidence_bbox") is None
                and parsed.get("visible_attributes") == []
                and parsed.get("diagnosis") == "uncertain"
                and parsed.get("reliability") == "insufficient_visual_evidence"
            )
        )
    )
    ebh = bool(
        not positive
        and (
            parsed is None
            or parsed.get("evidence_present") is not False
            or parsed.get("evidence_bbox") is not None
            or parsed.get("diagnosis") != "uncertain"
            or parsed.get("reliability") != "insufficient_visual_evidence"
        )
    )
    expected_present = bool(row.get("expected_evidence_present"))
    presence_correct = bool(
        parsed is not None and predicted_present == expected_present
    )
    prompt_compliant = parse_error is None and internally_consistent and not wrong_query_echo
    return {
        "job_id": row.get("job_id"),
        "audit_id": row.get("audit_id"),
        "family_id": row.get("family_id"),
        "group": row.get("group"),
        "condition": row.get("condition"),
        "schema_valid": parsed is not None,
        "parse_error": parse_error,
        "diagnosis_correct": diagnosis_correct,
        "presence_correct": presence_correct,
        "refusal_correct": refusal if not positive else False,
        "supported": supported,
        "prompt_compliant": prompt_compliant,
        "predicted_present": predicted_present,
        "specific_diagnosis": specific_diagnosis,
        "predicted_box": predicted_bbox is not None,
        "false_localization": predicted_present if not positive else False,
        "ebh": ebh if not positive else False,
        "diagnosis_id": diagnosis_id,
        "truth_diagnosis_id": row.get("actual_image_pest_id") if positive else None,
        "iou": iou,
        "pointing": pointing,
    }


def _positive_metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [item["truth_diagnosis_id"] for item in outcomes]
    predicted = [item["diagnosis_id"] for item in outcomes]
    ious = [float(item["iou"] or 0.0) for item in outcomes]
    return {
        "samples": len(outcomes),
        "diagnosis_accuracy": _safe_divide(
            sum(item["diagnosis_correct"] for item in outcomes), len(outcomes)
        ),
        "diagnosis_macro_f1": _macro_f1(truth, predicted),
        "mean_iou": _safe_divide(sum(ious), len(outcomes)),
        "pointing_game": _safe_divide(
            sum(float(item["pointing"] or 0.0) for item in outcomes), len(outcomes)
        ),
        "supported_diagnosis_rate": _safe_divide(
            sum(item["supported"] for item in outcomes), len(outcomes)
        ),
    }


def _counterfactual_metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(outcomes),
        "null_fpr": _safe_divide(
            sum(item["specific_diagnosis"] for item in outcomes), len(outcomes)
        ),
        "refusal_accuracy": _safe_divide(
            sum(item["refusal_correct"] for item in outcomes), len(outcomes)
        ),
        "false_localization_rate": _safe_divide(
            sum(item["false_localization"] for item in outcomes), len(outcomes)
        ),
        "predicted_box_on_null_rate": _safe_divide(
            sum(item["predicted_box"] for item in outcomes), len(outcomes)
        ),
        "ebhr": _safe_divide(sum(item["ebh"] for item in outcomes), len(outcomes)),
        "prompt_compliance_error": _safe_divide(
            sum(not item["prompt_compliant"] for item in outcomes), len(outcomes)
        ),
    }


def _overall_metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [item for item in outcomes if item["condition"] == POSITIVE_CONDITION]
    negatives = [item for item in outcomes if item["condition"] != POSITIVE_CONDITION]
    true_positive = sum(
        item["schema_valid"] and item["predicted_present"] for item in positives
    )
    false_positive = sum(
        item["schema_valid"] and item["predicted_present"] for item in negatives
    )
    false_negative = len(positives) - true_positive
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    presence_f1 = _safe_divide(2 * precision * recall, precision + recall)
    sensitivity = _safe_divide(
        sum(item["presence_correct"] for item in positives), len(positives)
    )
    specificity = _safe_divide(
        sum(item["presence_correct"] for item in negatives), len(negatives)
    )
    return {
        "samples": len(outcomes),
        "schema_valid_rate": _safe_divide(
            sum(item["schema_valid"] for item in outcomes), len(outcomes)
        ),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "presence_f1": presence_f1,
    }


def compute_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [_row_outcome(row) for row in rows]
    positives = [item for item in outcomes if item["condition"] == POSITIVE_CONDITION]
    counterfactuals = [
        item for item in outcomes if item["condition"] != POSITIVE_CONDITION
    ]
    by_condition: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in outcomes:
        grouped[str(item["condition"])].append(item)
    for condition, items in sorted(grouped.items()):
        by_condition[condition] = (
            _positive_metrics(items)
            if condition == POSITIVE_CONDITION
            else _counterfactual_metrics(items)
        )
    return {
        "positive": _positive_metrics(positives),
        "counterfactual": _counterfactual_metrics(counterfactuals),
        "overall": _overall_metrics(outcomes),
        "by_condition": by_condition,
        "row_outcomes": outcomes,
    }


def _mean_boolean(rows: list[dict[str, Any]], key: str) -> float:
    return _safe_divide(sum(bool(row[key]) for row in rows), len(rows))


def _metric_from_outcomes(
    outcomes: list[dict[str, Any]], section: str, metric: str
) -> float:
    positives = [row for row in outcomes if row["condition"] == POSITIVE_CONDITION]
    counterfactuals = [row for row in outcomes if row["condition"] != POSITIVE_CONDITION]
    if section == "positive":
        return float(_positive_metrics(positives)[metric])
    if section == "counterfactual":
        return float(_counterfactual_metrics(counterfactuals)[metric])
    if section == "overall":
        return float(_overall_metrics(outcomes)[metric])
    raise ValueError(f"unknown metric section: {section}")


def _decision(metrics: dict[str, dict[str, Any]], paired: dict[str, Any]) -> dict[str, Any]:
    b2 = metrics.get("B2")
    if b2 is None or b2["positive"]["samples"] < 50:
        return {
            "value": "inconclusive",
            "reasons": ["fewer than 50 positive families; smoke results cannot select A or B"],
        }
    b3 = metrics.get("B3")
    delta = paired.get("original_correct.diagnosis_correct", {}).get(
        "bootstrap_delta", {}
    )
    shortcut_gap = (
        b3["positive"]["diagnosis_macro_f1"]
        - b2["positive"]["diagnosis_macro_f1"]
        if b3 is not None
        else 0.0
    )
    blank = b2["by_condition"].get("blank_image", {})
    shuffled = b2["by_condition"].get("shuffled_image", {})
    failures = []
    if shortcut_gap > 0.15:
        failures.append(f"B3-B2 Macro-F1 shortcut gap is {shortcut_gap:.4f} (>0.15)")
    if blank.get("null_fpr", 0.0) >= 0.5:
        failures.append("B2 blank-image Null FPR is at least 0.5")
    if shuffled.get("null_fpr", 0.0) >= 0.5:
        failures.append("B2 shuffled-image Null FPR is at least 0.5")
    if b2["overall"]["schema_valid_rate"] < 0.8:
        failures.append("B2 schema-valid rate is below 0.8")
    if failures:
        return {"value": "B", "reasons": failures}
    passes = [
        delta.get("low", float("-inf")) > 0,
        blank.get("refusal_accuracy", 0.0) >= 0.8,
        shuffled.get("refusal_accuracy", 0.0) >= 0.8,
        shortcut_gap <= 0.15,
        b2["overall"]["schema_valid_rate"] >= 0.95,
    ]
    if all(passes):
        return {
            "value": "A",
            "reasons": [
                "B2 improves over B1 with a positive paired 95% CI and passes causal/template gates"
            ],
        }
    return {
        "value": "inconclusive",
        "reasons": ["no hard shortcut failure, but not every pre-registered A gate passed"],
    }


def evaluate_experiment(
    group_rows: dict[str, list[dict[str, Any]]],
    repetitions: int = 1000,
    seed: int = 20260715,
) -> dict[str, Any]:
    metrics = {group: compute_group_metrics(rows) for group, rows in group_rows.items()}
    confidence: dict[str, dict[str, Any]] = {}
    metric_paths = {
        "positive": (
            "diagnosis_accuracy",
            "diagnosis_macro_f1",
            "mean_iou",
            "pointing_game",
            "supported_diagnosis_rate",
        ),
        "counterfactual": (
            "null_fpr",
            "refusal_accuracy",
            "false_localization_rate",
            "predicted_box_on_null_rate",
            "ebhr",
            "prompt_compliance_error",
        ),
        "overall": ("balanced_accuracy", "presence_f1", "schema_valid_rate"),
    }
    for group, result in metrics.items():
        outcomes = result["row_outcomes"]
        confidence[group] = {}
        for section, names in metric_paths.items():
            if result[section]["samples"] == 0:
                continue
            for name in names:
                confidence[group][f"{section}.{name}"] = bootstrap_ci(
                    outcomes,
                    lambda sample, s=section, n=name: _metric_from_outcomes(sample, s, n),
                    repetitions,
                    seed,
                )
        for condition, condition_metrics in result["by_condition"].items():
            condition_outcomes = [row for row in outcomes if row["condition"] == condition]
            section = "positive" if condition == POSITIVE_CONDITION else "counterfactual"
            for name in metric_paths[section]:
                if name not in condition_metrics:
                    continue
                confidence[group][f"by_condition.{condition}.{name}"] = bootstrap_ci(
                    condition_outcomes,
                    lambda sample, s=section, n=name: _metric_from_outcomes(sample, s, n),
                    repetitions,
                    seed,
                )

    paired: dict[str, Any] = {}
    if "B1" in metrics and "B2" in metrics:
        left = metrics["B1"]["row_outcomes"]
        right = metrics["B2"]["row_outcomes"]
        left_by_id = {str(row["audit_id"]): row for row in left}
        right_by_id = {str(row["audit_id"]): row for row in right}
        if set(left_by_id) != set(right_by_id):
            raise ValueError("B1/B2 paired statistics require identical audit IDs")
        for condition in sorted({row["condition"] for row in left}):
            ids = sorted(
                audit_id
                for audit_id, row in left_by_id.items()
                if row["condition"] == condition
            )
            keys = ["presence_correct", "prompt_compliant"]
            if condition == POSITIVE_CONDITION:
                keys += ["diagnosis_correct", "supported"]
            else:
                keys += ["refusal_correct"]
            for key in keys:
                baseline = [{**left_by_id[audit_id], "value": bool(left_by_id[audit_id][key])} for audit_id in ids]
                model = [{**right_by_id[audit_id], "value": bool(right_by_id[audit_id][key])} for audit_id in ids]
                paired[f"{condition}.{key}"] = {
                    "bootstrap_delta": paired_bootstrap_delta(
                        baseline,
                        model,
                        lambda rows: _mean_boolean(rows, "value"),
                        repetitions,
                        seed,
                    ),
                    "mcnemar": exact_mcnemar(
                        [row["value"] for row in baseline],
                        [row["value"] for row in model],
                    ),
                }
        positive_left = [row for row in left if row["condition"] == POSITIVE_CONDITION]
        positive_right = [row for row in right if row["condition"] == POSITIVE_CONDITION]
        paired["original_correct.diagnosis_macro_f1"] = {
            "bootstrap_delta": paired_bootstrap_delta(
                positive_left,
                positive_right,
                lambda rows: _macro_f1(
                    [row["truth_diagnosis_id"] for row in rows],
                    [row["diagnosis_id"] for row in rows],
                ),
                repetitions,
                seed,
            )
        }
        paired["original_correct.mean_iou"] = {
            "bootstrap_delta": paired_bootstrap_delta(
                positive_left,
                positive_right,
                lambda rows: _safe_divide(
                    sum(float(row["iou"] or 0.0) for row in rows), len(rows)
                ),
                repetitions,
                seed,
            )
        }
    return {
        "groups": metrics,
        "confidence_intervals": confidence,
        "paired_b1_b2": paired,
        "decision": _decision(metrics, paired),
        "bootstrap": {"repetitions": repetitions, "seed": seed, "unit": "family_id"},
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 8 condition-aware metrics")
    parser.add_argument("--prediction", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group_rows: dict[str, list[dict[str, Any]]] = {}
    for path in args.prediction:
        rows = _load_jsonl(path)
        if not rows:
            raise ValueError(f"empty prediction file: {path}")
        groups = {str(row.get("group")) for row in rows}
        if len(groups) != 1:
            raise ValueError(f"prediction file must contain one group: {path}")
        group_rows[next(iter(groups))] = rows
    result = evaluate_experiment(group_rows, repetitions=1000, seed=20260715)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "groups": {
                    group: value["overall"] for group, value in result["groups"].items()
                },
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
