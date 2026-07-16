"""Pre-registered elimination and selection rules for Task 9D."""

from __future__ import annotations

import statistics
from typing import Any


def _aggregate(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "accuracy": lambda m: m["positive"]["accuracy"],
        "macro_f1": lambda m: m["positive"]["macro_f1"],
        "null_fpr": lambda m: m["null"]["overall_null_fpr"],
        "supported_diagnosis_rate": lambda m: m["positive"]["supported_diagnosis_rate"],
        "prompt_gap": lambda m: m["prompt_gap"],
    }
    result: dict[str, Any] = {"seeds": len(seeds)}
    for name, extract in paths.items():
        values = [float(extract(seed)) for seed in seeds]
        result[name] = {
            "mean": statistics.fmean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "worst": min(values) if name in {"accuracy", "macro_f1", "supported_diagnosis_rate"} else max(values),
            "values": values,
        }
    return result


def decide_task9d(base: dict[str, Any], variants: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if set(variants) != {"A", "B", "C"} or any(len(rows) != 3 for rows in variants.values()):
        raise ValueError("Task 9D decision requires exactly A/B/C with three seeds each")
    base_accuracy = float(base["positive"]["accuracy"])
    reports: dict[str, Any] = {}
    for variant in ("A", "B", "C"):
        seeds = variants[variant]
        reasons = []
        for index, metrics in enumerate(seeds):
            label = f"seed_index_{index}"
            if not metrics.get("shortcut_gate_passed", False):
                reasons.append(f"{label}: shortcut gate failed")
            if float(metrics["positive"]["accuracy"]) - base_accuracy < -0.03:
                reasons.append(f"{label}: accuracy below Base by more than 3pp")
            if float(metrics.get("macro_f1_delta_ci", {}).get("high", 0.0)) < 0:
                reasons.append(f"{label}: Macro-F1 significantly below Base")
            for condition in ("blank", "blur"):
                if float(metrics["by_condition"][condition]["null_fpr"]) >= 0.10:
                    reasons.append(f"{label}: {condition} Null FPR >=10%")
            if float(metrics["prompt_gap"]) >= 0.05:
                reasons.append(f"{label}: neutral/training prompt gap >=5pp")
            json_metrics = metrics["json"]
            if float(json_metrics["schema_validity"]) < 0.99:
                reasons.append(f"{label}: schema validity below 99%")
            if float(json_metrics["semantic_consistency"]) < 0.99 or float(json_metrics["task_compliance"]) < 0.99:
                reasons.append(f"{label}: semantic consistency or task compliance below 99%")
        reports[variant] = {"passed": not reasons, "reasons": reasons, "aggregate": _aggregate(seeds)}
    eligible = [variant for variant in ("A", "B", "C") if reports[variant]["passed"]]
    selected = None
    if eligible:
        selected = min(eligible, key=lambda variant: (
            -reports[variant]["aggregate"]["macro_f1"]["mean"],
            reports[variant]["aggregate"]["null_fpr"]["mean"],
            reports[variant]["aggregate"]["null_fpr"]["worst"],
            -reports[variant]["aggregate"]["supported_diagnosis_rate"]["mean"],
            reports[variant]["aggregate"]["macro_f1"]["sample_std"],
        ))
    scientific = False
    if selected is not None:
        selected_seeds = variants[selected]
        aggregate = reports[selected]["aggregate"]
        diagnosis_not_worse = aggregate["accuracy"]["mean"] >= base_accuracy - 0.03
        reliability_significant = all(
            float(seed.get("null_fpr_delta_ci", {}).get("high", 1.0)) < 0 for seed in selected_seeds
        )
        localization_significant = all(
            float(seed.get("localization_delta_ci", {}).get("low", -1.0)) > 0 for seed in selected_seeds
        )
        scientific = diagnosis_not_worse and (reliability_significant or localization_significant)
    return {
        "version": "task9d-decision-v1", "variants": reports,
        "selected_protocol": selected,
        "protocol_repair_passed": selected is not None,
        "scientific_passed": scientific,
        "authorize_9e_recommendation": scientific,
        "note": "Recommendation only; this function never starts Task 9E.",
    }
