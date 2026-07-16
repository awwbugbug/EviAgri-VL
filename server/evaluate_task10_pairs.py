"""Hallusion-style family/pair forensics for frozen Task 10A predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evaluate_task9d import _parse
from task10_audit_common import sha256_file, write_json_new


NULL_CONDITIONS = (
    "semantic_null",
    "source_visual_null",
    "blank",
    "blur",
    "shuffle",
)
REQUIRED_CONDITIONS = ("original",) + NULL_CONDITIONS


def _rate(values: Iterable[bool | int | float]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else 0.0


def _prediction_map(predictions: Any) -> dict[str, dict[str, Any]]:
    if isinstance(predictions, dict):
        result = {str(identifier): value for identifier, value in predictions.items()}
    else:
        rows = list(predictions)
        identifiers = [str(row.get("id")) for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate prediction ID")
        result = {str(row["id"]): row for row in rows}
    if not all(isinstance(value, dict) for value in result.values()):
        raise ValueError("prediction entries must be objects")
    return result


def _parse_prediction(entry: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse(str(entry.get("raw_text", "")))
    value = parsed["value"] if parsed["semantic_consistent"] else None
    diagnosis = value.get("diagnosis") if isinstance(value, dict) else None
    concrete = bool(
        isinstance(value, dict)
        and (
            value.get("evidence_present") is True
            or (
                isinstance(diagnosis, dict)
                and (
                    diagnosis.get("status") == "supported"
                    or any(
                        diagnosis.get(key) is not None
                        for key in ("pest_id", "pest_name", "species", "stage")
                    )
                )
            )
        )
    )
    rejected = bool(
        isinstance(value, dict)
        and value.get("evidence_present") is False
        and value.get("evidence_region") is None
        and isinstance(diagnosis, dict)
        and diagnosis.get("status") in {"abstain", "uncertain"}
        and all(
            diagnosis.get(key) is None
            for key in ("pest_id", "pest_name", "species", "stage")
        )
    )
    return {
        "valid": bool(parsed["semantic_consistent"]),
        "value": value,
        "concrete": concrete,
        "rejected": rejected,
    }


def _select_family_rows(manifest: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    identifiers = [str(row.get("id")) for row in manifest]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate manifest ID")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest:
        condition = str(row.get("condition"))
        if str(row.get("prompt_view")) != "canonical" or condition not in REQUIRED_CONDITIONS:
            continue
        family = str(row.get("family_id"))
        if condition in grouped[family]:
            raise ValueError(f"duplicate family condition: {family}/{condition}")
        grouped[family][condition] = row
    if not grouped:
        raise ValueError("family condition set is empty")
    for family, conditions in grouped.items():
        if set(conditions) != set(REQUIRED_CONDITIONS):
            raise ValueError(
                f"family condition set mismatch for {family}: {sorted(conditions)}"
            )
        if str(conditions["original"].get("role")) != "positive":
            raise ValueError(f"family original is not positive: {family}")
        if any(
            str(conditions[condition].get("role")) == "positive"
            for condition in NULL_CONDITIONS
        ):
            raise ValueError(f"family null condition marked positive: {family}")
    return dict(sorted(grouped.items()))


def evaluate_family_pairs(
    manifest: Iterable[dict[str, Any]],
    predictions: Any,
) -> dict[str, Any]:
    manifest_rows = list(manifest)
    families = _select_family_rows(manifest_rows)
    predicted = _prediction_map(predictions)
    required_ids = {
        str(row["id"])
        for conditions in families.values()
        for row in conditions.values()
    }
    missing = sorted(required_ids - set(predicted))
    if missing:
        raise ValueError(f"prediction ID mismatch: missing {len(missing)} required IDs")

    invalid_count = 0
    family_outcomes = []
    for family, conditions in families.items():
        parsed = {
            condition: _parse_prediction(predicted[str(row["id"])])
            for condition, row in conditions.items()
        }
        invalid_count += sum(not result["valid"] for result in parsed.values())
        original_value = parsed["original"]["value"]
        diagnosis = (
            original_value.get("diagnosis")
            if isinstance(original_value, dict)
            else None
        )
        original_success = bool(
            parsed["original"]["valid"]
            and parsed["original"]["concrete"]
            and isinstance(diagnosis, dict)
            and diagnosis.get("pest_id") == int(conditions["original"]["query_class_id"])
        )
        null_rejected = {
            condition: parsed[condition]["valid"] and parsed[condition]["rejected"]
            for condition in NULL_CONDITIONS
        }
        pair_success = {
            condition: original_success and null_rejected[condition]
            for condition in NULL_CONDITIONS
        }
        contradiction = bool(
            not parsed["original"]["concrete"]
            and any(parsed[condition]["concrete"] for condition in NULL_CONDITIONS)
        )
        family_outcomes.append({
            "family_id": family,
            "original_success": original_success,
            "original_concrete": parsed["original"]["concrete"],
            "null_rejected": null_rejected,
            "null_concrete": {
                condition: parsed[condition]["concrete"]
                for condition in NULL_CONDITIONS
            },
            "pair_success": pair_success,
            "strict_family_success": original_success and all(null_rejected.values()),
            "contradiction": contradiction,
        })

    by_condition = {}
    for condition in NULL_CONDITIONS:
        by_condition[condition] = {
            "samples": len(family_outcomes),
            "null_fpr": _rate(
                not outcome["null_rejected"][condition] for outcome in family_outcomes
            ),
            "pair_success": _rate(
                outcome["pair_success"][condition] for outcome in family_outcomes
            ),
            "concrete_diagnosis_drop": _rate(
                int(outcome["original_concrete"])
                - int(outcome["null_concrete"][condition])
                for outcome in family_outcomes
            ),
        }
    return {
        "version": "task10a-family-pair-audit-v1",
        "family_count": len(family_outcomes),
        "evaluated_prediction_count": len(required_ids),
        "invalid_prediction_count": invalid_count,
        "original_positive_tpr": _rate(
            outcome["original_success"] for outcome in family_outcomes
        ),
        "strict_family_success": _rate(
            outcome["strict_family_success"] for outcome in family_outcomes
        ),
        "contradiction_rate": _rate(
            outcome["contradiction"] for outcome in family_outcomes
        ),
        "by_condition": by_condition,
        "families": family_outcomes,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()

    report = evaluate_family_pairs(
        _read_jsonl(args.manifest),
        _read_jsonl(args.predictions),
    )
    report["group"] = args.group
    report["inputs"] = {
        "manifest_sha256": sha256_file(args.manifest),
        "predictions_sha256": sha256_file(args.predictions),
    }
    write_json_new(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
