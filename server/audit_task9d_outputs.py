"""Read-only forensic analysis for frozen Task 9D outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OUTPUT_KEYS = (
    "evidence_present", "evidence_region", "visible_attributes", "diagnosis", "reliability"
)
DIAGNOSIS_KEYS = ("status", "pest_id", "pest_name", "species", "stage")
QUERY_PATTERN = re.compile(r"queried pest '([^']+)'", re.IGNORECASE)


def _tolerant_json(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def analyze_raw_output(raw: str) -> dict[str, Any]:
    strict_value = None
    try:
        candidate = json.loads(raw)
        if isinstance(candidate, dict):
            strict_value = candidate
    except json.JSONDecodeError:
        pass
    value = strict_value or _tolerant_json(raw)
    schema_valid = bool(
        isinstance(value, dict)
        and tuple(value) == OUTPUT_KEYS
        and isinstance(value.get("diagnosis"), dict)
        and tuple(value["diagnosis"]) == DIAGNOSIS_KEYS
    )
    issues: list[str] = []
    if value is None:
        issues.append("no_recoverable_json")
    else:
        present = value.get("evidence_present")
        region = value.get("evidence_region")
        if present is True:
            if not (
                isinstance(region, list) and len(region) == 4
                and all(isinstance(item, (int, float)) for item in region)
            ):
                issues.append("invalid_evidence_region")
        elif present is False:
            if region is not None:
                issues.append("invalid_evidence_region")
        else:
            issues.append("invalid_evidence_present")
        diagnosis = value.get("diagnosis")
        if not isinstance(diagnosis, dict) or tuple(diagnosis) != DIAGNOSIS_KEYS:
            issues.append("invalid_diagnosis_keys")
        expected_reliability = (
            "supported" if present is True else "insufficient_visual_evidence"
        )
        if value.get("reliability") != expected_reliability:
            issues.append("invalid_reliability")
    return {
        "strict_syntax_valid": strict_value is not None,
        "tolerant_json_recovered": value is not None,
        "strict_schema_valid": schema_valid,
        "issues": issues,
        "value": value,
    }


def _divide(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _normalize_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.casefold().split())


def _query_name(row: dict[str, Any]) -> str | None:
    for message in row.get("messages", []):
        for item in message.get("content", []):
            if item.get("type") != "text":
                continue
            match = QUERY_PATTERN.search(str(item.get("text", "")))
            if match:
                return _normalize_name(match.group(1))
    return None


def analyze_group_outputs(
    manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    truth = {str(row["id"]): row for row in manifest}
    predicted = {str(row["id"]): row for row in predictions}
    if len(truth) != len(manifest) or len(predicted) != len(predictions) or set(truth) != set(predicted):
        raise ValueError("forensic manifest/prediction IDs mismatch or duplicate")
    records = []
    for identifier in sorted(truth):
        row = truth[identifier]
        parsed = analyze_raw_output(str(predicted[identifier].get("raw_text", "")))
        value = parsed["value"]
        diagnosis = value.get("diagnosis") if isinstance(value, dict) else None
        diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
        records.append({
            "id": identifier,
            "family_id": str(row.get("family_id")),
            "role": str(row.get("role")),
            "condition": str(row.get("condition")),
            "prompt_view": str(row.get("prompt_view") or "canonical"),
            "query_class_id": row.get("query_class_id"),
            "query_name": _query_name(row),
            "evidence_present": value.get("evidence_present") if isinstance(value, dict) else None,
            "predicted_id": diagnosis.get("pest_id"),
            "predicted_name": _normalize_name(diagnosis.get("pest_name")),
            **{key: parsed[key] for key in (
                "strict_syntax_valid", "tolerant_json_recovered", "strict_schema_valid", "issues"
            )},
        })
    positives = [
        row for row in records
        if row["role"] == "positive" and row["prompt_view"] == "canonical"
    ]
    predicted_ids = [row["predicted_id"] for row in positives if isinstance(row["predicted_id"], int)]
    id_counts = Counter(predicted_ids)
    if len(id_counts) > 1:
        entropy = -sum((count / len(predicted_ids)) * math.log(count / len(predicted_ids)) for count in id_counts.values())
        normalized_entropy = entropy / math.log(len(id_counts))
    else:
        normalized_entropy = 0.0
    name_ids: dict[str, Counter[int]] = defaultdict(Counter)
    for row in positives:
        if row["predicted_name"] is not None and isinstance(row["predicted_id"], int):
            name_ids[row["predicted_name"]][row["predicted_id"]] += 1
    named_pairs = sum(sum(counts.values()) for counts in name_ids.values())
    consistent_pairs = sum(max(counts.values()) for counts in name_ids.values())
    evidence_rows = [row for row in positives if row["evidence_present"] is True]
    exact = lambda row: row["predicted_id"] == row["query_class_id"]
    positive_metrics = {
        "samples": len(positives),
        "evidence_present_rate": _divide(sum(row["evidence_present"] is True for row in positives), len(positives)),
        "query_name_echo_rate": _divide(sum(
            row["predicted_name"] is not None and row["predicted_name"] == row["query_name"]
            for row in positives
        ), len(positives)),
        "exact_id_accuracy": _divide(sum(exact(row) for row in positives), len(positives)),
        "conditional_id_accuracy_given_evidence": _divide(sum(exact(row) for row in evidence_rows), len(evidence_rows)),
        "unique_predicted_ids": len(id_counts),
        "top1_predicted_id_share": _divide(max(id_counts.values(), default=0), len(predicted_ids)),
        "top5_predicted_id_share": _divide(sum(count for _, count in id_counts.most_common(5)), len(predicted_ids)),
        "predicted_name_to_id_consistency": _divide(consistent_pairs, named_pairs),
        "normalized_predicted_id_entropy": float(normalized_entropy),
    }
    null_by_condition: dict[str, Any] = {}
    grouped_nulls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["role"] != "positive":
            grouped_nulls[row["condition"]].append(row)
    for condition, rows in sorted(grouped_nulls.items()):
        refusal = lambda row: (
            row["evidence_present"] is False
            and row["predicted_id"] is None
            and row["predicted_name"] is None
        )
        concrete = lambda row: isinstance(row["predicted_id"], int) or row["predicted_name"] is not None
        null_by_condition[condition] = {
            "samples": len(rows),
            "evidence_false_rate": _divide(sum(row["evidence_present"] is False for row in rows), len(rows)),
            "refusal_rate": _divide(sum(refusal(row) for row in rows), len(rows)),
            "concrete_diagnosis_rate": _divide(sum(concrete(row) for row in rows), len(rows)),
            "query_name_echo_rate": _divide(sum(
                row["predicted_name"] is not None and row["predicted_name"] == row["query_name"]
                for row in rows
            ), len(rows)),
        }
    issue_counts = Counter(issue for row in records for issue in row["issues"])
    format_metrics = {
        "samples": len(records),
        "strict_syntax_validity": _divide(sum(row["strict_syntax_valid"] for row in records), len(records)),
        "tolerant_json_recovery": _divide(sum(row["tolerant_json_recovered"] for row in records), len(records)),
        "strict_schema_validity": _divide(sum(row["strict_schema_valid"] for row in records), len(records)),
        "tolerant_recovery_gain": _divide(sum(row["tolerant_json_recovered"] for row in records), len(records))
        - _divide(sum(row["strict_syntax_valid"] for row in records), len(records)),
        "issue_counts": dict(sorted(issue_counts.items())),
    }
    return {
        "canonical_positive": positive_metrics,
        "null_by_condition": null_by_condition,
        "format": format_metrics,
        "row_records": records,
    }


def assess_label_map(schedules: list[dict[str, Any]]) -> dict[str, Any]:
    name_ids: dict[str, set[int]] = defaultdict(set)
    id_names: dict[int, set[str]] = defaultdict(set)
    query_name_matches = 0
    positives = 0
    invalid_targets = 0
    for row in schedules:
        if row.get("role") != "positive":
            continue
        positives += 1
        messages = row.get("model", {}).get("messages", [])
        query_name = _query_name({"messages": messages})
        assistant_text = None
        for message in messages:
            if message.get("role") == "assistant":
                for item in message.get("content", []):
                    if item.get("type") == "text":
                        assistant_text = str(item.get("text", ""))
                        break
        parsed = analyze_raw_output(assistant_text or "")
        value = parsed["value"]
        diagnosis = value.get("diagnosis") if isinstance(value, dict) else None
        diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
        pest_id = diagnosis.get("pest_id")
        pest_name = _normalize_name(diagnosis.get("pest_name"))
        if not isinstance(pest_id, int) or pest_name is None:
            invalid_targets += 1
            continue
        name_ids[pest_name].add(pest_id)
        id_names[pest_id].add(pest_name)
        query_name_matches += query_name is not None and query_name == pest_name
    names_with_multiple_ids = {
        name: sorted(ids) for name, ids in sorted(name_ids.items()) if len(ids) > 1
    }
    ids_with_multiple_names = {
        str(pest_id): sorted(names)
        for pest_id, names in sorted(id_names.items()) if len(names) > 1
    }
    return {
        "positive_targets": positives,
        "invalid_positive_targets": invalid_targets,
        "unique_names": len(name_ids),
        "unique_ids": len(id_names),
        "query_target_name_match_rate": _divide(query_name_matches, positives),
        "names_with_multiple_ids": names_with_multiple_ids,
        "ids_with_multiple_names": ids_with_multiple_names,
        "consistent": not names_with_multiple_ids and not ids_with_multiple_names and invalid_targets == 0,
    }


def choose_dominant_cause(
    group_reports: dict[str, dict[str, Any]], label_map: dict[str, Any]
) -> dict[str, Any]:
    if not label_map.get("consistent", False):
        return {
            "value": "label_map_corruption",
            "evidence": {"label_map_consistent": False},
            "rule": "source target name-ID mapping must be one-to-one before model diagnosis",
        }
    adapters = [report for group, report in group_reports.items() if group != "Base"]
    if not adapters:
        raise ValueError("dominant-cause gate requires adapter group reports")
    mean = lambda path: sum(float(report[path[0]][path[1]]) for report in adapters) / len(adapters)
    evidence = {
        "label_map_consistent": True,
        "mean_query_name_echo_rate": mean(("canonical_positive", "query_name_echo_rate")),
        "mean_exact_id_accuracy": mean(("canonical_positive", "exact_id_accuracy")),
        "mean_predicted_name_to_id_consistency": mean((
            "canonical_positive", "predicted_name_to_id_consistency"
        )),
        "mean_positive_evidence_present_rate": mean((
            "canonical_positive", "evidence_present_rate"
        )),
        "mean_tolerant_recovery_gain": mean(("format", "tolerant_recovery_gain")),
    }
    if (
        evidence["mean_query_name_echo_rate"] >= 0.80
        and evidence["mean_exact_id_accuracy"] < 0.10
        and evidence["mean_predicted_name_to_id_consistency"] < 0.50
    ):
        value = "numeric_id_generation_bottleneck"
        rule = "clean source map + query-name echo >=80% + exact ID <10% + name-ID consistency <50%"
    elif evidence["mean_positive_evidence_present_rate"] < 0.50:
        value = "positive_over_abstention"
        rule = "canonical-positive evidence-present rate <50%"
    elif evidence["mean_tolerant_recovery_gain"] >= 0.20:
        value = "format_bottleneck"
        rule = "adapter tolerant JSON recovery improves usable output by >=20pp"
    else:
        value = "visual_or_objective_bottleneck"
        rule = "no earlier deterministic cause gate fired"
    return {"value": value, "evidence": evidence, "rule": rule}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_forensics_artifacts(
    output_root: Path, report: dict[str, Any], cases: list[dict[str, Any]]
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "forensics_report.json"
    cases_path = output_root / "forensics_cases.jsonl"
    summary_path = output_root / "run_summary.json"
    _write_text_atomic(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    _write_text_atomic(
        cases_path,
        "".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases),
    )
    summary = {
        "version": "task9d-output-forensics-summary-v1",
        "state": "completed",
        "case_count": len(cases),
        "dominant_cause": report.get("dominant_cause", {}).get("value"),
        "report_sha256": _sha256(report_path),
        "cases_sha256": _sha256(cases_path),
        "code_sha256": _sha256(Path(__file__)),
    }
    _write_text_atomic(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    artifacts = (report_path, cases_path, summary_path)
    (output_root / "completion.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _representative_cases(group: str, records: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        case_type = None
        if not row["strict_syntax_valid"]:
            case_type = "strict_format_invalid"
        if (
            row["role"] == "positive" and row["prompt_view"] == "canonical"
            and row["evidence_present"] is True
            and row["predicted_name"] is not None
            and row["predicted_name"] == row["query_name"]
            and row["predicted_id"] != row["query_class_id"]
        ):
            case_type = "name_echo_id_mismatch"
        elif (
            row["role"] == "positive" and row["prompt_view"] == "canonical"
            and row["evidence_present"] is False
        ):
            case_type = "positive_abstention"
        elif (
            row["condition"] == "semantic_null"
            and (isinstance(row["predicted_id"], int) or row["predicted_name"] is not None)
        ):
            case_type = "semantic_null_concrete_diagnosis"
        if case_type is not None and len(buckets[case_type]) < limit:
            buckets[case_type].append({
                "id": row["id"], "family_id": row["family_id"], "group": group,
                "case_type": case_type, "role": row["role"], "condition": row["condition"],
                "query_class_id": row["query_class_id"], "query_name": row["query_name"],
                "predicted_id": row["predicted_id"], "predicted_name": row["predicted_name"],
                "evidence_present": row["evidence_present"], "issues": row["issues"],
            })
    return [case for case_type in sorted(buckets) for case in buckets[case_type]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Task 9D outputs without training")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, action="append", required=True)
    parser.add_argument("--formal-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    groups = ("Base", "A17", "A29", "A43", "B17", "B29", "B43", "C17", "C29", "C43")
    manifest = _read_jsonl(args.manifest)
    group_reports: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {
        "manifest": _sha256(args.manifest),
        "formal_report": _sha256(args.formal_report),
    }
    for group in groups:
        prediction_path = args.inference_root / group / "predictions.jsonl"
        result = analyze_group_outputs(manifest, _read_jsonl(prediction_path))
        records = result.pop("row_records")
        group_reports[group] = result
        cases.extend(_representative_cases(group, records))
        input_hashes[f"predictions/{group}"] = _sha256(prediction_path)
    schedules: list[dict[str, Any]] = []
    for index, path in enumerate(args.schedule):
        schedules.extend(_read_jsonl(path))
        input_hashes[f"schedule/{index}"] = _sha256(path)
    label_map = assess_label_map(schedules)
    dominant = choose_dominant_cause(group_reports, label_map)
    formal = json.loads(args.formal_report.read_text(encoding="utf-8"))
    report = {
        "version": "task9d-output-forensics-v1",
        "input_sha256": input_hashes,
        "label_map": label_map,
        "groups": group_reports,
        "dominant_cause": dominant,
        "formal_task9d_decision": formal.get("decision"),
        "task8_locked_set_read": False,
        "training_started": False,
    }
    write_forensics_artifacts(args.output_root, report, cases)
    print(json.dumps({
        "state": "completed",
        "dominant_cause": dominant["value"],
        "cases": len(cases),
    }, indent=2))


if __name__ == "__main__":
    main()
