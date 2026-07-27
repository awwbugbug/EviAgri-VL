"""Compile a complete Task 11A.3 reviewer CSV from an explicit rejection declaration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_task11a3_human_review_bundle import CRITERIA, validate_review_rows
from task10_audit_common import write_json_new


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compile_review(*, manifest_path: Path, declaration_path: Path, output_csv: Path) -> dict:
    manifest = _read_jsonl(Path(manifest_path))
    expected_ids = {str(row["audit_id"]) for row in manifest}
    if len(expected_ids) != len(manifest):
        raise ValueError("review manifest audit IDs are not unique")
    declaration = json.loads(Path(declaration_path).read_text(encoding="utf-8"))
    reviewer_id = str(declaration.get("reviewer_id", "")).strip()
    declaration_status = declaration.get("status")
    if declaration_status not in {"DRAFT_PENDING_CONFIRMATION", "CONFIRMED"}:
        raise ValueError("declaration status must be DRAFT_PENDING_CONFIRMATION or CONFIRMED")
    failures = declaration.get("failures", {})
    unknown_criteria = set(failures) - set(CRITERIA)
    if unknown_criteria:
        raise ValueError(f"unknown review criteria: {sorted(unknown_criteria)}")
    failure_reasons: dict[str, list[str]] = {audit_id: [] for audit_id in expected_ids}
    for criterion, ids in failures.items():
        for audit_id in ids:
            if audit_id not in expected_ids:
                raise ValueError(f"unknown audit_id: {audit_id}")
            if criterion in failure_reasons[audit_id]:
                raise ValueError(f"duplicate failure assignment: {audit_id}/{criterion}")
            failure_reasons[audit_id].append(criterion)
    uncertain = declaration.get("uncertain", {})
    if set(uncertain) - set(CRITERIA):
        raise ValueError("unknown uncertain criteria")
    uncertain_reasons: dict[str, list[str]] = {audit_id: [] for audit_id in expected_ids}
    for criterion, ids in uncertain.items():
        for audit_id in ids:
            if audit_id not in expected_ids:
                raise ValueError(f"unknown audit_id: {audit_id}")
            if criterion in failure_reasons[audit_id]:
                raise ValueError(f"same criterion cannot be FAIL and UNCERTAIN: {audit_id}")
            uncertain_reasons[audit_id].append(criterion)
    rows = []
    for audit_id in sorted(expected_ids):
        row = {"audit_id": audit_id, "reviewer_id": reviewer_id}
        for criterion in CRITERIA:
            if criterion in failure_reasons[audit_id]:
                row[criterion] = "FAIL"
            elif criterion in uncertain_reasons[audit_id]:
                row[criterion] = "UNCERTAIN"
            else:
                row[criterion] = "PASS"
        reasons = [f"FAIL:{value}" for value in failure_reasons[audit_id]]
        reasons += [f"UNCERTAIN:{value}" for value in uncertain_reasons[audit_id]]
        row["notes"] = ";".join(reasons)
        rows.append(row)
    validation = validate_review_rows(rows, expected_ids)
    output_csv = Path(output_csv)
    if output_csv.exists():
        raise FileExistsError(f"refusing to overwrite reviewer draft: {output_csv}")
    with output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "reviewer_id", *CRITERIA, "notes"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "version": "task11a3-reviewer-draft-1",
        "state": "completed",
        "status": declaration_status,
        "reviewer_id": reviewer_id,
        "audit_rows": len(rows),
        "decision_counts": validation["counts"],
        "failed_unique_rows": sum(bool(values) for values in failure_reasons.values()),
        "uncertain_unique_rows": sum(bool(values) for values in uncertain_reasons.values()),
        "unlisted_rows_assumed_all_pass": sum(
            not failure_reasons[audit_id] and not uncertain_reasons[audit_id]
            for audit_id in expected_ids
        ),
    }
    write_json_new(output_csv.with_suffix(".summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_review(
        manifest_path=args.manifest,
        declaration_path=args.declaration,
        output_csv=args.output_csv,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
