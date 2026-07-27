"""Validate two independent Task 11A.3 reviews and emit adjudication inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_task11a3_human_review_bundle import CRITERIA, review_decision, validate_review_rows
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["audit_id", "reviewer_id", *CRITERIA, "notes"]
        if reader.fieldnames != required:
            raise ValueError(f"review columns must exactly match frozen schema: {required}")
        return list(reader)


def _failures(row: dict[str, str]) -> list[str]:
    return [criterion for criterion in CRITERIA if row[criterion].strip().upper() == "FAIL"]


def _validate_notes(rows: list[dict[str, str]]) -> None:
    for row in rows:
        expected = {f"FAIL:{criterion}" for criterion in _failures(row)}
        actual = {value for value in row["notes"].split(";") if value}
        if actual != expected:
            raise ValueError(f"notes/failure mismatch for {row['audit_id']}")


def validate_dual_review(
    *, manifest_path: Path, reviewer_a_path: Path, reviewer_b_path: Path, output_root: Path
) -> dict:
    manifest_path, reviewer_a_path, reviewer_b_path = map(
        Path, (manifest_path, reviewer_a_path, reviewer_b_path)
    )
    manifest = _read_jsonl(manifest_path)
    expected_ids = {str(row["audit_id"]) for row in manifest}
    if len(expected_ids) != len(manifest):
        raise ValueError("manifest audit IDs are not unique")
    rows_a, rows_b = _read_review(reviewer_a_path), _read_review(reviewer_b_path)
    validation_a = validate_review_rows(rows_a, expected_ids)
    validation_b = validate_review_rows(rows_b, expected_ids)
    if validation_a["reviewer_id"] == validation_b["reviewer_id"]:
        raise ValueError("reviewer IDs must be distinct")
    _validate_notes(rows_a)
    _validate_notes(rows_b)
    by_a = {row["audit_id"]: row for row in rows_a}
    by_b = {row["audit_id"]: row for row in rows_b}
    categories = {
        "both_pass": 0,
        "both_reject": 0,
        "a_reject_b_pass": 0,
        "a_pass_b_reject": 0,
        "uncertain_involved": 0,
    }
    disagreements = []
    for audit_id in sorted(expected_ids):
        decision_a, decision_b = review_decision(by_a[audit_id]), review_decision(by_b[audit_id])
        if "UNCERTAIN" in {decision_a, decision_b}:
            category = "uncertain_involved"
        elif decision_a == decision_b == "PASS":
            category = "both_pass"
        elif decision_a == decision_b == "REJECT":
            category = "both_reject"
        elif decision_a == "REJECT":
            category = "a_reject_b_pass"
        else:
            category = "a_pass_b_reject"
        categories[category] += 1
        if decision_a != decision_b or "UNCERTAIN" in {decision_a, decision_b}:
            disagreements.append({
                "audit_id": audit_id,
                "reviewer_a_decision": decision_a,
                "reviewer_a_failures": _failures(by_a[audit_id]),
                "reviewer_b_decision": decision_b,
                "reviewer_b_failures": _failures(by_b[audit_id]),
            })
    total = len(expected_ids)
    observed = (categories["both_pass"] + categories["both_reject"]) / total
    a_reject = validation_a["counts"]["REJECT"] / total
    b_reject = validation_b["counts"]["REJECT"] / total
    expected = a_reject * b_reject + (1 - a_reject) * (1 - b_reject)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    output_root = Path(output_root)
    ensure_new_directory(output_root)
    with (output_root / "adjudication_manifest.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "version": "task11a3-dual-review-validation-1",
        "state": "completed",
        "decision": "PENDING_DISAGREEMENT_ADJUDICATION" if disagreements else "DUAL_REVIEW_COMPLETE",
        "audit_rows": total,
        "reviewer_a": validation_a,
        "reviewer_b": validation_b,
        "agreement_counts": categories,
        "agreement_rate": observed,
        "cohen_kappa": kappa,
        "disagreement_rows": len(disagreements),
        "input_sha256": {
            "manifest": sha256_file(manifest_path),
            "reviewer_a": sha256_file(reviewer_a_path),
            "reviewer_b": sha256_file(reviewer_b_path),
        },
    }
    write_json_new(output_root / "dual_review_report.json", report)
    with (output_root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in ("adjudication_manifest.jsonl", "dual_review_report.json"):
            handle.write(f"{sha256_file(output_root / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_dual_review(
        manifest_path=args.manifest,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        output_root=args.output_root,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
