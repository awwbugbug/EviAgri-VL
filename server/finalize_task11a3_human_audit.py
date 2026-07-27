"""Merge Task 11A.3 dual reviews and adjudication into a signed final audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_task11a3_human_review_bundle import CRITERIA, review_decision, validate_review_rows
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"columns must exactly match frozen schema: {columns}")
        return list(reader)


def _failures(row: dict[str, str]) -> list[str]:
    return [criterion for criterion in CRITERIA if row[criterion].strip().upper() == "FAIL"]


def finalize_human_audit(
    *, manifest_path: Path, reviewer_a_path: Path, reviewer_b_path: Path,
    adjudication_path: Path, output_root: Path,
) -> dict:
    manifest_path, reviewer_a_path, reviewer_b_path, adjudication_path = map(
        Path, (manifest_path, reviewer_a_path, reviewer_b_path, adjudication_path)
    )
    manifest = _read_jsonl(manifest_path)
    expected_ids = {str(row["audit_id"]) for row in manifest}
    if len(expected_ids) != len(manifest):
        raise ValueError("manifest audit IDs are not unique")
    review_columns = ["audit_id", "reviewer_id", *CRITERIA, "notes"]
    rows_a = _read_csv(reviewer_a_path, review_columns)
    rows_b = _read_csv(reviewer_b_path, review_columns)
    validation_a = validate_review_rows(rows_a, expected_ids)
    validation_b = validate_review_rows(rows_b, expected_ids)
    if validation_a["reviewer_id"] == validation_b["reviewer_id"]:
        raise ValueError("reviewers must be distinct")
    by_a = {row["audit_id"]: row for row in rows_a}
    by_b = {row["audit_id"]: row for row in rows_b}
    disagreement_ids = {
        audit_id for audit_id in expected_ids
        if review_decision(by_a[audit_id]) != review_decision(by_b[audit_id])
        or "UNCERTAIN" in {review_decision(by_a[audit_id]), review_decision(by_b[audit_id])}
    }
    adjudication_columns = ["audit_id", "adjudicator_id", "final_decision", "failed_criteria", "notes"]
    adjudication = _read_csv(adjudication_path, adjudication_columns)
    adjudication_ids = [row["audit_id"].strip() for row in adjudication]
    if len(adjudication_ids) != len(set(adjudication_ids)) or set(adjudication_ids) != disagreement_ids:
        raise ValueError("adjudication must cover every disagreement exactly once")
    adjudicators = {row["adjudicator_id"].strip() for row in adjudication}
    if "" in adjudicators or len(adjudicators) != 1:
        raise ValueError("one non-empty adjudicator_id is required")
    by_adjudication = {row["audit_id"].strip(): row for row in adjudication}
    allowed_decisions = {"KEEP", "EXCLUDE", "UNCERTAIN"}
    for row in adjudication:
        decision = row["final_decision"].strip().upper()
        failures = [value for value in row["failed_criteria"].split(";") if value]
        if decision not in allowed_decisions or set(failures) - set(CRITERIA):
            raise ValueError(f"invalid adjudication for {row['audit_id']}")
        if decision == "EXCLUDE" and not failures:
            raise ValueError(f"EXCLUDE requires a failed criterion: {row['audit_id']}")
        if decision != "EXCLUDE" and failures:
            raise ValueError(f"non-EXCLUDE cannot carry failed criteria: {row['audit_id']}")
    final_rows = []
    for audit_id in sorted(expected_ids):
        decision_a, decision_b = review_decision(by_a[audit_id]), review_decision(by_b[audit_id])
        failures_a, failures_b = _failures(by_a[audit_id]), _failures(by_b[audit_id])
        if audit_id in by_adjudication:
            adjudicated = by_adjudication[audit_id]
            final_decision = adjudicated["final_decision"].strip().upper()
            final_failures = [value for value in adjudicated["failed_criteria"].split(";") if value]
            source = "joint_adjudication"
        elif decision_a == decision_b == "PASS":
            final_decision, final_failures, source = "KEEP", [], "dual_agreement"
        elif decision_a == decision_b == "REJECT":
            final_decision = "EXCLUDE"
            final_failures = sorted(set(failures_a) | set(failures_b))
            source = "dual_agreement"
        else:
            raise ValueError(f"unresolved review state for {audit_id}")
        final_rows.append({
            "audit_id": audit_id,
            "final_decision": final_decision,
            "strict_real_null_eligible": final_decision == "KEEP",
            "resolution_source": source,
            "final_failed_criteria": final_failures,
            "reviewer_a_decision": decision_a,
            "reviewer_a_failures": failures_a,
            "reviewer_b_decision": decision_b,
            "reviewer_b_failures": failures_b,
        })
    counts = {decision: sum(row["final_decision"] == decision for row in final_rows) for decision in allowed_decisions}
    if sum(counts.values()) != len(expected_ids):
        raise ValueError("final decision counts do not reconcile")
    output_root = Path(output_root)
    ensure_new_directory(output_root)
    with (output_root / "final_human_audit.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_root / "strict_real_null_ids.txt").write_text(
        "\n".join(row["audit_id"] for row in final_rows if row["strict_real_null_eligible"]) + "\n",
        encoding="utf-8", newline="\n",
    )
    (output_root / "excluded_ids.txt").write_text(
        "\n".join(row["audit_id"] for row in final_rows if row["final_decision"] == "EXCLUDE") + "\n",
        encoding="utf-8", newline="\n",
    )
    report = {
        "version": "task11a3-final-human-audit-1",
        "state": "completed",
        "human_audit_status": "COMPLETE",
        "decision": "READY_FOR_FROZEN_ROUTER",
        "audit_rows": len(final_rows),
        "final_decision_counts": counts,
        "strict_real_null_candidates": counts["KEEP"],
        "adjudicator_id": next(iter(adjudicators)),
        "resolution_counts": {
            "dual_agreement": sum(row["resolution_source"] == "dual_agreement" for row in final_rows),
            "joint_adjudication": sum(row["resolution_source"] == "joint_adjudication" for row in final_rows),
        },
        "input_sha256": {
            "manifest": sha256_file(manifest_path),
            "reviewer_a": sha256_file(reviewer_a_path),
            "reviewer_b": sha256_file(reviewer_b_path),
            "adjudication": sha256_file(adjudication_path),
        },
        "raw_data_deleted": False,
    }
    write_json_new(output_root / "final_audit_report.json", report)
    signed = ["final_human_audit.jsonl", "strict_real_null_ids.txt", "excluded_ids.txt", "final_audit_report.json"]
    with (output_root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in signed:
            handle.write(f"{sha256_file(output_root / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_human_audit(
        manifest_path=args.manifest,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        adjudication_path=args.adjudication,
        output_root=args.output_root,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
