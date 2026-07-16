from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_task8 import parse_task8_prediction
from task8_protocol import GROUPS, protocol_hash


def _schema_counts(predictions: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
    valid = invalid = 0
    for rows in predictions.values():
        for row in rows:
            try:
                parse_task8_prediction(str(row["prediction"]))
                valid += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
    return valid, invalid


def validate_smoke(
    audit_rows: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    leakage_report: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    run_summary: dict[str, Any],
    failure_files: list[str],
    peak_vram_bytes: int,
    gpu_memory_bytes: int,
) -> dict[str, Any]:
    families = {str(row.get("family_id")) for row in audit_rows}
    job_ids = [str(job.get("job_id")) for job in jobs]
    prediction_rows = [row for group in GROUPS for row in predictions.get(group, [])]
    prediction_ids = [str(row.get("job_id")) for row in prediction_rows]
    schema_valid, schema_invalid = _schema_counts(predictions)

    audit_sha = {
        str(row.get("audit_id")): str(row.get("image_sha256")) for row in audit_rows
    }
    jobs_by_audit: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        jobs_by_audit.setdefault(str(job.get("audit_id")), []).append(job)
    pixels_match = all(
        len(rows) == 4
        and {str(row.get("group")) for row in rows} == set(GROUPS)
        and len({str(row.get("image_sha256")) for row in rows}) == 1
        and str(rows[0].get("image_sha256")) == audit_sha.get(audit_id)
        for audit_id, rows in jobs_by_audit.items()
    ) and set(jobs_by_audit) == set(audit_sha)
    jobs_by_id = {str(job.get("job_id")): job for job in jobs}
    prediction_pixels_match = all(
        row.get("job_id") in jobs_by_id
        and row.get("image_sha256") == jobs_by_id[str(row.get("job_id"))].get("image_sha256")
        for row in prediction_rows
    )

    registered_protocols = all(
        job.get("group") in GROUPS
        and job.get("protocol_hash") == protocol_hash(str(job.get("group")))
        for job in jobs
    )
    b1_b2_equal = all(
        next(row for row in rows if row.get("group") == "B1").get("protocol_hash")
        == next(row for row in rows if row.get("group") == "B2").get("protocol_hash")
        and next(row for row in rows if row.get("group") == "B1").get("prompt")
        == next(row for row in rows if row.get("group") == "B2").get("prompt")
        for rows in jobs_by_audit.values()
        if {row.get("group") for row in rows} == set(GROUPS)
    )

    checks = {
        "manifest_counts": len(families) == 4 and len(audit_rows) == 24 and len(jobs) == 96,
        "unique_job_ids": len(job_ids) == len(set(job_ids)) == 96,
        "registered_protocol_hashes": registered_protocols,
        "b1_b2_same_protocol": b1_b2_equal and len(jobs_by_audit) == 24,
        "leakage_passed": leakage_report.get("passed") is True,
        "prediction_counts": all(len(predictions.get(group, [])) == 24 for group in GROUPS)
        and len(prediction_rows) == 96,
        "prediction_ids_match_jobs": len(prediction_ids) == len(set(prediction_ids))
        and set(prediction_ids) == set(job_ids),
        "schema_accounting_complete": schema_valid + schema_invalid == len(prediction_rows),
        "image_sha_equality": pixels_match and prediction_pixels_match,
        "run_completed": run_summary.get("completed") is True,
        "no_failure_reports": not failure_files,
        "vram_within_capacity": 0 <= peak_vram_bytes < gpu_memory_bytes,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "families": len(families),
            "audit_rows": len(audit_rows),
            "jobs": len(jobs),
            "predictions": len(prediction_rows),
            "schema_valid": schema_valid,
            "schema_invalid": schema_invalid,
        },
        "peak_vram_bytes": peak_vram_bytes,
        "gpu_memory_bytes": gpu_memory_bytes,
        "failure_files": failure_files,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Task 8 four-family smoke")
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--inference-jobs", required=True, type=Path)
    parser.add_argument("--leakage-report", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_summary = json.loads((args.run_root / "run_summary.json").read_text(encoding="utf-8"))
    result = validate_smoke(
        _load_jsonl(args.audit_manifest),
        _load_jsonl(args.inference_jobs),
        json.loads(args.leakage_report.read_text(encoding="utf-8")),
        {
            group: _load_jsonl(args.run_root / group / "predictions.jsonl")
            for group in GROUPS
        },
        run_summary,
        [str(path) for path in args.run_root.glob("failure*.json")],
        int(run_summary.get("peak_vram_bytes", -1)),
        int(run_summary.get("gpu_memory_bytes", 0)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
