import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task8_protocol import protocol_hash
from validate_task8_smoke import validate_smoke


def valid_output() -> str:
    return json.dumps(
        {
            "evidence_present": False,
            "evidence_bbox": None,
            "visible_attributes": [],
            "diagnosis": "uncertain",
            "reliability": "insufficient_visual_evidence",
        },
        separators=(",", ":"),
    )


def fixture():
    audit_rows = []
    jobs = []
    predictions = {group: [] for group in ("B0", "B1", "B2", "B3")}
    for family in range(4):
        for condition in range(6):
            audit_id = f"a-{family}-{condition}"
            image_sha = hashlib.sha256(audit_id.encode()).hexdigest()
            audit_rows.append(
                {
                    "audit_id": audit_id,
                    "family_id": f"f-{family}",
                    "image_sha256": image_sha,
                }
            )
            for group in predictions:
                job = {
                    "job_id": f"{group}:{audit_id}",
                    "audit_id": audit_id,
                    "family_id": f"f-{family}",
                    "group": group,
                    "image_sha256": image_sha,
                    "protocol_hash": protocol_hash(group),
                }
                jobs.append(job)
                predictions[group].append({**job, "prediction": valid_output()})
    return audit_rows, jobs, predictions


def test_smoke_gate_accepts_complete_same_protocol_run():
    audit_rows, jobs, predictions = fixture()

    result = validate_smoke(
        audit_rows,
        jobs,
        {"passed": True},
        predictions,
        {"completed": True},
        failure_files=[],
        peak_vram_bytes=30 * 1024**3,
        gpu_memory_bytes=48 * 1024**3,
    )

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["counts"] == {
        "families": 4,
        "audit_rows": 24,
        "jobs": 96,
        "predictions": 96,
        "schema_valid": 96,
        "schema_invalid": 0,
    }


def test_smoke_gate_fails_closed_for_each_critical_gate():
    audit_rows, jobs, predictions = fixture()
    predictions["B2"][0]["image_sha256"] = "wrong"
    predictions["B3"].pop()

    result = validate_smoke(
        audit_rows,
        jobs,
        {"passed": False},
        predictions,
        {"completed": False},
        failure_files=["failure.json"],
        peak_vram_bytes=48 * 1024**3,
        gpu_memory_bytes=48 * 1024**3,
    )

    assert result["passed"] is False
    assert result["checks"]["leakage_passed"] is False
    assert result["checks"]["prediction_counts"] is False
    assert result["checks"]["image_sha_equality"] is False
    assert result["checks"]["no_failure_reports"] is False
    assert result["checks"]["vram_within_capacity"] is False


def test_schema_failures_are_accounted_not_silently_dropped():
    audit_rows, jobs, predictions = fixture()
    predictions["B1"][0]["prediction"] = "not-json"

    result = validate_smoke(
        audit_rows,
        jobs,
        {"passed": True},
        predictions,
        {"completed": True},
        failure_files=[],
        peak_vram_bytes=1,
        gpu_memory_bytes=2,
    )

    assert result["passed"] is True
    assert result["checks"]["schema_accounting_complete"] is True
    assert result["counts"]["schema_valid"] == 95
    assert result["counts"]["schema_invalid"] == 1

