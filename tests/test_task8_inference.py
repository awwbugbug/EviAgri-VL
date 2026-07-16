import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from run_task8_inference import (
    build_messages,
    generate_group_predictions,
    pending_jobs,
    validate_jobs,
)
from task8_protocol import protocol_hash


def jobs(tmp_path: Path) -> list[dict]:
    image = tmp_path / "images" / ("a" * 64 + ".png")
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    rows = []
    for group in ("B0", "B1", "B2", "B3"):
        rows.append(
            {
                "job_id": f"{group}:audit-1",
                "audit_id": "audit-1",
                "family_id": "family-1",
                "condition": "original_correct",
                "group": group,
                "image": str(image),
                "image_sha256": "a" * 64,
                "prompt": "neutral" if group in {"B1", "B2"} else group,
                "protocol_hash": protocol_hash(group),
            }
        )
    return rows


def test_validate_jobs_rejects_duplicate_and_bad_protocol(tmp_path):
    rows = jobs(tmp_path)
    validate_jobs(rows)
    duplicate = rows + [dict(rows[0])]
    try:
        validate_jobs(duplicate)
    except ValueError as error:
        assert "duplicate job_id" in str(error)
    else:
        raise AssertionError("expected duplicate refusal")
    rows[0]["protocol_hash"] = "wrong"
    try:
        validate_jobs(rows)
    except ValueError as error:
        assert "protocol hash" in str(error)
    else:
        raise AssertionError("expected protocol refusal")


def test_b1_b2_model_inputs_are_identical(tmp_path):
    rows = jobs(tmp_path)
    b1 = next(row for row in rows if row["group"] == "B1")
    b2 = next(row for row in rows if row["group"] == "B2")

    assert build_messages(b1) == build_messages(b2)


def test_pending_jobs_is_resume_safe_and_rejects_unknown_ids(tmp_path):
    rows = jobs(tmp_path)
    group_rows = [row for row in rows if row["group"] == "B1"]
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"job_id": group_rows[0]["job_id"]}) + "\n", encoding="utf-8")

    assert pending_jobs(group_rows, path) == []
    path.write_text(json.dumps({"job_id": "B1:unknown"}) + "\n", encoding="utf-8")
    try:
        pending_jobs(group_rows, path)
    except ValueError as error:
        assert "unknown completed job_id" in str(error)
    else:
        raise AssertionError("expected unknown-id refusal")


def test_generation_appends_once_and_preserves_registered_metadata(tmp_path):
    rows = [row for row in jobs(tmp_path) if row["group"] == "B2"]
    output = tmp_path / "predictions.jsonl"
    calls = []

    first = generate_group_predictions(
        rows, output, lambda row: calls.append(row["job_id"]) or '{"ok":true}'
    )
    second = generate_group_predictions(
        rows, output, lambda row: (_ for _ in ()).throw(AssertionError(row["job_id"]))
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert first == {"existing": 0, "generated": 1, "total": 1}
    assert second == {"existing": 1, "generated": 0, "total": 1}
    assert calls == ["B2:audit-1"]
    assert written["group"] == "B2"
    assert written["audit_id"] == "audit-1"
    assert written["prediction"] == '{"ok":true}'
