import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_reviewer_handoff import build_handoff


def test_handoff_excludes_reviewer_a_and_ai_results(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "images").mkdir()
    (source / "overlays").mkdir()
    (source / "review_manifest.jsonl").write_text(
        json.dumps({"audit_id": "abc"}) + "\n", encoding="utf-8"
    )
    (source / "images" / "abc.jpg").write_bytes(b"image")
    (source / "overlays" / "abc.jpg").write_bytes(b"overlay")
    (source / "review.html").write_text("blind", encoding="utf-8")
    (source / "reviewer_b.csv").write_text("audit_id\nabc\n", encoding="utf-8")
    (source / "reviewer_a_completed.confirmed.csv").write_text("secret", encoding="utf-8")
    (source / "ai_pretriage_report.json").write_text("secret", encoding="utf-8")

    output = tmp_path / "handoff"
    report = build_handoff(source_root=source, output_root=output, reviewer_id="reviewer_B")

    assert report["audit_rows"] == 1
    assert report["reviewer_a_results_included"] is False
    assert report["ai_pretriage_included"] is False
    assert not (output / "reviewer_a_completed.confirmed.csv").exists()
    assert not (output / "ai_pretriage_report.json").exists()
    assert (output / "reviewer_b.csv").is_file()
    assert (output / "completion.sha256").is_file()
