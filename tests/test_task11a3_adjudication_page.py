import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_adjudication_page import build_adjudication_page


def test_adjudication_page_is_decision_first_and_exportable(tmp_path):
    manifest = tmp_path / "disagreements.jsonl"
    manifest.write_text(json.dumps({
        "audit_id": "abc",
        "reviewer_a_decision": "REJECT",
        "reviewer_a_failures": ["no_collage"],
        "reviewer_b_decision": "PASS",
        "reviewer_b_failures": [],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "adjudication.html"

    result = build_adjudication_page(disagreement_path=manifest, output_html=output)

    page = output.read_text(encoding="utf-8")
    assert result["disagreement_rows"] == 1
    assert "images/abc.jpg" in page and "overlays/abc.jpg" in page
    assert "<details>" in page
    assert "task11a3_adjudication.csv" in page
    assert "const ids=[\"abc\"]" in page
    assert "s.decision==='EXCLUDE'" in page
