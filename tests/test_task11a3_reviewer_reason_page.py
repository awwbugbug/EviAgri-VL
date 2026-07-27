import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_reviewer_reason_page import build_reason_page


def test_reason_page_requires_confirmed_independence_and_embeds_full_manifest(tmp_path):
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps({
        "status": "IDENTITY_CONFIRMED_PENDING_CRITERIA",
        "reject_ids": ["b"],
    }), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"audit_id":"a"}\n{"audit_id":"b"}\n', encoding="utf-8")
    output = tmp_path / "reason.html"

    result = build_reason_page(pending_path=pending, manifest_path=manifest, output_html=output)

    page = output.read_text(encoding="utf-8")
    assert result == {"audit_rows": 2, "reject_rows": 1, "output": str(output)}
    assert "images/b.jpg" in page
    assert "overlays/b.jpg" in page
    assert "reviewer_b_completed.csv" in page
    assert 'allIds=["a", "b"]' in page
    assert "Reviewer A" not in page
    assert '/[",\\n]/' in page
    assert "join('\\r\\n')" in page
    assert "function loadSaved(){try{" in page
    assert "function persistSaved(){try{" in page
    assert "countEl.textContent" in page
    assert "barEl.value" in page
    assert "exportButton.onclick" in page
    assert "export.onclick" not in page


def test_reason_page_rejects_unconfirmed_or_unknown_ids(tmp_path):
    pending = tmp_path / "pending.json"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"audit_id":"a"}\n', encoding="utf-8")
    pending.write_text(json.dumps({"status": "PENDING", "reject_ids": ["a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="independence"):
        build_reason_page(pending_path=pending, manifest_path=manifest, output_html=tmp_path / "x.html")
    pending.write_text(json.dumps({
        "status": "IDENTITY_CONFIRMED_PENDING_CRITERIA", "reject_ids": ["missing"]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown reject IDs"):
        build_reason_page(pending_path=pending, manifest_path=manifest, output_html=tmp_path / "y.html")
