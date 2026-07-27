import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_reviewer_draft import compile_review


def _write(path, value):
    path.write_text(value, encoding="utf-8")


def test_reviewer_draft_expands_unlisted_rows_as_pass(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write(manifest, '{"audit_id":"a"}\n{"audit_id":"b"}\n')
    declaration = tmp_path / "declaration.json"
    _write(declaration, json.dumps({
        "status": "DRAFT_PENDING_CONFIRMATION",
        "reviewer_id": "reviewer_A",
        "failures": {"mask_valid": ["b"]},
        "uncertain": {},
    }))
    summary = compile_review(
        manifest_path=manifest, declaration_path=declaration, output_csv=tmp_path / "review.csv"
    )
    assert summary["decision_counts"] == {"PASS": 1, "REJECT": 1, "UNCERTAIN": 0}
    assert summary["unlisted_rows_assumed_all_pass"] == 1


def test_reviewer_draft_rejects_unknown_ids_and_non_draft_status(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write(manifest, '{"audit_id":"a"}\n')
    declaration = tmp_path / "declaration.json"
    output = tmp_path / "review.csv"
    _write(declaration, json.dumps({
        "status": "DRAFT_PENDING_CONFIRMATION", "reviewer_id": "r",
        "failures": {"no_collage": ["missing"]}, "uncertain": {},
    }))
    with pytest.raises(ValueError, match="unknown audit_id"):
        compile_review(manifest_path=manifest, declaration_path=declaration, output_csv=output)
    _write(declaration, json.dumps({
        "status": "CONFIRMED", "reviewer_id": "r", "failures": {}, "uncertain": {},
    }))
    with pytest.raises(ValueError, match="remain a draft"):
        compile_review(manifest_path=manifest, declaration_path=declaration, output_csv=output)
