import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_plantseg_blind_audit import direct_media_url, metadata_candidates


def _row(name="a.jpg", **updates):
    row = {
        "Name": name,
        "Plant": "Apple",
        "Disease": "disease",
        "Resolution": "640x480",
        "Label file": name.replace(".jpg", ".png"),
        "Mask ratio": "0.1",
        "URL": "https://example.org/image.jpg?size=large",
        "License": "CC-BY-NC",
        "Split": "Validation",
    }
    row.update(updates)
    return row


def test_direct_media_url_uses_path_not_article_or_query_text():
    assert direct_media_url("https://example.org/a.JPG?x=1")
    assert not direct_media_url("https://example.org/article?image=a.jpg")
    assert not direct_media_url("https://example.org/page")


def test_metadata_candidates_apply_all_frozen_filters():
    rows = [
        _row("ok.jpg"),
        _row("train.jpg", Split="Training"),
        _row("small.jpg", Resolution="200x500"),
        _row("ratio.jpg", **{"Mask ratio": "0.5"}),
        _row("article.jpg", URL="https://example.org/article"),
        _row("license.jpg", License="unknown"),
    ]
    assert [row["Name"] for row in metadata_candidates(rows)] == ["ok.jpg"]


def test_metadata_candidates_are_deterministic():
    rows = [_row("b.jpg"), _row("a.jpg")]
    assert [row["Name"] for row in metadata_candidates(rows)] == ["a.jpg", "b.jpg"]


def test_invalid_resolution_fails_closed():
    with pytest.raises(ValueError):
        metadata_candidates([_row(Resolution="bad")])
