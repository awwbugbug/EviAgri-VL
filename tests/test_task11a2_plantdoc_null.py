import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from prepare_task11a2_plantdoc_null import deterministic_selection, git_blob_sha1


def test_git_blob_sha_matches_known_git_object():
    assert git_blob_sha1(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"


def test_selection_is_deterministic_and_ignores_non_images():
    entries = [
        {"type": "file", "name": f"{index}.jpg", "download_url": f"u{index}", "sha": str(index)}
        for index in range(8)
    ] + [{"type": "file", "name": "README.md", "download_url": "x", "sha": "x"}]
    first = deterministic_selection(entries, commit="a" * 40, class_name="Apple leaf", count=4)
    second = deterministic_selection(entries, commit="a" * 40, class_name="Apple leaf", count=4)
    assert first == second
    assert len(first) == 4
    assert all(row["name"].endswith(".jpg") for row in first)
