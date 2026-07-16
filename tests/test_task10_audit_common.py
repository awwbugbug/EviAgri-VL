import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10_audit_common import (
    ensure_new_directory,
    family_bootstrap_delta,
    sha256_file,
    write_json_new,
)


def test_sha256_file_matches_standard_library(tmp_path):
    target = tmp_path / "payload.bin"
    payload = b"eviagri-task10\x00audit"
    target.write_bytes(payload)

    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_write_json_new_refuses_existing_file(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_json_new(target, {"passed": True})

    assert target.read_text(encoding="utf-8") == "keep"


def test_write_json_new_creates_parent_and_canonical_utf8_json(tmp_path):
    target = tmp_path / "nested" / "report.json"

    write_json_new(target, {"z": 1, "研究": "农业"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "z": 1,
        "研究": "农业",
    }
    assert target.read_bytes().endswith(b"\n")


def test_ensure_new_directory_refuses_any_existing_path(tmp_path):
    target = tmp_path / "new-output"
    ensure_new_directory(target)
    assert target.is_dir()

    with pytest.raises(FileExistsError):
        ensure_new_directory(target)


def test_family_bootstrap_delta_is_deterministic_and_paired():
    rows = [("f1", 0.8, 0.2), ("f2", 0.6, 0.4), ("f3", 0.7, 0.3)]

    left = family_bootstrap_delta(rows, repetitions=1000, seed=20260717)
    right = family_bootstrap_delta(rows, repetitions=1000, seed=20260717)

    assert left == right
    assert left["estimate"] == pytest.approx(0.4)
    assert left["low"] > 0
    assert left["high"] >= left["low"]
    assert left["unit"] == "family_id"


@pytest.mark.parametrize(
    "rows",
    [[], [("f1", 0.8, 0.2), ("f1", 0.7, 0.3)]],
)
def test_family_bootstrap_rejects_empty_or_duplicate_families(rows):
    with pytest.raises(ValueError, match="one non-empty paired row per family"):
        family_bootstrap_delta(rows, repetitions=10, seed=1)


def test_family_bootstrap_rejects_invalid_repetition_count():
    with pytest.raises(ValueError, match="repetitions"):
        family_bootstrap_delta([("f1", 1.0, 0.0)], repetitions=0, seed=1)
