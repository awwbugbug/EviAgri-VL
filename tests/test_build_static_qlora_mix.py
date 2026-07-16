import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_static_qlora_mix import build_mix


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_record(tmp_path: Path, record_id: str, split: str, kind: str) -> dict:
    image = tmp_path / "images" / f"{record_id}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fixture")
    return {
        "id": record_id,
        "image": str(image),
        "split": split,
        "task_type": "pest_evidence_grounding" if kind == "positive" else "prompt_conflict_null_evidence",
    }


def make_fixture_source(tmp_path: Path, train_positive: int = 4, train_null: int = 4) -> Path:
    source = tmp_path / "source"
    rows: dict[tuple[str, str], list[dict]] = {}
    for split in ("train", "val", "test"):
        positive_count = train_positive if split == "train" else 2
        null_count = train_null if split == "train" else 2
        rows[(split, "positive")] = [
            make_record(tmp_path, f"{split}-positive-{index}", split, "positive")
            for index in range(positive_count)
        ]
        rows[(split, "null")] = [
            make_record(tmp_path, f"{split}-null-{index}", split, "null")
            for index in range(null_count)
        ]
        write_jsonl(source / "vlm_sft" / f"{split}_evidence_positive.jsonl", rows[(split, "positive")])
        write_jsonl(source / "hallucination" / f"{split}_prompt_conflict.jsonl", rows[(split, "null")])
    return source


def fixture_config(source: Path, positive: int = 4, null: int = 2) -> dict:
    return {
        "seed": 20260714,
        "source_data_root": str(source),
        "data": {"train_positive": positive, "train_null": null},
    }


def test_build_mix_selects_stable_two_to_one_train_ratio(tmp_path):
    source = make_fixture_source(tmp_path)
    output = tmp_path / "out"

    summary = build_mix(fixture_config(source), output)

    assert summary["counts"]["train"] == {"positive": 4, "null": 2, "total": 6}
    assert summary["counts"]["val"] == {"positive": 2, "null": 2, "total": 4}
    assert summary["counts"]["test"] == {"positive": 2, "null": 2, "total": 4}
    first = (output / "train.jsonl").read_bytes()
    shutil.rmtree(output)
    build_mix(fixture_config(source), output)
    assert (output / "train.jsonl").read_bytes() == first
    assert (output / "manifest.json").is_file()
    assert (output / "sha256sum.txt").is_file()


def test_build_mix_rejects_duplicate_ids_before_writing(tmp_path):
    source = make_fixture_source(tmp_path)
    positive_path = source / "vlm_sft" / "train_evidence_positive.jsonl"
    rows = [json.loads(line) for line in positive_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["id"] = rows[0]["id"]
    write_jsonl(positive_path, rows)
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="duplicate record id"):
        build_mix(fixture_config(source), output)

    assert not output.exists()


def test_build_mix_rejects_missing_images_before_writing(tmp_path):
    source = make_fixture_source(tmp_path)
    positive_path = source / "vlm_sft" / "train_evidence_positive.jsonl"
    rows = [json.loads(line) for line in positive_path.read_text(encoding="utf-8").splitlines()]
    Path(rows[0]["image"]).unlink()
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="missing image"):
        build_mix(fixture_config(source), output)

    assert not output.exists()


def test_build_mix_refuses_to_overwrite_nonempty_output(tmp_path):
    source = make_fixture_source(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_mix(fixture_config(source), output)

    assert sentinel.read_text(encoding="utf-8") == "user data"
