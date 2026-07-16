import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10b_protocol import build_protocol, run_protocol_from_paths, write_protocol


QUOTAS = {"train": 12, "val": 3, "dev": 5}
BAND_QUOTAS = {"head": 6, "medium": 5, "tail": 5}


def _positive(image_id: str, class_id: int, *, source_split: str = "trainval") -> dict:
    return {
        "image": f"/images/{image_id}.jpg",
        "source_split": source_split,
        "metadata": {"image_id": image_id},
        "target": {"diagnosis": {"pest_id": class_id}},
    }


def _source_rows(class_specs: list[tuple[int, str, int]]):
    positives = []
    provenance = []
    for class_id, _band, count in class_specs:
        for index in range(count):
            image_id = f"c{class_id:03d}-i{index:03d}"
            positives.append(_positive(image_id, class_id))
            provenance.append(
                {
                    "source_image_id": image_id,
                    "source_image_sha256": f"{class_id:02x}{index:062x}"[-64:],
                    "near_duplicate_component_id": f"component-{class_id}-{index}",
                }
            )
    return positives, provenance


def _selected_classes(class_specs):
    return [{"class_id": class_id, "band": band} for class_id, band, _count in class_specs]


def test_build_protocol_is_deterministic_exact_and_leakage_free():
    specs = (
        [(class_id, "head", 30 + class_id) for class_id in range(1, 8)]
        + [(class_id, "medium", 28 + class_id) for class_id in range(11, 17)]
        + [(class_id, "tail", 22 + class_id) for class_id in range(21, 27)]
    )
    positives, provenance = _source_rows(specs)

    # These records must never become candidates.
    positives.append(_positive("official-test", 1, source_split="test"))
    provenance.append(
        {
            "source_image_id": "official-test",
            "source_image_sha256": "f" * 64,
            "near_duplicate_component_id": "official-test-component",
        }
    )
    locked_id = "c001-i000"
    locked_sha = next(
        row["source_image_sha256"] for row in provenance if row["source_image_id"] == "c002-i000"
    )
    used_sha = next(
        row["source_image_sha256"] for row in provenance if row["source_image_id"] == "c003-i000"
    )

    kwargs = dict(
        positive_rows=positives,
        provenance_rows=provenance,
        used_sha256={used_sha},
        locked_ids={locked_id},
        locked_sha256={locked_sha},
        selected_classes=_selected_classes(specs),
        class_bands={str(class_id): band for class_id, band, _count in specs},
        split_quotas=QUOTAS,
        band_quotas=BAND_QUOTAS,
    )
    first = build_protocol(**kwargs)
    second = build_protocol(**kwargs)

    assert first == second
    assert first["status"] == "PASSED_PROTOCOL"
    assert first["report"]["model_loaded"] is False
    assert first["report"]["rows_by_split"] == {"dev": 80, "train": 192, "val": 48}
    assert first["report"]["selected_class_count"] == 16
    assert Counter(row["class_band"] for row in first["selected_classes"]) == BAND_QUOTAS

    manifest = first["manifest"]
    assert len(manifest) == 320
    assert len({row["source_image_sha256"] for row in manifest}) == 320
    assert len({row["near_duplicate_component_id"] for row in manifest}) == 320
    assert not ({locked_id, "official-test"} & {row["source_image_id"] for row in manifest})
    assert locked_sha not in {row["source_image_sha256"] for row in manifest}
    assert used_sha not in {row["source_image_sha256"] for row in manifest}
    assert first["report"]["overlap"] == {
        "near_duplicate_component": 0,
        "source_image_sha256": 0,
    }


def test_build_protocol_excludes_multiclass_components():
    specs = (
        [(class_id, "head", 20) for class_id in range(1, 7)]
        + [(class_id, "medium", 20) for class_id in range(11, 16)]
        + [(class_id, "tail", 20) for class_id in range(21, 26)]
    )
    positives, provenance = _source_rows(specs)
    by_id = {row["source_image_id"]: row for row in provenance}
    by_id["c001-i000"]["near_duplicate_component_id"] = "cross-class-component"
    by_id["c011-i000"]["near_duplicate_component_id"] = "cross-class-component"

    result = build_protocol(
        positive_rows=positives,
        provenance_rows=provenance,
        used_sha256=set(),
        locked_ids=set(),
        locked_sha256=set(),
        selected_classes=_selected_classes(specs),
        class_bands={str(class_id): band for class_id, band, _count in specs},
        split_quotas=QUOTAS,
        band_quotas=BAND_QUOTAS,
    )

    assert result["status"] == "BLOCKED_CLASS_QUOTA"
    assert result["report"]["excluded_multiclass_components"] == 1
    assert result["report"]["eligible_by_band"]["head"] == 5
    assert result["report"]["eligible_by_band"]["medium"] == 4


def test_build_protocol_blocks_infeasible_tail_quota_without_degrading():
    specs = (
        [(class_id, "head", 25) for class_id in range(1, 7)]
        + [(class_id, "medium", 25) for class_id in range(11, 16)]
        + [(21, "tail", 25)]
        + [(class_id, "tail", 19) for class_id in range(22, 26)]
    )
    positives, provenance = _source_rows(specs)

    result = build_protocol(
        positive_rows=positives,
        provenance_rows=provenance,
        used_sha256=set(),
        locked_ids=set(),
        locked_sha256=set(),
        selected_classes=_selected_classes(specs),
        class_bands={str(class_id): band for class_id, band, _count in specs},
        split_quotas=QUOTAS,
        band_quotas=BAND_QUOTAS,
    )

    assert result["status"] == "BLOCKED_CLASS_QUOTA"
    assert result["manifest"] == []
    assert result["report"]["eligible_by_band"] == {"head": 6, "medium": 5, "tail": 1}
    assert result["report"]["required_by_band"] == BAND_QUOTAS
    assert result["report"]["model_loaded"] is False


def test_write_protocol_signs_passed_outputs_and_refuses_overwrite(tmp_path):
    specs = (
        [(class_id, "head", 20) for class_id in range(1, 7)]
        + [(class_id, "medium", 20) for class_id in range(11, 16)]
        + [(class_id, "tail", 20) for class_id in range(21, 26)]
    )
    positives, provenance = _source_rows(specs)
    result = build_protocol(
        positive_rows=positives,
        provenance_rows=provenance,
        used_sha256=set(),
        locked_ids=set(),
        locked_sha256=set(),
        selected_classes=_selected_classes(specs),
        class_bands={str(class_id): band for class_id, band, _count in specs},
        split_quotas=QUOTAS,
        band_quotas=BAND_QUOTAS,
    )
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    output = tmp_path / "protocol"

    write_protocol(result, output, [input_file])

    assert json.loads((output / "status.json").read_text())["state"] == "completed"
    assert len((output / "manifest.jsonl").read_text().splitlines()) == 320
    assert (output / "completion.sha256").is_file()
    with pytest.raises(FileExistsError):
        write_protocol(result, output, [input_file])


def test_write_protocol_records_block_without_completion(tmp_path):
    result = {
        "status": "BLOCKED_CLASS_QUOTA",
        "manifest": [],
        "selected_classes": [],
        "report": {"status": "BLOCKED_CLASS_QUOTA", "model_loaded": False},
        "config": {"split_quotas": QUOTAS, "band_quotas": BAND_QUOTAS},
    }
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    output = tmp_path / "blocked"

    write_protocol(result, output, [input_file])

    assert json.loads((output / "status.json").read_text())["state"] == "blocked"
    assert (output / "block_report.json").is_file()
    assert not (output / "manifest.jsonl").exists()
    assert not (output / "completion.sha256").exists()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_run_protocol_from_paths_checks_frozen_class_identity(tmp_path):
    specs = (
        [(class_id, "head", 20) for class_id in range(1, 7)]
        + [(class_id, "medium", 20) for class_id in range(11, 16)]
        + [(class_id, "tail", 20) for class_id in range(21, 26)]
    )
    positives, provenance = _source_rows(specs)
    positive_path = tmp_path / "train.jsonl"
    provenance_path = tmp_path / "provenance.jsonl"
    used_path = tmp_path / "used.jsonl"
    locked_path = tmp_path / "locked.json"
    selected_path = tmp_path / "selected.json"
    bands_path = tmp_path / "bands.json"
    _write_jsonl(positive_path, positives)
    _write_jsonl(provenance_path, provenance)
    _write_jsonl(used_path, [])
    locked_path.write_text(
        json.dumps({"image_ids": [], "image_sha256": []}), encoding="utf-8"
    )
    selected_path.write_text(json.dumps(_selected_classes(specs)), encoding="utf-8")
    bands_path.write_text(
        json.dumps({str(class_id): band for class_id, band, _count in specs}),
        encoding="utf-8",
    )
    expected = [class_id for class_id, _band, _count in specs]

    report = run_protocol_from_paths(
        positive_paths=[positive_path],
        provenance_path=provenance_path,
        used_provenance_path=used_path,
        locked_exclusion_path=locked_path,
        selected_classes_path=selected_path,
        class_bands_path=bands_path,
        output_root=tmp_path / "formal",
        expected_selected_class_ids=expected,
    )

    assert report["status"] == "PASSED_PROTOCOL"
    assert (tmp_path / "formal" / "completion.sha256").is_file()

    with pytest.raises(ValueError, match="frozen selected class mismatch"):
        run_protocol_from_paths(
            positive_paths=[positive_path],
            provenance_path=provenance_path,
            used_provenance_path=used_path,
            locked_exclusion_path=locked_path,
            selected_classes_path=selected_path,
            class_bands_path=bands_path,
            output_root=tmp_path / "must-not-exist",
            expected_selected_class_ids=expected[:-1] + [999],
        )
    assert not (tmp_path / "must-not-exist").exists()
