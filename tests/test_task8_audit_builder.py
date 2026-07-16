import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task8_audit import CONDITIONS, build_audit_dataset, select_families


def make_record(tmp_path: Path, pest_id: int, pest_name: str, index: int) -> dict:
    image = tmp_path / "source" / pest_name / f"{pest_name}-{index}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80 + index, 60 + index), color=(30 * pest_id, 40, 90)).save(image)
    return {
        "id": f"record-{pest_id}-{index}",
        "image": str(image),
        "split": "test",
        "task_type": "pest_evidence_grounding",
        "target": {
            "evidence_present": True,
            "evidence_bbox": [5, 6, 30, 35],
            "visible_attributes": [],
            "diagnosis": {"pest_id": pest_id, "pest_name": pest_name},
            "reliability": "supported",
        },
    }


def fixture_records(tmp_path: Path) -> list[dict]:
    return [
        make_record(tmp_path, pest_id, name, index)
        for pest_id, name in ((0, "alpha"), (1, "beta"), (2, "gamma"))
        for index in (0, 1)
    ]


def test_stratified_family_selection_is_deterministic(tmp_path):
    records = fixture_records(tmp_path)

    first = select_families(records, per_class=1, seed=20260715)
    second = select_families(list(reversed(records)), per_class=1, seed=20260715)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {row["target"]["diagnosis"]["pest_id"] for row in first} == {0, 1, 2}


def test_smoke_can_limit_the_number_of_stratified_classes(tmp_path):
    selected = select_families(
        fixture_records(tmp_path), per_class=1, seed=20260715, max_classes=2
    )

    assert len(selected) == 2
    assert len({row["target"]["diagnosis"]["pest_id"] for row in selected}) == 2


def test_selection_excludes_cross_split_near_duplicate_source_stems(tmp_path):
    records = fixture_records(tmp_path)
    excluded = {Path(records[0]["image"]).stem, Path(records[2]["image"]).stem}

    selected = select_families(
        records,
        per_class=1,
        seed=20260715,
        exclude_image_stems=excluded,
    )

    assert not {Path(row["image"]).stem for row in selected} & excluded
    assert len(selected) == 3


def test_builder_creates_six_paired_conditions_with_opaque_pixels(tmp_path):
    output = tmp_path / "audit"
    summary = build_audit_dataset(
        fixture_records(tmp_path), output, per_class=1, seed=20260715
    )
    rows = [
        json.loads(line)
        for line in (output / "audit_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["families"] == 3
    assert summary["audit_rows"] == 3 * 6
    assert {row["condition"] for row in rows} == set(CONDITIONS)
    for family_id in {row["family_id"] for row in rows}:
        family_rows = [row for row in rows if row["family_id"] == family_id]
        assert len(family_rows) == 6
        original = next(row for row in family_rows if row["condition"] == "original_correct")
        wrong = next(row for row in family_rows if row["condition"] == "original_wrong_query")
        no_target = next(row for row in family_rows if row["condition"] == "no_target_image")
        assert wrong["query_pest_id"] != wrong["actual_image_pest_id"]
        assert no_target["query_pest_id"] != no_target["actual_image_pest_id"]
        assert original["expected_evidence_present"] is True
        assert all(
            row["expected_evidence_present"] is False
            for row in family_rows
            if row["condition"] != "original_correct"
        )
        original_size = Image.open(original["image"]).size
        assert all(Image.open(row["image"]).size == original_size for row in family_rows)
        for row in family_rows:
            path = Path(row["image"])
            assert path.parent.name == "images"
            assert len(path.stem) == 64
            assert not any(name in path.name.lower() for name in ("alpha", "beta", "gamma"))


def test_same_audit_row_expands_to_identical_pixels_for_all_groups(tmp_path):
    output = tmp_path / "audit"
    build_audit_dataset(fixture_records(tmp_path), output, per_class=1, seed=20260715)
    jobs = [
        json.loads(line)
        for line in (output / "inference_jobs.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    audit_ids = {job["audit_id"] for job in jobs}
    assert len(jobs) == len(audit_ids) * 4
    for audit_id in audit_ids:
        group_jobs = [job for job in jobs if job["audit_id"] == audit_id]
        assert {job["group"] for job in group_jobs} == {"B0", "B1", "B2", "B3"}
        assert len({job["image_sha256"] for job in group_jobs}) == 1
        assert len({job["image"] for job in group_jobs}) == 1


def test_builder_refuses_to_overwrite_nonempty_output(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")

    try:
        build_audit_dataset(fixture_records(tmp_path), output, per_class=1, seed=20260715)
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("expected non-empty output refusal")
