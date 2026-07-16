import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task9b_v2 import build_dataset


def _fixtures(tmp_path: Path):
    rows = []
    assignment = {}
    for index in range(12):
        image_id = f"img_{index:02d}"
        image_path = tmp_path / f"source-class-{index % 3}" / f"pest-{index}.png"
        image_path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (48, 40), (20 + index, 80, 130)).save(image_path)
        rows.append(
            {
                "image_id": image_id,
                "image_sha256": f"{index + 1:064x}",
                "image_path": str(image_path),
                "class_id": index % 3,
                "class_name": f"pest {index % 3}",
                "bbox": [8, 7, 31, 29],
                "present_class_ids": [index % 3],
            }
        )
        assignment[image_id] = "train" if index < 6 else "val" if index < 9 else "dev"
    return rows, assignment


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builds_exact_three_row_families_and_separates_private_labels(tmp_path):
    rows, assignment = _fixtures(tmp_path)
    result = build_dataset(rows, assignment, tmp_path / "out", seed=41)
    model_rows = sum((_read_jsonl(Path(path)) for path in result["model_files"].values()), [])
    private = _read_jsonl(Path(result["provenance_file"]))

    assert all(set(row) == {"id", "messages"} for row in model_rows)
    assert len(model_rows) == len(private) == 3 * len(rows)
    by_family = defaultdict(list)
    for row in private:
        by_family[row["family_id"]].append(row)
    assert all(Counter(item["role"] for item in family) == {
        "positive": 1, "semantic_negative": 1, "visual_counterfactual": 1
    } for family in by_family.values())
    assert Counter(row["null_source"] for row in private) == {
        None: len(rows), "real_null": len(rows), "synthetic_null": len(rows)
    }
    assert all(row["query_class_id"] not in row["present_class_ids"]
               for row in private if row["role"] == "semantic_negative")


def test_templates_lengths_and_schema_are_role_invariant(tmp_path):
    rows, assignment = _fixtures(tmp_path)
    result = build_dataset(rows, assignment, tmp_path / "out", seed=43)
    private = _read_jsonl(Path(result["provenance_file"]))
    by_role = defaultdict(list)
    for row in private:
        by_role[row["role"]].append(row)
    assert {role: Counter(r["template_id"] for r in values) for role, values in by_role.items()}
    assert len({tuple(sorted(Counter(r["template_id"] for r in values).items()))
                for values in by_role.values()}) == 1
    assert len({tuple(sorted(Counter(r["length_bucket"] for r in values).items()))
                for values in by_role.values()}) == 1

    for path in result["model_files"].values():
        for row in _read_jsonl(Path(path)):
            assert row["messages"][0]["role"] == "system"
            target_text = row["messages"][2]["content"][0]["text"]
            target = json.loads(target_text)
            assert tuple(target) == (
                "evidence_present", "evidence_region", "visible_attributes", "diagnosis", "reliability"
            )


def test_visible_paths_are_opaque_and_splits_have_no_source_overlap(tmp_path):
    rows, assignment = _fixtures(tmp_path)
    result = build_dataset(rows, assignment, tmp_path / "out", seed=47)
    sources_by_split = defaultdict(set)
    private = _read_jsonl(Path(result["provenance_file"]))
    for row in private:
        sources_by_split[row["split"]].add(row["source_image_id"])
    assert all(left.isdisjoint(right) for name, left in sources_by_split.items()
               for other, right in sources_by_split.items() if name != other)

    for path in result["model_files"].values():
        for row in _read_jsonl(Path(path)):
            image_ref = row["messages"][1]["content"][0]["image"]
            name = Path(image_ref).name
            assert len(Path(name).stem) == 64
            assert "class" not in image_ref and "pest-" not in image_ref


def test_builder_refuses_existing_output(tmp_path):
    rows, assignment = _fixtures(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    try:
        build_dataset(rows, assignment, output, seed=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("non-empty or existing freeze destination must be refused")
