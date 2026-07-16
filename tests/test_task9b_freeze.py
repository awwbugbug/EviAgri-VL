import json
import sys
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task9b_v2 import build_dataset
from validate_task9b_freeze import validate_freeze


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _built(tmp_path: Path, *, shared_component=False):
    records, assignment, components = [], {}, {}
    for index in range(12):
        image_id = f"img{index}"
        path = tmp_path / "source" / f"class_{index % 3}" / f"named_{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 28), (index + 20, 90, 140)).save(path)
        records.append({
            "image_id": image_id, "image_sha256": f"{index + 1:064x}",
            "image_path": str(path), "class_id": index % 3,
            "class_name": f"pest {index % 3}", "bbox": [5, 5, 22, 21],
            "present_class_ids": [index % 3],
        })
        assignment[image_id] = "train" if index < 6 else "val" if index < 9 else "dev"
        components[image_id] = "shared" if shared_component and index in (0, 10) else f"c{index}"
    root = tmp_path / "dataset"
    build_dataset(records, assignment, root, seed=13, component_by_image_id=components)
    return root


def test_freeze_report_has_four_level_json_quality_and_three_probe_views(tmp_path):
    root = _built(tmp_path)
    report = validate_freeze(root, locked_exclusion={"image_ids": [], "image_sha256": []})
    assert report["passed"] is True
    assert report["json_quality"] == {
        "syntax_validity": 1.0,
        "schema_validity": 1.0,
        "semantic_consistency": 1.0,
        "task_compliance": 1.0,
    }
    assert set(report["probe_files"]) == {
        "user_prompt_only", "system_user_prompt", "prompt_nonimage_metadata"
    }
    assert all(Path(path).exists() for path in report["probe_files"].values())
    assert (root / "completion.sha256").exists()


def test_freeze_rejects_model_visible_label_field_and_prompt_leak(tmp_path):
    root = _built(tmp_path)
    path = root / "model" / "train.jsonl"
    rows = _read(path)
    rows[0]["task_type"] = "positive"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="model-visible key allowlist"):
        validate_freeze(root, locked_exclusion={"image_ids": [], "image_sha256": []})


def test_freeze_rejects_cluster_overlap_and_locked_source(tmp_path):
    root = _built(tmp_path, shared_component=True)
    with pytest.raises(ValueError, match="component overlap"):
        validate_freeze(root, locked_exclusion={"image_ids": [], "image_sha256": []})

    root = _built(tmp_path / "second")
    with pytest.raises(ValueError, match="locked source overlap"):
        validate_freeze(root, locked_exclusion={"image_ids": ["img0"], "image_sha256": []})


def test_freeze_rejects_dev_transform_collision_and_existing_report(tmp_path):
    root = _built(tmp_path)
    private = root / "private" / "provenance.jsonl"
    rows = _read(private)
    dev_cf = next(row for row in rows if row["split"] == "dev" and row["role"] == "visual_counterfactual")
    dev_cf["transform_id"] = "train_blur"
    private.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="transform registry collision"):
        validate_freeze(root, locked_exclusion={"image_ids": [], "image_sha256": []})

    clean = _built(tmp_path / "clean")
    validate_freeze(clean, locked_exclusion={"image_ids": [], "image_sha256": []})
    with pytest.raises(FileExistsError, match="freeze outputs already exist"):
        validate_freeze(clean, locked_exclusion={"image_ids": [], "image_sha256": []})
