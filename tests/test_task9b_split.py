import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9b_split import assign_components, connected_components, locked_exclusion


def test_connected_components_are_transitive_and_deterministic():
    pairs = [
        {"left_id": "a", "right_id": "b", "high_confidence": True},
        {"left_id": "b", "right_id": "c", "high_confidence": True},
        {"left_id": "x", "right_id": "y", "high_confidence": False},
    ]

    first = connected_components({"a", "b", "c", "x", "y"}, pairs)
    second = connected_components({"a", "b", "c", "x", "y"}, list(reversed(pairs)))

    assert first == second
    assert first["a"] == first["b"] == first["c"]
    assert first["x"] != first["y"]


def test_locked_exclusion_is_blinded_and_contains_only_image_ids_and_sha():
    families = [
        {
            "source_id": "ip102det_test_IP000000572_positive",
            "source_image_sha256": "a" * 64,
            "pest_id": 7,
            "pest_name": "rice leaf roller",
            "gt_bbox": [1, 2, 3, 4],
        }
    ]

    exclusion = locked_exclusion(families, source_manifest_sha256="b" * 64)
    serialized = json.dumps(exclusion, sort_keys=True).lower()

    assert exclusion["image_ids"] == ["IP000000572"]
    assert exclusion["image_sha256"] == ["a" * 64]
    assert exclusion["source_manifest_sha256"] == "b" * 64
    assert "positive" not in serialized
    assert "pest" not in serialized
    assert "bbox" not in serialized


def test_assignment_keeps_components_together_and_excludes_locked_neighbors():
    records = [
        {"image_id": "locked", "class_id": 0, "image_sha256": "1" * 64},
        {"image_id": "neighbor", "class_id": 0, "image_sha256": "2" * 64},
        {"image_id": "free-a", "class_id": 0, "image_sha256": "3" * 64},
        {"image_id": "free-b", "class_id": 0, "image_sha256": "4" * 64},
    ]
    pairs = [
        {"left_id": "locked", "right_id": "neighbor", "high_confidence": True},
        {"left_id": "free-a", "right_id": "free-b", "high_confidence": True},
    ]
    exclusions = {"image_ids": ["locked"], "image_sha256": [], "source_manifest_sha256": "x"}

    result = assign_components(records, pairs, exclusions, seed=20260715)

    assert result["assignment"]["locked"] == "excluded"
    assert result["assignment"]["neighbor"] == "excluded"
    assert result["assignment"]["free-a"] == result["assignment"]["free-b"]
    assert result["assignment"]["free-a"] in {"train", "val", "dev"}


def test_assignment_is_deterministic_and_each_component_has_one_split():
    records = [
        {"image_id": f"img-{index:02d}", "class_id": index % 2, "image_sha256": f"{index:064x}"}
        for index in range(40)
    ]
    pairs = [
        {"left_id": "img-00", "right_id": "img-02", "high_confidence": True},
        {"left_id": "img-01", "right_id": "img-03", "high_confidence": True},
    ]
    exclusions = {"image_ids": [], "image_sha256": [], "source_manifest_sha256": "x"}

    first = assign_components(records, pairs, exclusions, seed=20260715)
    second = assign_components(list(reversed(records)), list(reversed(pairs)), exclusions, seed=20260715)

    assert first == second
    assert first["assignment"]["img-00"] == first["assignment"]["img-02"]
    assert first["assignment"]["img-01"] == first["assignment"]["img-03"]
    assert {"train", "val", "dev"}.issubset(set(first["assignment"].values()))
