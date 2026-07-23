import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task11a1_background_null import (
    candidate_crops,
    choose_candidate,
    expand_boxes,
    intersects,
    select_smoke,
)


def test_background_candidates_never_intersect_expanded_boxes():
    boxes = expand_boxes([(40, 40, 60, 60)], width=100, height=100, margin_fraction=0.1)
    candidates = candidate_crops(
        width=100, height=100, boxes=boxes, crop_size=20, grid=5
    )
    assert candidates
    assert all(not intersects(crop, boxes[0]) for crop in candidates)


def test_candidate_selection_is_deterministic():
    candidates = [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)]
    assert choose_candidate(candidates, split="dev", source_id="x") == choose_candidate(
        candidates, split="dev", source_id="x"
    )


def test_smoke_selection_is_balanced_and_does_not_cross_split():
    rows = []
    for split in ("val", "dev"):
        for band in ("head", "medium", "tail"):
            rows.extend(
                {"id": f"{split}-{band}-{index}", "split": split, "class_band": band}
                for index in range(6)
            )
    selected = select_smoke(rows, 12)
    assert len(selected) == 24
    for split in ("val", "dev"):
        for band in ("head", "medium", "tail"):
            assert sum(
                row["split"] == split and row["class_band"] == band
                for row in selected
            ) == 4
