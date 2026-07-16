import sys
from pathlib import Path

from PIL import Image, ImageEnhance


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from review_task8_near_duplicates import phash64, review_candidates


def gradient(path: Path, horizontal: bool = True) -> None:
    image = Image.new("L", (64, 48))
    for x in range(64):
        for y in range(48):
            image.putpixel((x, y), (x if horizontal else y) * 4)
    image.convert("RGB").save(path)


def test_phash_is_stable_to_small_brightness_change(tmp_path):
    base = tmp_path / "base.png"
    changed = tmp_path / "changed.png"
    gradient(base)
    with Image.open(base) as image:
        ImageEnhance.Brightness(image).enhance(1.02).save(changed)

    assert (phash64(base) ^ phash64(changed)).bit_count() <= 8


def test_review_promotes_only_structurally_similar_candidates(tmp_path):
    train = tmp_path / "train.png"
    similar = tmp_path / "similar.png"
    different = tmp_path / "different.png"
    gradient(train)
    with Image.open(train) as image:
        ImageEnhance.Brightness(image).enhance(1.02).save(similar)
    gradient(different, horizontal=False)
    split_rows = {
        "train": [{"image": str(train)}],
        "val": [{"image": str(similar)}],
        "test": [{"image": str(different)}],
    }
    candidates = [
        {"left_split": "train", "left_id": "train", "right_split": "val", "right_id": "similar"},
        {"left_split": "train", "left_id": "train", "right_split": "test", "right_id": "different"},
    ]

    result = review_candidates(split_rows, candidates)

    assert result["candidate_count"] == 2
    assert result["high_confidence_count"] == 1
    assert result["high_confidence_candidates"][0]["right_id"] == "similar"
    assert result["requires_manual_resolution"] is True
    assert result["contaminated_image_ids_by_split"] == {
        "test": [],
        "train": ["train"],
        "val": ["similar"],
    }
