import json
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageEnhance


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from audit_task8_leakage import audit_task8_leakage, dhash64, hamming_distance


def save(path: Path, color=(40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def clean_rows(tmp_path: Path):
    staging = tmp_path / "staging.png"
    save(staging)
    digest = hashlib.sha256(staging.read_bytes()).hexdigest()
    derived = tmp_path / "audit" / "images" / f"{digest}.png"
    derived.parent.mkdir(parents=True)
    staging.replace(derived)
    audit_rows = [
        {
            "audit_id": "audit-1",
            "family_id": "family-1",
            "image": str(derived),
            "image_sha256": digest,
            "actual_image_pest_name": "Miridae",
        }
    ]
    neutral = "Is Miridae visibly present? Return JSON with evidence_bbox=null."
    jobs = [
        {
            **audit_rows[0],
            "job_id": f"{group}:audit-1",
            "group": group,
            "prompt": neutral if group in {"B1", "B2"} else f"registered-{group}",
            "protocol_hash": "same" if group in {"B1", "B2"} else group,
        }
        for group in ("B0", "B1", "B2", "B3")
    ]
    train = tmp_path / "train" / "JPEGImages" / "train-a.jpg"
    val = tmp_path / "val" / "JPEGImages" / "val-a.jpg"
    test = tmp_path / "test" / "JPEGImages" / "test-a.jpg"
    save(train, (20, 30, 40))
    save(val, (60, 70, 80))
    save(test, (100, 110, 120))
    split_rows = {
        "train": [{"id": "train-a", "image": str(train), "messages": []}],
        "val": [{"id": "val-a", "image": str(val), "messages": []}],
        "test": [{"id": "test-a", "image": str(test), "messages": []}],
    }
    return split_rows, audit_rows, jobs


def test_clean_registered_protocol_passes(tmp_path):
    split_rows, audit_rows, jobs = clean_rows(tmp_path)

    report = audit_task8_leakage(split_rows, audit_rows, jobs)

    assert report["passed"] is True
    assert report["hard_failures"] == []


def test_b1_b2_mismatch_and_duplicate_audit_ids_are_hard_failures(tmp_path):
    split_rows, audit_rows, jobs = clean_rows(tmp_path)
    audit_rows.append(dict(audit_rows[0]))
    next(job for job in jobs if job["group"] == "B2")["prompt"] += " changed"

    report = audit_task8_leakage(split_rows, audit_rows, jobs)

    joined = " ".join(report["hard_failures"])
    assert report["passed"] is False
    assert "duplicate audit_id" in joined
    assert "B1/B2" in joined


def test_class_bearing_path_and_forbidden_prompt_state_fail(tmp_path):
    split_rows, audit_rows, jobs = clean_rows(tmp_path)
    leaked = tmp_path / "audit" / "Miridae" / "Miridae-positive.jpg"
    save(leaked)
    audit_rows[0]["image"] = str(leaked)
    for job in jobs:
        job["image"] = str(leaked)
    next(job for job in jobs if job["group"] == "B1")["prompt"] = "this is a positive sample /root/x.jpg"
    next(job for job in jobs if job["group"] == "B2")["prompt"] = "this is a positive sample /root/x.jpg"

    report = audit_task8_leakage(split_rows, audit_rows, jobs)

    joined = " ".join(report["hard_failures"])
    assert "opaque" in joined
    assert "forbidden prompt token" in joined


def test_cross_split_exact_duplicate_and_source_id_crossing_fail(tmp_path):
    split_rows, audit_rows, jobs = clean_rows(tmp_path)
    split_rows["val"][0]["image"] = split_rows["train"][0]["image"]
    split_rows["val"][0]["id"] = split_rows["train"][0]["id"]

    report = audit_task8_leakage(split_rows, audit_rows, jobs)

    joined = " ".join(report["hard_failures"])
    assert "exact image duplicate across splits" in joined
    assert "source image id across splits" in joined


def test_dhash_and_hamming_detect_small_visual_change(tmp_path):
    base = tmp_path / "base.png"
    changed = tmp_path / "changed.png"
    image = Image.new("L", (32, 24))
    for x in range(32):
        for y in range(24):
            image.putpixel((x, y), x * 7)
    image.convert("RGB").save(base)
    ImageEnhance.Brightness(image).enhance(1.02).convert("RGB").save(changed)

    distance = hamming_distance(dhash64(base), dhash64(changed))

    assert 0 <= distance <= 4


def test_perceptual_candidates_count_each_physical_image_once(tmp_path):
    split_rows, audit_rows, jobs = clean_rows(tmp_path)
    train_path = Path(split_rows["train"][0]["image"])
    val_path = Path(split_rows["val"][0]["image"])
    gradient = Image.new("L", (32, 24))
    for x in range(32):
        for y in range(24):
            gradient.putpixel((x, y), x * 7)
    gradient.convert("RGB").save(train_path)
    ImageEnhance.Brightness(gradient).enhance(1.02).convert("RGB").save(val_path)
    split_rows["train"].append(
        {"id": "train-a-null", "image": str(train_path), "messages": []}
    )

    report = audit_task8_leakage(split_rows, audit_rows, jobs)

    matching = [
        row
        for row in report["near_duplicate_candidates"]
        if {row["left_split"], row["right_split"]} == {"train", "val"}
    ]
    assert report["perceptual_unique_images"] == 3
    assert len(matching) == 1
