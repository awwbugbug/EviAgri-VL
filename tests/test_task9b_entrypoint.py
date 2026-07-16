import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from run_task9b_freeze import source_record


def test_source_record_uses_pixels_and_all_valid_object_ids(tmp_path):
    image = tmp_path / "class-bearing" / "pest-name.jpg"
    image.parent.mkdir()
    image.write_bytes(b"pixel-bytes")
    row = {
        "image": str(image),
        "target": {
            "evidence_bbox": [1, 2, 11, 12],
            "diagnosis": {"pest_id": 7, "pest_name": "target pest"},
        },
        "metadata": {
            "image_id": "IP0001",
            "all_valid_objects": [
                {"pest_id": 7, "pest_name": "target pest"},
                {"pest_id": 9, "pest_name": "other visible pest"},
            ],
        },
    }
    converted = source_record(row)
    assert converted["image_id"] == "IP0001"
    assert converted["image_sha256"] == hashlib.sha256(b"pixel-bytes").hexdigest()
    assert converted["present_class_ids"] == [7, 9]
    assert converted["class_id"] == 7
    assert converted["bbox"] == [1, 2, 11, 12]
