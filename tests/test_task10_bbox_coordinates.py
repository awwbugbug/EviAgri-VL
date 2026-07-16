import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from audit_task10_bbox_coordinates import (
    BLOCKED_COORDINATE_PROTOCOL,
    ORIGINAL_PIXEL_FRAME,
    audit_coordinate_record,
    audit_coordinate_records,
    box_iou,
    collect_coordinate_records,
    interpret_box,
    scale_box,
    validate_box,
)


def _record(family_id="f01"):
    return {
        "family_id": family_id,
        "original_size": [1000, 600],
        "vision_size": [1000, 600],
        "processor_size": [616, 392],
        "image_grid_thw": [1, 28, 44],
        "patch_size": 14,
        "gt_box": [100, 50, 900, 550],
        "predicted_box": [110, 60, 890, 540],
        "prompt_frame": ORIGINAL_PIXEL_FRAME,
        "target_frame": ORIGINAL_PIXEL_FRAME,
        "evaluator_frame": ORIGINAL_PIXEL_FRAME,
    }


def test_bbox_roundtrip_preserves_original_coordinates():
    original = (1000, 600)
    processed = (616, 392)
    box = [100, 50, 900, 550]

    restored = scale_box(scale_box(box, original, processed), processed, original)

    assert max(abs(a - b) for a, b in zip(box, restored)) <= 1.0
    assert box_iou(box, restored) >= 0.999


def test_invalid_or_degenerate_box_is_rejected():
    with pytest.raises(ValueError):
        validate_box([5, 5, 5, 10], image_size=(100, 100))
    with pytest.raises(ValueError):
        validate_box([-1, 5, 10, 10], image_size=(100, 100))


def test_fixed_coordinate_interpretations_are_recorded_in_original_frame():
    raw = [100, 50, 900, 550]

    interpretations = interpret_box(
        raw,
        original_size=(1000, 600),
        processor_size=(500, 300),
    )

    assert interpretations["original_image_pixels"]["box"] == pytest.approx(raw)
    assert interpretations["processor_input_pixels"]["box"] == pytest.approx(
        [200, 100, 1800, 1100]
    )
    assert interpretations["normalized_0_1000"]["box"] == pytest.approx(
        [100, 30, 900, 330]
    )
    assert interpretations["processor_input_pixels"]["valid"] is False


def test_coordinate_record_contains_grid_transform_and_roundtrip_gate():
    result = audit_coordinate_record(_record())

    assert result["passed"] is True
    assert result["derived_processor_size"] == [616, 392]
    assert result["declared_frame"] == ORIGINAL_PIXEL_FRAME
    assert result["synthetic_roundtrip"]["max_coordinate_error"] <= 1.0
    assert result["synthetic_roundtrip"]["iou"] >= 0.999
    assert set(result["gt_interpretations"]) == {
        "original_image_pixels",
        "processor_input_pixels",
        "normalized_0_1000",
    }


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("gt_box", None, "missing_gt_box"),
        ("image_grid_thw", None, "missing_image_grid_thw"),
        ("processor_size", [615, 392], "processor_grid_size_mismatch"),
        ("prompt_frame", "normalized_0_1000", "inconsistent_declared_frames"),
    ],
)
def test_coordinate_record_fails_closed_on_protocol_defects(field, value, reason):
    record = _record()
    record[field] = value

    result = audit_coordinate_record(record)

    assert result["passed"] is False
    assert reason in result["blocked_reasons"]


def test_coordinate_report_requires_exactly_32_unique_families():
    short = audit_coordinate_records([_record("f01")])
    duplicate = audit_coordinate_records([_record("same") for _ in range(32)])

    assert short["status"] == BLOCKED_COORDINATE_PROTOCOL
    assert "expected_32_families_got_1" in short["blocked_reasons"]
    assert duplicate["status"] == BLOCKED_COORDINATE_PROTOCOL
    assert "duplicate_family_id" in duplicate["blocked_reasons"]


def test_coordinate_report_passes_32_valid_unique_families():
    report = audit_coordinate_records([_record(f"f{index:02d}") for index in range(32)])

    assert report["passed"] is True
    assert report["status"] == "PASSED_COORDINATE_PROTOCOL"
    assert report["valid_record_count"] == 32
    assert report["max_coordinate_error"] <= 1.0
    assert report["minimum_synthetic_iou"] >= 0.999


def test_collect_coordinate_records_uses_actual_vision_and_processor_sizes(tmp_path):
    image_path = tmp_path / "image.png"
    from PIL import Image

    Image.new("RGB", (1000, 600), "white").save(image_path)
    manifest = [{
        "id": "f01-original",
        "family_id": "f01",
        "role": "positive",
        "condition": "original",
        "prompt_view": "canonical",
        "gt_bbox": [100, 50, 900, 550],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "prompt"},
            ]},
        ],
    }]

    class FakeProcessor:
        image_processor = type("ImageProcessor", (), {"patch_size": 14})()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "rendered"

        def __call__(self, **kwargs):
            return {"image_grid_thw": [[1, 28, 44]]}

    def fake_vision_info(messages):
        return [Image.open(image_path).resize((840, 504))], []

    records = collect_coordinate_records(
        manifest,
        processor=FakeProcessor(),
        vision_info_fn=fake_vision_info,
        predicted_boxes={"f01-original": [110, 60, 890, 540]},
        expected_families=1,
    )

    assert records[0]["original_size"] == [1000, 600]
    assert records[0]["vision_size"] == [840, 504]
    assert records[0]["processor_size"] == [616, 392]
    assert records[0]["predicted_box"] == [110, 60, 890, 540]
