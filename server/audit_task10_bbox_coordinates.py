"""Fail-closed Task 10A audit of the historical bbox coordinate chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from task10_audit_common import sha256_file, write_json_new


ORIGINAL_PIXEL_FRAME = "original_image_pixels"
PROCESSOR_PIXEL_FRAME = "processor_input_pixels"
NORMALIZED_FRAME = "normalized_0_1000"
BLOCKED_COORDINATE_PROTOCOL = "BLOCKED_COORDINATE_PROTOCOL"
PASSED_COORDINATE_PROTOCOL = "PASSED_COORDINATE_PROTOCOL"


def validate_box(box: Any, image_size: tuple[int, int] | list[int]) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box must contain four coordinates")
    width, height = map(float, image_size)
    if min(width, height) <= 0:
        raise ValueError("image dimensions must be positive")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("box coordinates must be finite")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box is outside its frame or degenerate")
    return values


def scale_box(
    box: list[int | float] | tuple[int | float, ...],
    from_size: tuple[int, int] | list[int],
    to_size: tuple[int, int] | list[int],
) -> list[float]:
    from_width, from_height = map(float, from_size)
    to_width, to_height = map(float, to_size)
    if min(from_width, from_height, to_width, to_height) <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = map(float, box)
    return [
        x1 * to_width / from_width,
        y1 * to_height / from_height,
        x2 * to_width / from_width,
        y2 * to_height / from_height,
    ]


def box_iou(a: Any, b: Any) -> float:
    a_values = [float(value) for value in a]
    b_values = [float(value) for value in b]
    ax1, ay1, ax2, ay2 = a_values
    bx1, by1, bx2, by2 = b_values
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _interpretation(box: list[float], original_size: tuple[int, int]) -> dict[str, Any]:
    try:
        validate_box(box, original_size)
        valid = True
    except ValueError:
        valid = False
    return {"box": box, "valid": valid, "output_frame": ORIGINAL_PIXEL_FRAME}


def interpret_box(
    box: list[int | float],
    *,
    original_size: tuple[int, int] | list[int],
    processor_size: tuple[int, int] | list[int],
) -> dict[str, dict[str, Any]]:
    """Map the three preregistered interpretations into original-image pixels."""
    raw = [float(value) for value in box]
    original_width, original_height = map(float, original_size)
    return {
        ORIGINAL_PIXEL_FRAME: _interpretation(raw, tuple(map(int, original_size))),
        PROCESSOR_PIXEL_FRAME: _interpretation(
            scale_box(raw, processor_size, original_size), tuple(map(int, original_size))
        ),
        NORMALIZED_FRAME: _interpretation(
            [
                raw[0] * original_width / 1000.0,
                raw[1] * original_height / 1000.0,
                raw[2] * original_width / 1000.0,
                raw[3] * original_height / 1000.0,
            ],
            tuple(map(int, original_size)),
        ),
    }


def _size(value: Any, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid_{field}")
    result = [int(item) for item in value]
    if min(result) <= 0:
        raise ValueError(f"invalid_{field}")
    return result


def audit_coordinate_record(record: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    result: dict[str, Any] = {"family_id": str(record.get("family_id", ""))}

    try:
        original_size = _size(record.get("original_size"), "original_size")
        vision_size = _size(record.get("vision_size"), "vision_size")
        processor_size = _size(record.get("processor_size"), "processor_size")
    except ValueError as exc:
        reasons.append(str(exc))
        original_size = vision_size = processor_size = [0, 0]

    grid = record.get("image_grid_thw")
    if not isinstance(grid, (list, tuple)) or len(grid) != 3:
        reasons.append("missing_image_grid_thw")
        derived_size = None
    else:
        try:
            _, grid_height, grid_width = [int(value) for value in grid]
            patch_size = int(record.get("patch_size"))
            if min(grid_height, grid_width, patch_size) <= 0:
                raise ValueError
            derived_size = [grid_width * patch_size, grid_height * patch_size]
            if processor_size != derived_size:
                reasons.append("processor_grid_size_mismatch")
            if any(value % 28 != 0 for value in processor_size):
                reasons.append("processor_size_not_multiple_of_28")
        except (TypeError, ValueError):
            derived_size = None
            reasons.append("invalid_image_grid_thw")

    frames = [
        record.get("prompt_frame"),
        record.get("target_frame"),
        record.get("evaluator_frame"),
    ]
    if len(set(frames)) != 1 or frames[0] != ORIGINAL_PIXEL_FRAME:
        reasons.append("inconsistent_declared_frames")

    gt_box = record.get("gt_box")
    if gt_box is None:
        reasons.append("missing_gt_box")
        gt_interpretations = None
    elif min(original_size) > 0:
        try:
            validate_box(gt_box, original_size)
            gt_interpretations = interpret_box(
                gt_box, original_size=original_size, processor_size=processor_size
            )
        except ValueError:
            reasons.append("invalid_gt_box")
            gt_interpretations = None
    else:
        gt_interpretations = None

    predicted_box = record.get("predicted_box")
    if predicted_box is not None and min(original_size) > 0:
        try:
            predicted_interpretations = interpret_box(
                predicted_box, original_size=original_size, processor_size=processor_size
            )
        except (TypeError, ValueError):
            predicted_interpretations = None
    else:
        predicted_interpretations = None

    if min(original_size + processor_size) > 0:
        width, height = original_size
        synthetic = [0.1 * width, 0.1 * height, 0.9 * width, 0.9 * height]
        restored = scale_box(
            scale_box(synthetic, original_size, processor_size), processor_size, original_size
        )
        max_error = max(abs(left - right) for left, right in zip(synthetic, restored))
        synthetic_iou = box_iou(synthetic, restored)
        if max_error > 1.0:
            reasons.append("roundtrip_coordinate_error_exceeds_1px")
        if synthetic_iou < 0.999:
            reasons.append("roundtrip_iou_below_0.999")
        roundtrip = {
            "original_box": synthetic,
            "restored_box": restored,
            "max_coordinate_error": max_error,
            "iou": synthetic_iou,
        }
    else:
        roundtrip = None

    result.update({
        "passed": not reasons,
        "blocked_reasons": sorted(set(reasons)),
        "original_size": original_size,
        "vision_size": vision_size,
        "processor_size": processor_size,
        "image_grid_thw": list(grid) if isinstance(grid, (list, tuple)) else None,
        "patch_size": record.get("patch_size"),
        "derived_processor_size": derived_size,
        "declared_frame": frames[0] if len(set(frames)) == 1 else None,
        "transform_original_to_processor": [
            processor_size[0] / original_size[0] if original_size[0] else None,
            processor_size[1] / original_size[1] if original_size[1] else None,
        ],
        "gt_box": gt_box,
        "gt_interpretations": gt_interpretations,
        "predicted_box": predicted_box,
        "predicted_interpretations": predicted_interpretations,
        "synthetic_roundtrip": roundtrip,
    })
    return result


def audit_coordinate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source = list(records)
    audited = [audit_coordinate_record(record) for record in source]
    reasons: list[str] = []
    family_ids = [str(record.get("family_id", "")) for record in source]
    if len(source) != 32:
        reasons.append(f"expected_32_families_got_{len(source)}")
    if len(family_ids) != len(set(family_ids)):
        reasons.append("duplicate_family_id")
    if any(not family_id for family_id in family_ids):
        reasons.append("missing_family_id")
    for record in audited:
        reasons.extend(record["blocked_reasons"])

    roundtrips = [record["synthetic_roundtrip"] for record in audited if record["synthetic_roundtrip"]]
    passed = not reasons
    return {
        "version": "task10a-bbox-coordinate-audit-v1",
        "passed": passed,
        "status": PASSED_COORDINATE_PROTOCOL if passed else BLOCKED_COORDINATE_PROTOCOL,
        "blocked_reasons": sorted(set(reasons)),
        "record_count": len(audited),
        "valid_record_count": sum(record["passed"] for record in audited),
        "declared_primary_frame": ORIGINAL_PIXEL_FRAME,
        "diagnostic_frames": [PROCESSOR_PIXEL_FRAME, NORMALIZED_FRAME],
        "max_coordinate_error": max(
            (record["max_coordinate_error"] for record in roundtrips), default=None
        ),
        "minimum_synthetic_iou": min((record["iou"] for record in roundtrips), default=None),
        "records": audited,
    }


def _grid_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def collect_coordinate_records(
    manifest: Iterable[dict[str, Any]],
    *,
    processor: Any,
    vision_info_fn: Callable,
    predicted_boxes: dict[str, Any] | None = None,
    expected_families: int = 32,
) -> list[dict[str, Any]]:
    positives = [
        row for row in manifest
        if str(row.get("role")) == "positive"
        and str(row.get("condition")) == "original"
        and str(row.get("prompt_view")) == "canonical"
    ]
    family_ids = [str(row.get("family_id", "")) for row in positives]
    if len(positives) != expected_families or len(set(family_ids)) != expected_families:
        raise ValueError(
            f"expected {expected_families} unique canonical positive families, got {len(positives)}"
        )
    boxes = predicted_boxes or {}
    records = []
    for row in positives:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"missing messages for {row.get('id')}")
        image_item = messages[1]["content"][0]
        image_path = Path(str(image_item["image"]))
        with Image.open(image_path) as opened:
            original_size = list(opened.size)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = vision_info_fn(messages)
        if not images or not hasattr(images[0], "size"):
            raise ValueError(f"missing processed vision image for {row.get('id')}")
        vision_size = list(images[0].size)
        inputs = processor(
            text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt"
        )
        if "image_grid_thw" not in inputs:
            raise ValueError(f"missing image_grid_thw for {row.get('id')}")
        grid = _grid_list(inputs["image_grid_thw"])
        patch_size = int(processor.image_processor.patch_size)
        processor_size = [grid[2] * patch_size, grid[1] * patch_size]
        identifier = str(row["id"])
        records.append({
            "id": identifier,
            "family_id": str(row["family_id"]),
            "image_path": str(image_path),
            "original_size": original_size,
            "vision_size": vision_size,
            "processor_size": processor_size,
            "image_grid_thw": grid,
            "patch_size": patch_size,
            "gt_box": row.get("gt_bbox"),
            "predicted_box": boxes.get(identifier),
            "prompt_frame": ORIGINAL_PIXEL_FRAME,
            "target_frame": ORIGINAL_PIXEL_FRAME,
            "evaluator_frame": ORIGINAL_PIXEL_FRAME,
        })
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prediction_boxes(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    from evaluate_task9d import _parse

    result = {}
    for row in _read_jsonl(path):
        parsed = _parse(str(row.get("raw_text", "")))
        value = parsed.get("value") if parsed.get("schema_valid") else None
        result[str(row["id"])] = value.get("evidence_region") if value else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    args = parser.parse_args()

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=False,
        local_files_only=True,
    )
    records = collect_coordinate_records(
        _read_jsonl(args.manifest),
        processor=processor,
        vision_info_fn=process_vision_info,
        predicted_boxes=_prediction_boxes(args.predictions),
    )
    report = audit_coordinate_records(records)
    report["inputs"] = {
        "manifest_sha256": sha256_file(args.manifest),
        "predictions_sha256": (
            sha256_file(args.predictions) if args.predictions is not None else None
        ),
        "model_path": str(args.model_path),
    }
    write_json_new(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
