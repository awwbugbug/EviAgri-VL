from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalized_bbox_to_pixels(
    bbox: Any, width: int, height: int
) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return [x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000]


def bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def pointing_hit(predicted: list[float], truths: list[list[float]]) -> bool:
    center_x = (predicted[0] + predicted[2]) / 2
    center_y = (predicted[1] + predicted[3]) / 2
    return any(
        truth[0] <= center_x <= truth[2] and truth[1] <= center_y <= truth[3]
        for truth in truths
    )


def select_targets(records: list[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        for value in record["class_ids"].split("|"):
            by_class[int(value)].append(record)
    candidates: list[dict[str, Any]] = []
    for class_id, class_records in by_class.items():
        chosen = min(class_records, key=lambda record: record["image_sha256"])
        target = dict(chosen)
        target["target_class_id"] = class_id
        candidates.append(target)
    candidates.sort(key=lambda record: (record["image_sha256"], record["target_class_id"]))
    return candidates[:limit]


def target_class_name(record: dict[str, Any]) -> str:
    ids = [int(value) for value in record["class_ids"].split("|")]
    names = record["class_names"].split("|")
    return names[ids.index(int(record["target_class_id"]))]


def build_prompt(label: str) -> str:
    return (
        f'Target pest label: "{label}". Locate one visible instance that supports this label. '
        "Return JSON only, without markdown, using exactly: "
        '{"target_visible":true,"evidence_bbox":[x1,y1,x2,y2],'
        '"visible_attributes":["attribute"],"reliability":"supported|insufficient_evidence"}. '
        "Coordinates must be integers normalized to 0-1000. Do not return pixel coordinates. "
        "If no target instance is visibly supported, use target_visible=false, evidence_bbox=null, "
        "visible_attributes=[], reliability=insufficient_evidence. Do not invent attributes."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.dataset_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    targets = select_targets(records, args.limit)
    if len(targets) != args.limit:
        raise SystemExit(f"requested {args.limit} distinct classes, found {len(targets)}")

    manifest_path = args.model_manifest
    if manifest_path is None:
        value = os.environ.get("MODEL_MANIFEST")
        if not value:
            raise SystemExit("MODEL_MANIFEST is not set")
        manifest_path = Path(value)
    model_path = json.loads(manifest_path.read_text(encoding="utf-8"))["resolved_path"]

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28, use_fast=False
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.generation_config.temperature = None
    torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    result_path = args.output_dir / "grounding_results.jsonl"
    with result_path.open("w", encoding="utf-8") as output:
        for index, record in enumerate(targets, start=1):
            label = target_class_name(record)
            xml_root = ET.parse(args.dataset_root / record["annotation_file"]).getroot()
            width = int(xml_root.findtext("size/width", "0"))
            height = int(xml_root.findtext("size/height", "0"))
            truth_boxes = []
            for obj in xml_root.findall("object"):
                if int(obj.findtext("name", "-1")) == int(record["target_class_id"]):
                    truth_boxes.append(
                        [
                            float(obj.findtext("bndbox/xmin", "0")),
                            float(obj.findtext("bndbox/ymin", "0")),
                            float(obj.findtext("bndbox/xmax", "0")),
                            float(obj.findtext("bndbox/ymax", "0")),
                        ]
                    )
            if not truth_boxes:
                raise SystemExit(f"target class missing from annotation: {record['image_id']}")

            image_path = (args.dataset_root / record["image_file"]).resolve()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": build_prompt(label)},
                    ],
                }
            ]
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
            raw = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            parsed = extract_json(raw)
            predicted_normalized = parsed.get("evidence_bbox") if parsed else None
            predicted_pixels = normalized_bbox_to_pixels(
                predicted_normalized, width, height
            )
            max_iou = (
                max(bbox_iou(predicted_pixels, truth) for truth in truth_boxes)
                if predicted_pixels
                else 0.0
            )
            hit = pointing_hit(predicted_pixels, truth_boxes) if predicted_pixels else False
            result = {
                "index": index,
                "image_id": record["image_id"],
                "split": record["split"],
                "target_class_id": int(record["target_class_id"]),
                "target_class_name": label,
                "image_size": [width, height],
                "truth_boxes_pixels": truth_boxes,
                "raw": raw,
                "parsed": parsed,
                "predicted_bbox_normalized": predicted_normalized,
                "predicted_bbox_pixels": predicted_pixels,
                "max_iou": max_iou,
                "pointing_hit": hit,
                "seconds": seconds,
            }
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "index": index,
                        "parsed": parsed is not None,
                        "bbox_valid": predicted_pixels is not None,
                        "max_iou": round(max_iou, 4),
                        "pointing_hit": hit,
                        "seconds": round(seconds, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    valid = [result for result in results if result["predicted_bbox_pixels"] is not None]
    summary = {
        "status": "GROUNDING_PROBE_OK",
        "protocol": "oracle-label grounding; MVP only",
        "sample_count": len(results),
        "parsed_count": sum(result["parsed"] is not None for result in results),
        "valid_bbox_count": len(valid),
        "pointing_hits": sum(result["pointing_hit"] for result in results),
        "pointing_accuracy": sum(result["pointing_hit"] for result in results) / len(results),
        "mean_iou": sum(result["max_iou"] for result in results) / len(results),
        "iou_at_05": sum(result["max_iou"] >= 0.5 for result in results) / len(results),
        "mean_seconds": sum(result["seconds"] for result in results) / len(results),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (args.output_dir / "grounding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
