#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from audit_full_datasets import parse_annotation


SPLITS = ("train", "val", "test")
OUTPUT_KEYS = (
    "evidence_present",
    "evidence_bbox",
    "visible_attributes",
    "diagnosis",
    "reliability",
)


def load_class_map(path: Path) -> dict[int, str]:
    class_map = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid class line {line_number}: {line!r}")
        declared_id, name = parts
        class_map[int(declared_id) - 1] = " ".join(name.split())
    return class_map


def stable_bucket(seed: str, value: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def make_messages(image: str, question: str, target: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                }
            ],
        },
    ]


def valid_objects(annotation_root, class_map: dict[int, str]) -> tuple[list[dict], int]:
    width = int(annotation_root.findtext("size/width", "0"))
    height = int(annotation_root.findtext("size/height", "0"))
    objects = []
    invalid = 0
    for obj in annotation_root.findall("object"):
        try:
            class_id = int(obj.findtext("name", ""))
            box = obj.find("bndbox")
            xmin = int(float(box.findtext("xmin")))
            ymin = int(float(box.findtext("ymin")))
            xmax = int(float(box.findtext("xmax")))
            ymax = int(float(box.findtext("ymax")))
        except (AttributeError, TypeError, ValueError):
            invalid += 1
            continue
        if class_id not in class_map or xmax <= xmin or ymax <= ymin:
            invalid += 1
            continue
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            invalid += 1
            continue
        objects.append(
            {
                "pest_id": class_id,
                "pest_name": class_map[class_id],
                "bbox": [xmin, ymin, xmax, ymax],
                "area": (xmax - xmin) * (ymax - ymin),
            }
        )
    return objects, invalid


def build_detection_records(
    voc_root: Path,
    class_map: dict[int, str],
    image_prefix: str,
    val_percent: int = 10,
    seed: str = "eviagridiag-v1",
) -> tuple[dict, dict]:
    if not 0 <= val_percent < 100:
        raise ValueError("val_percent must be in [0, 100)")
    voc_root = Path(voc_root)
    bundle = {split: {"positive": [], "null": []} for split in SPLITS}
    image_by_id = {
        path.stem: path
        for path in sorted((voc_root / "JPEGImages").iterdir())
        if path.is_file()
    }
    source_splits = {}
    for source_split in ("trainval", "test"):
        split_path = voc_root / "ImageSets" / "Main" / f"{source_split}.txt"
        for image_id in split_path.read_text(encoding="utf-8").splitlines():
            image_id = image_id.strip()
            if image_id:
                source_splits[image_id] = source_split

    summary = {
        "version": "eviagridiag-detection-v1",
        "seed": seed,
        "val_percent": val_percent,
        "source_split_images": {"trainval": 0, "test": 0},
        "valid_images": 0,
        "skipped_missing_image_or_xml": 0,
        "skipped_without_valid_boxes": 0,
        "invalid_boxes_dropped": 0,
        "duplicate_root_annotations_repaired": 0,
        "output_counts": {},
        "class_map": {str(key): value for key, value in sorted(class_map.items())},
    }

    all_class_ids = sorted(class_map)
    for image_id, source_split in sorted(source_splits.items()):
        summary["source_split_images"][source_split] += 1
        image_path = image_by_id.get(image_id)
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        if image_path is None or not xml_path.is_file():
            summary["skipped_missing_image_or_xml"] += 1
            continue
        root, duplicate_root, parse_error = parse_annotation(xml_path)
        if parse_error or root is None:
            summary["skipped_missing_image_or_xml"] += 1
            continue
        if duplicate_root:
            summary["duplicate_root_annotations_repaired"] += 1
        objects, invalid_count = valid_objects(root, class_map)
        summary["invalid_boxes_dropped"] += invalid_count
        if not objects:
            summary["skipped_without_valid_boxes"] += 1
            continue

        if source_split == "test":
            split = "test"
        elif stable_bucket(seed, image_id, 100) < val_percent:
            split = "val"
        else:
            split = "train"

        primary = max(objects, key=lambda item: (item["area"], -item["pest_id"]))
        present_ids = {item["pest_id"] for item in objects}
        absent_ids = [class_id for class_id in all_class_ids if class_id not in present_ids]
        if not absent_ids:
            raise ValueError(f"no absent class available for null sample: {image_id}")
        query_pest_id = absent_ids[stable_bucket(seed + ":null", image_id, len(absent_ids))]
        query_pest_name = class_map[query_pest_id]
        image = f"{image_prefix.rstrip('/')}/{image_path.name}"

        positive_target = {
            "evidence_present": True,
            "evidence_bbox": primary["bbox"],
            "visible_attributes": [],
            "diagnosis": {"pest_id": primary["pest_id"], "pest_name": primary["pest_name"]},
            "reliability": "supported",
        }
        positive_question = (
            "Identify the pest supported by visible evidence. Return one JSON object in this exact order: "
            "evidence_present, evidence_bbox, visible_attributes, diagnosis, reliability. "
            "Use image-pixel coordinates and do not infer attributes that are not annotated."
        )
        positive = {
            "id": f"ip102det_{split}_{image_id}_positive",
            "image": image,
            "source": "ip102_detection",
            "split": split,
            "source_split": source_split,
            "task_type": "pest_evidence_grounding",
            "question": positive_question,
            "target": positive_target,
            "metadata": {"image_id": image_id, "all_valid_objects": objects},
        }
        positive["messages"] = make_messages(image, positive_question, positive_target)

        null_target = {
            "evidence_present": False,
            "evidence_bbox": None,
            "visible_attributes": [],
            "diagnosis": "uncertain",
            "reliability": "insufficient_visual_evidence",
        }
        null_question = (
            f"Is {query_pest_name} visibly present in this image? Return one JSON object in this exact order: "
            "evidence_present, evidence_bbox, visible_attributes, diagnosis, reliability. "
            "If the queried pest is not supported, use null evidence and an uncertain diagnosis."
        )
        null_record = {
            "id": f"ip102det_{split}_{image_id}_null_{query_pest_id}",
            "image": image,
            "source": "ip102_detection",
            "split": split,
            "source_split": source_split,
            "task_type": "prompt_conflict_null_evidence",
            "question": null_question,
            "query_pest_id": query_pest_id,
            "query_pest_name": query_pest_name,
            "target": null_target,
            "metadata": {"image_id": image_id, "present_pest_ids": sorted(present_ids)},
        }
        null_record["messages"] = make_messages(image, null_question, null_target)
        bundle[split]["positive"].append(positive)
        bundle[split]["null"].append(null_record)
        summary["valid_images"] += 1

    summary["output_counts"] = {
        split: {kind: len(bundle[split][kind]) for kind in ("positive", "null")}
        for split in SPLITS
    }
    return bundle, summary


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_jsonl_bundle(bundle: dict, output_root: Path, summary: dict) -> None:
    output_root = Path(output_root)
    for split in SPLITS:
        write_jsonl(output_root / "vlm_sft" / f"{split}_evidence_positive.jsonl", bundle[split]["positive"])
        write_jsonl(output_root / "hallucination" / f"{split}_prompt_conflict.jsonl", bundle[split]["null"])
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--image-prefix", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--seed", default="eviagridiag-v1")
    args = parser.parse_args()
    class_map = load_class_map(args.classes_file)
    bundle, summary = build_detection_records(
        args.voc_root,
        class_map,
        image_prefix=args.image_prefix,
        val_percent=args.val_percent,
        seed=args.seed,
    )
    write_jsonl_bundle(bundle, args.output_root, summary)
    print(json.dumps(summary["output_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
