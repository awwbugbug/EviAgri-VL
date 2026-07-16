from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass
class AnnotationRecord:
    image_id: str
    split: str
    class_ids: tuple[int, ...]
    root: ET.Element
    duplicate_root_repaired: bool
    invalid_boxes_dropped: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rank_hash(seed: str, class_id: int, image_id: str) -> str:
    return sha256_bytes(f"{seed}|{class_id}|{image_id}".encode())


def read_classes(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            result[int(fields[0]) - 1] = fields[1].strip()
    return result


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def parse_annotation(path: Path, split: str) -> tuple[AnnotationRecord | None, str | None]:
    text = path.read_text(encoding="utf-8-sig")
    duplicate_repaired = text.count("<annotation") > 1
    if duplicate_repaired:
        closing = text.find("</annotation>")
        if closing < 0:
            return None, "duplicate_root_without_closing_tag"
        text = text[: closing + len("</annotation>")]
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        return None, f"parse_error:{error}"

    image_id = path.stem
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    invalid_boxes = 0
    class_ids: list[int] = []
    for obj in list(root.findall("object")):
        try:
            class_id = int(obj.findtext("name", ""))
            xmin = int(obj.findtext("bndbox/xmin", ""))
            ymin = int(obj.findtext("bndbox/ymin", ""))
            xmax = int(obj.findtext("bndbox/xmax", ""))
            ymax = int(obj.findtext("bndbox/ymax", ""))
            valid = (
                0 <= class_id <= 101
                and 0 <= xmin < xmax <= width
                and 0 <= ymin < ymax <= height
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            root.remove(obj)
            invalid_boxes += 1
            continue
        class_ids.append(class_id)

    if not class_ids:
        return None, f"no_valid_boxes:dropped={invalid_boxes}"
    return (
        AnnotationRecord(
            image_id=image_id,
            split=split,
            class_ids=tuple(sorted(set(class_ids))),
            root=root,
            duplicate_root_repaired=duplicate_repaired,
            invalid_boxes_dropped=invalid_boxes,
        ),
        None,
    )


def safe_image_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not member.isfile():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            result[path.stem] = member
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", default="ip102-detection-mvp-20260712-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise SystemExit(f"output root already exists: {args.output_root}")
    if args.per_class < 1:
        raise SystemExit("per-class must be positive")

    annotations_dir = args.voc_root / "Annotations"
    image_tar = args.voc_root / "JPEGImages.tar"
    classes = read_classes(args.classes_file)
    if not annotations_dir.is_dir() or not image_tar.is_file() or len(classes) < 2:
        raise SystemExit("VOC annotations, JPEGImages.tar, or class list is missing")

    trainval = read_split(args.voc_root / "ImageSets" / "Main" / "trainval.txt")
    test = read_split(args.voc_root / "ImageSets" / "Main" / "test.txt")
    if set(trainval) & set(test):
        raise SystemExit("trainval/test overlap")
    split_by_id = {image_id: "trainval" for image_id in trainval}
    split_by_id.update({image_id: "test" for image_id in test})

    records: list[AnnotationRecord] = []
    parse_errors: list[dict[str, str]] = []
    duplicate_repairs = 0
    invalid_boxes = 0
    no_valid = 0
    for path in sorted(annotations_dir.glob("*.xml")):
        record, error = parse_annotation(path, split_by_id.get(path.stem, "none"))
        if error:
            parse_errors.append({"image_id": path.stem, "error": error})
            if error.startswith("no_valid_boxes"):
                no_valid += 1
                invalid_boxes += int(error.rsplit("=", 1)[1])
            continue
        assert record is not None
        duplicate_repairs += int(record.duplicate_root_repaired)
        invalid_boxes += record.invalid_boxes_dropped
        records.append(record)

    by_class: dict[int, list[AnnotationRecord]] = defaultdict(list)
    for record in records:
        for class_id in record.class_ids:
            by_class[class_id].append(record)
    selected_ids: set[str] = set()
    for class_id, candidates in by_class.items():
        ranked = sorted(candidates, key=lambda record: rank_hash(args.seed, class_id, record.image_id))
        selected_ids.update(record.image_id for record in ranked[: args.per_class])
    selected = sorted((record for record in records if record.image_id in selected_ids), key=lambda r: r.image_id)

    images_out = args.output_root / "images"
    annotations_out = args.output_root / "annotations"
    images_out.mkdir(parents=True)
    annotations_out.mkdir(parents=True)

    manifest_rows: list[dict[str, str | int | bool]] = []
    with tarfile.open(image_tar, "r") as archive:
        members = safe_image_members(archive)
        for record in selected:
            member = members.get(record.image_id)
            if member is None:
                raise SystemExit(f"image missing from tar: {record.image_id}")
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"cannot read image member: {member.name}")
            image_bytes = source.read()
            image_path = images_out / Path(member.name).name
            image_path.write_bytes(image_bytes)

            ET.indent(record.root, space="  ")
            xml_bytes = ET.tostring(record.root, encoding="utf-8", xml_declaration=True)
            xml_path = annotations_out / f"{record.image_id}.xml"
            xml_path.write_bytes(xml_bytes)
            manifest_rows.append(
                {
                    "image_id": record.image_id,
                    "split": record.split,
                    "class_ids": "|".join(str(value) for value in record.class_ids),
                    "class_names": "|".join(classes[value] for value in record.class_ids),
                    "duplicate_root_repaired": record.duplicate_root_repaired,
                    "invalid_boxes_dropped": record.invalid_boxes_dropped,
                    "image_file": f"images/{image_path.name}",
                    "annotation_file": f"annotations/{xml_path.name}",
                    "image_bytes": len(image_bytes),
                    "image_sha256": sha256_bytes(image_bytes),
                    "annotation_sha256": sha256_bytes(xml_bytes),
                }
            )

    with (args.output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    audit = {
        "protocol": "deterministic_per_class_bbox_mvp",
        "selection_seed": args.seed,
        "per_class": args.per_class,
        "source_annotations": len(list(annotations_dir.glob("*.xml"))),
        "valid_annotations": len(records),
        "duplicate_root_repairs": duplicate_repairs,
        "invalid_boxes_dropped": invalid_boxes,
        "annotations_without_valid_boxes": no_valid,
        "parse_errors": parse_errors,
        "covered_classes": len(by_class),
        "missing_classes": sorted(set(classes) - set(by_class)),
        "selected_images": len(selected),
        "selected_trainval": sum(record.split == "trainval" for record in selected),
        "selected_test": sum(record.split == "test" for record in selected),
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "IP102_DETECTION_MVP_OK",
                "selected_images": len(selected),
                "covered_classes": len(by_class),
                "duplicate_root_repairs": duplicate_repairs,
                "invalid_boxes_dropped": invalid_boxes,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
