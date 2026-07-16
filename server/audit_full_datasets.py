#!/usr/bin/env python3
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def inspect_image(path: Path) -> tuple[bool, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        with Image.open(path) as image:
            image.verify()
        readable = True
    except Exception:
        readable = False
    return readable, digest.hexdigest()


def inspect_records(records: list[tuple[str, str, Path]], workers: int) -> tuple[list[str], list[dict]]:
    unreadable: list[str] = []
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(inspect_image, (record[2] for record in records))
        for (split, relative, _), (readable, digest) in zip(records, results):
            key = f"{split}/{relative}"
            if not readable:
                unreadable.append(key)
            by_hash[digest].append((split, key))
    cross_split_duplicates = []
    for digest, entries in sorted(by_hash.items()):
        if len({split for split, _ in entries}) > 1:
            cross_split_duplicates.append({"sha256": digest, "files": [key for _, key in entries]})
    return sorted(unreadable), cross_split_duplicates


def audit_classification_dataset(root: Path, class_directories: bool, workers: int = 8) -> dict:
    root = Path(root)
    records: list[tuple[str, str, Path]] = []
    split_image_counts: dict[str, int] = {}
    split_classes: dict[str, list[str]] = {}
    for split in SPLITS:
        split_root = root / split
        paths = image_files(split_root) if split_root.is_dir() else []
        split_image_counts[split] = len(paths)
        for path in paths:
            records.append((split, path.relative_to(split_root).as_posix(), path))
        if class_directories and split_root.is_dir():
            split_classes[split] = sorted(path.name for path in split_root.iterdir() if path.is_dir())

    unreadable, cross_split_duplicates = inspect_records(records, workers)
    output = {
        "root": str(root),
        "split_image_counts": split_image_counts,
        "total_images": sum(split_image_counts.values()),
        "unreadable_images": unreadable,
        "cross_split_content_duplicates": cross_split_duplicates,
    }

    if class_directories:
        classes = sorted({name for names in split_classes.values() for name in names})
        species = set()
        stages = set()
        malformed_classes = []
        for name in classes:
            if "-" not in name:
                malformed_classes.append(name)
                continue
            species_name, stage = name.rsplit("-", 1)
            species.add(species_name)
            stages.add(stage)
        output.update(
            {
                "class_count": len(classes),
                "classes": classes,
                "split_class_counts": {split: len(names) for split, names in split_classes.items()},
                "classes_missing_by_split": {
                    split: sorted(set(classes) - set(names)) for split, names in split_classes.items()
                },
                "species_count": len(species),
                "stages": sorted(stages),
                "malformed_classes": malformed_classes,
            }
        )
        return output

    classes_path = root / "classes.txt"
    declared_classes = [line.strip() for line in classes_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    manifest_missing_files = []
    images_not_in_manifest = []
    invalid_manifest_labels = []
    manifest_counts = {}
    for split in SPLITS:
        manifest_path = root / f"{split}.txt"
        declared_names = set()
        rows = [line.split() for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        manifest_counts[split] = len(rows)
        for parts in rows:
            if len(parts) != 2:
                invalid_manifest_labels.append(f"{split}/{' '.join(parts)}")
                continue
            name, label_text = parts
            try:
                label = int(label_text)
            except ValueError:
                invalid_manifest_labels.append(f"{split}/{name}:{label_text}")
                continue
            relative_name = f"{label}/{name}"
            declared_names.add(relative_name)
            if not 0 <= label < len(declared_classes):
                invalid_manifest_labels.append(f"{split}/{name}:{label_text}")
            if not (root / split / relative_name).is_file():
                manifest_missing_files.append(f"{split}/{relative_name}")
        actual_names = {relative for record_split, relative, _ in records if record_split == split}
        images_not_in_manifest.extend(f"{split}/{name}" for name in sorted(actual_names - declared_names))
    output.update(
        {
            "declared_class_count": len(declared_classes),
            "manifest_counts": manifest_counts,
            "manifest_missing_files": sorted(manifest_missing_files),
            "images_not_in_manifest": sorted(images_not_in_manifest),
            "invalid_manifest_labels": sorted(invalid_manifest_labels),
        }
    )
    return output


def parse_annotation(path: Path) -> tuple[ET.Element | None, bool, str | None]:
    try:
        return ET.parse(path).getroot(), False, None
    except ET.ParseError as error:
        text = path.read_text(encoding="utf-8", errors="replace")
        end = text.find("</annotation>")
        if text.count("<annotation") > 1 and end >= 0:
            try:
                return ET.fromstring(text[: end + len("</annotation>")]), True, None
            except ET.ParseError:
                pass
        return None, False, str(error)


def audit_detection_dataset(voc_root: Path, workers: int = 8) -> dict:
    voc_root = Path(voc_root)
    images = image_files(voc_root / "JPEGImages")
    image_by_id = {path.stem: path for path in images}
    xml_paths = sorted((voc_root / "Annotations").glob("*.xml"))
    xml_by_id = {path.stem: path for path in xml_paths}
    records = [("all", path.name, path) for path in images]
    unreadable, duplicate_images = inspect_records(records, workers)

    box_count = 0
    degenerate_box_count = 0
    out_of_bounds_box_count = 0
    duplicate_root_annotations = []
    annotation_parse_errors = []
    observed_class_ids = set()
    non_integer_class_names = []
    for path in xml_paths:
        root, duplicate_root, error = parse_annotation(path)
        if error:
            annotation_parse_errors.append(f"{path.name}:{error}")
            continue
        if duplicate_root:
            duplicate_root_annotations.append(path.name)
        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))
        for obj in root.findall("object"):
            box_count += 1
            name = obj.findtext("name", "").strip()
            try:
                observed_class_ids.add(int(name))
            except ValueError:
                non_integer_class_names.append(f"{path.name}:{name}")
            box = obj.find("bndbox")
            try:
                xmin = int(float(box.findtext("xmin")))
                ymin = int(float(box.findtext("ymin")))
                xmax = int(float(box.findtext("xmax")))
                ymax = int(float(box.findtext("ymax")))
            except (AttributeError, TypeError, ValueError):
                degenerate_box_count += 1
                continue
            if xmax <= xmin or ymax <= ymin:
                degenerate_box_count += 1
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                out_of_bounds_box_count += 1

    split_counts = {}
    split_ids = {}
    split_missing_images = []
    split_missing_annotations = []
    for split in ("trainval", "test"):
        path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        split_ids[split] = set(ids)
        split_counts[split] = len(ids)
        split_missing_images.extend(f"{split}/{image_id}" for image_id in ids if image_id not in image_by_id)
        split_missing_annotations.extend(f"{split}/{image_id}" for image_id in ids if image_id not in xml_by_id)

    return {
        "root": str(voc_root),
        "image_count": len(images),
        "annotation_count": len(xml_paths),
        "box_count": box_count,
        "observed_class_count": len(observed_class_ids),
        "observed_class_ids": sorted(observed_class_ids),
        "degenerate_box_count": degenerate_box_count,
        "out_of_bounds_box_count": out_of_bounds_box_count,
        "duplicate_root_annotations": sorted(duplicate_root_annotations),
        "annotation_parse_errors": sorted(annotation_parse_errors),
        "non_integer_class_names": sorted(non_integer_class_names),
        "images_without_annotations": sorted(set(image_by_id) - set(xml_by_id)),
        "annotations_without_images": sorted(set(xml_by_id) - set(image_by_id)),
        "unreadable_images": unreadable,
        "duplicate_image_content": duplicate_images,
        "split_counts": split_counts,
        "split_overlap": sorted(split_ids["trainval"] & split_ids["test"]),
        "split_missing_images": sorted(split_missing_images),
        "split_missing_annotations": sorted(split_missing_annotations),
        "images_not_in_splits": sorted(set(image_by_id) - (split_ids["trainval"] | split_ids["test"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--age-root", type=Path, required=True)
    parser.add_argument("--ip102-classification-root", type=Path, required=True)
    parser.add_argument("--ip102-detection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    report = {
        "age": audit_classification_dataset(args.age_root, class_directories=True, workers=args.workers),
        "ip102_classification": audit_classification_dataset(
            args.ip102_classification_root, class_directories=False, workers=args.workers
        ),
        "ip102_detection": audit_detection_dataset(args.ip102_detection_root, workers=args.workers),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: data.get("total_images", data.get("image_count")) for name, data in report.items()}))


if __name__ == "__main__":
    main()
