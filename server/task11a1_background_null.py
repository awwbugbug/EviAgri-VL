"""Build and materialize paired background-only nulls for Task 11A.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


SPLIT_CONFIG = {
    "val": {
        "crop_size": 60,
        "margin_fraction": 0.05,
        "grid": 17,
        "bottom_exclusion_fraction": 0.10,
    },
    "dev": {
        "crop_size": 72,
        "margin_fraction": 0.08,
        "grid": 19,
        "bottom_exclusion_fraction": 0.12,
    },
}
BANDS = ("head", "medium", "tail")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def parse_annotation(path: Path) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid VOC dimensions: {path}")
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"missing VOC bbox: {path}")
        values = tuple(
            float(box.findtext(name, "nan"))
            for name in ("xmin", "ymin", "xmax", "ymax")
        )
        if not (0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height):
            raise ValueError(f"invalid VOC bbox geometry: {path}")
        boxes.append(values)
    if not boxes:
        raise ValueError(f"annotation has no target boxes: {path}")
    return width, height, boxes


def expand_boxes(
    boxes: Iterable[tuple[float, float, float, float]],
    *,
    width: int,
    height: int,
    margin_fraction: float,
) -> list[tuple[float, float, float, float]]:
    margin = float(margin_fraction) * min(width, height)
    return [
        (
            max(0.0, x1 - margin),
            max(0.0, y1 - margin),
            min(float(width), x2 + margin),
            min(float(height), y2 + margin),
        )
        for x1, y1, x2, y2 in boxes
    ]


def intersects(
    first: tuple[int, int, int, int],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def candidate_crops(
    *,
    width: int,
    height: int,
    boxes: list[tuple[float, float, float, float]],
    crop_size: int,
    grid: int,
    bottom_exclusion_fraction: float = 0.0,
) -> list[tuple[int, int, int, int]]:
    if crop_size <= 0 or grid < 2 or width < crop_size or height < crop_size:
        return []
    candidates = []
    for iy in range(grid):
        top = round((height - crop_size) * iy / (grid - 1))
        for ix in range(grid):
            left = round((width - crop_size) * ix / (grid - 1))
            crop = (left, top, left + crop_size, top + crop_size)
            if crop[3] > (1.0 - bottom_exclusion_fraction) * height:
                continue
            if all(not intersects(crop, box) for box in boxes):
                candidates.append(crop)
    return sorted(set(candidates))


def choose_candidate(
    candidates: list[tuple[int, int, int, int]], *, split: str, source_id: str
) -> tuple[int, int, int, int]:
    if not candidates:
        raise ValueError("cannot choose from empty background candidates")
    digest = hashlib.sha256(
        f"task11a1-background-v1|{split}|{source_id}".encode("utf-8")
    ).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def build_rows(
    source_rows: Iterable[dict[str, Any]], annotation_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = []
    ineligible = []
    for row in source_rows:
        split = str(row.get("split"))
        if split not in SPLIT_CONFIG:
            continue
        config = SPLIT_CONFIG[split]
        annotation = Path(annotation_root) / f"{row['source_image_id']}.xml"
        if not annotation.is_file():
            raise ValueError(f"missing annotation: {annotation}")
        width, height, raw_boxes = parse_annotation(annotation)
        boxes = expand_boxes(
            raw_boxes,
            width=width,
            height=height,
            margin_fraction=float(config["margin_fraction"]),
        )
        candidates = candidate_crops(
            width=width,
            height=height,
            boxes=boxes,
            crop_size=int(config["crop_size"]),
            grid=int(config["grid"]),
            bottom_exclusion_fraction=float(config["bottom_exclusion_fraction"]),
        )
        common = {
            **row,
            "annotation": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "image_width": width,
            "image_height": height,
            "bbox_count": len(raw_boxes),
            "crop_size": int(config["crop_size"]),
            "margin_fraction": float(config["margin_fraction"]),
            "candidate_grid": int(config["grid"]),
            "bottom_exclusion_fraction": float(config["bottom_exclusion_fraction"]),
            "candidate_count": len(candidates),
        }
        if not candidates:
            ineligible.append({**common, "reason": "no_safe_background_crop"})
            continue
        crop = choose_candidate(candidates, split=split, source_id=str(row["source_image_id"]))
        if any(intersects(crop, box) for box in boxes):
            raise AssertionError("selected crop intersects expanded bbox")
        eligible.append({**common, "crop_xyxy": list(crop)})
    return eligible, ineligible


def select_smoke(rows: list[dict[str, Any]], per_split: int) -> list[dict[str, Any]]:
    if per_split <= 0 or per_split % len(BANDS):
        raise ValueError("smoke count must be positive and divisible by three bands")
    per_band = per_split // len(BANDS)
    selected = []
    for split in SPLIT_CONFIG:
        for band in BANDS:
            pool = [
                row
                for row in rows
                if row["split"] == split and row["class_band"] == band
            ]
            pool.sort(
                key=lambda row: hashlib.sha256(
                    f"task11a1-smoke|{split}|{row['id']}".encode("utf-8")
                ).hexdigest()
            )
            if len(pool) < per_band:
                raise ValueError(f"insufficient smoke rows: {split}/{band}")
            selected.extend(pool[:per_band])
    return selected


def write_protocol(
    *,
    source_manifest: Path,
    annotation_root: Path,
    output_root: Path,
    smoke_per_split: int | None,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    source_rows = _read_jsonl(Path(source_manifest))
    eligible, ineligible = build_rows(source_rows, Path(annotation_root))
    selected = select_smoke(eligible, smoke_per_split) if smoke_per_split else eligible
    _write_jsonl_new(destination / "manifest.jsonl", selected)
    _write_jsonl_new(destination / "ineligible.jsonl", ineligible)
    config = {
        "version": "task11a1-background-protocol-config-1",
        "source_manifest": str(Path(source_manifest)),
        "source_manifest_sha256": sha256_file(Path(source_manifest)),
        "annotation_root": str(Path(annotation_root)),
        "split_config": SPLIT_CONFIG,
        "smoke_per_split": smoke_per_split,
    }
    write_json_new(destination / "config.snapshot.json", config)
    by_split = {
        split: sum(row["split"] == split for row in selected) for split in SPLIT_CONFIG
    }
    by_band = {
        split: {
            band: sum(
                row["split"] == split and row["class_band"] == band for row in selected
            )
            for band in BANDS
        }
        for split in SPLIT_CONFIG
    }
    eligible_by_split = {
        split: sum(row["split"] == split for row in eligible) for split in SPLIT_CONFIG
    }
    report = {
        "version": "task11a1-background-protocol-report-1",
        "state": "completed",
        "selected_count": len(selected),
        "selected_by_split": by_split,
        "selected_by_band": by_band,
        "eligible_by_split": eligible_by_split,
        "ineligible_count": len(ineligible),
        "geometry_gate_passed": all(by_split.values())
        and all(value >= 4 for bands in by_band.values() for value in bands.values()),
    }
    write_json_new(destination / "protocol_report.json", report)
    signed = ["manifest.jsonl", "ineligible.jsonl", "config.snapshot.json", "protocol_report.json"]
    with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return report


def materialize(protocol_root: Path, output_root: Path) -> dict[str, Any]:
    protocol = Path(protocol_root)
    destination = Path(output_root)
    ensure_new_directory(destination)
    crops_root = destination / "crops"
    crops_root.mkdir()
    rows = _read_jsonl(protocol / "manifest.jsonl")
    derived = []
    thumbnails = []
    for row in rows:
        image_path = Path(str(row["image"]))
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        if image.size != (int(row["image_width"]), int(row["image_height"])):
            raise ValueError(f"image/XML dimension mismatch: {image_path}")
        crop_box = tuple(int(value) for value in row["crop_xyxy"])
        crop = image.crop(crop_box)
        expected = int(row["crop_size"])
        if crop.size != (expected, expected):
            raise ValueError(f"unexpected crop size: {row['id']}")
        name = f"{row['split']}_{row['source_image_id']}.png"
        path = crops_root / name
        crop.save(path, format="PNG", optimize=False)
        derived.append(
            {
                **row,
                "derived_image": str(path),
                "derived_sha256": sha256_file(path),
            }
        )
        thumb = crop.resize((192, 192), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (210, 220), "white")
        panel.paste(thumb, (9, 20))
        ImageDraw.Draw(panel).text(
            (8, 4), f"{row['split']} {row['source_image_id']}", fill="black"
        )
        thumbnails.append(panel)
    _write_jsonl_new(destination / "derived_manifest.jsonl", derived)
    columns = 4
    rows_count = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (210 * columns, 220 * rows_count), "white")
    for index, panel in enumerate(thumbnails):
        sheet.paste(panel, ((index % columns) * 210, (index // columns) * 220))
    sheet.save(destination / "audit_sheet.jpg", quality=95)
    report = {
        "version": "task11a1-background-materialization-report-1",
        "state": "completed",
        "derived_count": len(derived),
        "all_geometry_checked": True,
        "visual_gate": "PENDING_MANUAL_AUDIT",
        "protocol_manifest_sha256": sha256_file(protocol / "manifest.jsonl"),
    }
    write_json_new(destination / "materialization_report.json", report)
    signed = ["derived_manifest.jsonl", "audit_sheet.jpg", "materialization_report.json"] + [
        f"crops/{Path(row['derived_image']).name}" for row in derived
    ]
    with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 11A.1 background-only null protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--annotation-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--smoke-per-split", type=int)
    render = subparsers.add_parser("materialize")
    render.add_argument("--protocol-root", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        report = write_protocol(
            source_manifest=arguments.source_manifest,
            annotation_root=arguments.annotation_root,
            output_root=arguments.output_root,
            smoke_per_split=arguments.smoke_per_split,
        )
    else:
        report = materialize(arguments.protocol_root, arguments.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
