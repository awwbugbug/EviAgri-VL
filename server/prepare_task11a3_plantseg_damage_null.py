"""Prepare a deterministic PlantSeg damage-only real-null micro audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


RECORD_ID = "17719108"
ARCHIVE_URL = "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"
ARCHIVE_SIZE = 1_057_281_724
ARCHIVE_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
RECORD_LICENSE = "CC-BY-NC-4.0"
PLANTS = ("Apple", "Citrus", "Corn", "Grape", "Rice", "Soybean", "Tomato", "Wheat")


class HTTPRangeReader(io.RawIOBase):
    def __init__(
        self,
        url: str,
        size: int,
        *,
        block_size: int = 2 * 1024 * 1024,
        fetcher: Callable[[int, int], bytes] | None = None,
    ) -> None:
        self.url = url
        self.size = size
        self.block_size = block_size
        self.position = 0
        self.cache_start = -1
        self.cache = b""
        self.fetcher = fetcher or self._fetch_http

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError("invalid seek mode")
        if target < 0:
            raise ValueError("negative seek position")
        self.position = min(target, self.size)
        return self.position

    def _fetch_http(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end - 1}",
                "User-Agent": "EviAgri-VL",
            },
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                    if response.status != 206 or len(payload) != end - start:
                        raise IOError("invalid HTTP range response")
                    return payload
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        raise IOError(f"HTTP range fetch failed: {start}-{end}") from last_error

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        output = bytearray()
        while len(output) < size:
            if not (
                self.cache_start <= self.position < self.cache_start + len(self.cache)
            ):
                start = (self.position // self.block_size) * self.block_size
                end = min(self.size, max(start + self.block_size, self.position + size))
                self.cache = self.fetcher(start, end)
                if len(self.cache) != end - start:
                    raise IOError("range fetcher returned an unexpected length")
                self.cache_start = start
            offset = self.position - self.cache_start
            available = min(size - len(output), len(self.cache) - offset)
            output.extend(self.cache[offset : offset + available])
            self.position += available
        return bytes(output)


def parse_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", str(value).strip())
    if not match:
        raise ValueError(f"invalid PlantSeg resolution: {value}")
    return int(match.group(1)), int(match.group(2))


def deterministic_selection(
    rows: Iterable[dict[str, str]], *, images_per_plant: int
) -> list[dict[str, str]]:
    if images_per_plant not in {1, 3}:
        raise ValueError("Task 11A.3 only permits 1-image smoke or 3-image formal")
    eligible: dict[str, list[dict[str, str]]] = {plant: [] for plant in PLANTS}
    for row in rows:
        plant = str(row.get("Plant", ""))
        if plant not in eligible or str(row.get("Split")) != "Validation":
            continue
        width, height = parse_resolution(str(row.get("Resolution", "")))
        mask_ratio = float(row.get("Mask ratio", "nan"))
        if min(width, height) < 224 or not 0.02 <= mask_ratio <= 0.40:
            continue
        if str(row.get("License")) not in {"CC-BY-NC", "CC0"}:
            continue
        eligible[plant].append(dict(row))
    selected = []
    for plant in PLANTS:
        candidates = eligible[plant]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"task11a3|{RECORD_ID}|{ARCHIVE_MD5}|{plant}|{row['Name']}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        if len(candidates) < images_per_plant:
            raise ValueError(f"insufficient eligible PlantSeg rows: {plant}")
        selected.extend(candidates[:images_per_plant])
    return selected


def _read_hashes(path: Path, field: str) -> set[str]:
    return {
        str(json.loads(line).get(field))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _mask_ratio(payload: bytes) -> tuple[float, Image.Image]:
    with Image.open(io.BytesIO(payload)) as loaded:
        mask = loaded.convert("L")
    values = np.asarray(mask)
    return float((values > 0).mean()), mask


def prepare_dataset(
    *,
    output_root: Path,
    task10b_feature_rows: Path,
    task11a2_manifest: Path,
    images_per_plant: int,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    images_root = destination / "images"
    masks_root = destination / "masks"
    images_root.mkdir()
    masks_root.mkdir()
    prior_hashes = _read_hashes(Path(task10b_feature_rows), "source_image_sha256")
    prior_hashes |= _read_hashes(Path(task11a2_manifest), "image_sha256")
    with HTTPRangeReader(ARCHIVE_URL, ARCHIVE_SIZE) as remote, zipfile.ZipFile(
        remote
    ) as archive:
        metadata_payload = archive.read("plantseg/Metadata.csv")
        metadata_rows = list(
            csv.DictReader(io.StringIO(metadata_payload.decode("utf-8-sig")))
        )
        chosen = deterministic_selection(
            metadata_rows, images_per_plant=images_per_plant
        )
        rows = []
        panels = []
        for source in chosen:
            image_member = f"plantseg/images/val/{source['Name']}"
            mask_member = f"plantseg/annotations/val/{source['Label file']}"
            image_info = archive.getinfo(image_member)
            mask_info = archive.getinfo(mask_member)
            image_payload = archive.read(image_info)
            mask_payload = archive.read(mask_info)
            image_sha = hashlib.sha256(image_payload).hexdigest()
            mask_sha = hashlib.sha256(mask_payload).hexdigest()
            if image_sha in prior_hashes:
                raise ValueError("PlantSeg overlaps an earlier frozen audit/training image")
            prior_hashes.add(image_sha)
            with Image.open(io.BytesIO(image_payload)) as loaded:
                image = loaded.convert("RGB")
            expected_width, expected_height = parse_resolution(source["Resolution"])
            if image.size != (expected_width, expected_height):
                raise ValueError(f"PlantSeg resolution mismatch: {source['Name']}")
            observed_ratio, mask = _mask_ratio(mask_payload)
            expected_ratio = float(source["Mask ratio"])
            if mask.size != image.size or abs(observed_ratio - expected_ratio) > 1e-6:
                raise ValueError(f"PlantSeg mask ratio mismatch: {source['Name']}")
            image_name = f"{image_sha}.jpg"
            mask_name = f"{mask_sha}.png"
            (images_root / image_name).write_bytes(image_payload)
            (masks_root / mask_name).write_bytes(mask_payload)
            sample_id = hashlib.sha256(
                f"task11a3|{RECORD_ID}|{ARCHIVE_MD5}|{source['Plant']}|{source['Name']}".encode(
                    "utf-8"
                )
            ).hexdigest()[:32]
            rows.append(
                {
                    "id": sample_id,
                    "condition": "external_damage_real_null",
                    "plant": source["Plant"],
                    "disease": source["Disease"],
                    "source_split": "Validation",
                    "source_name": source["Name"],
                    "source_url": source["URL"],
                    "source_license": source["License"],
                    "source_image_member": image_member,
                    "source_mask_member": mask_member,
                    "source_image_crc32": f"{image_info.CRC:08x}",
                    "source_mask_crc32": f"{mask_info.CRC:08x}",
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "image": str(images_root / image_name),
                    "mask": str(masks_root / mask_name),
                    "width": image.width,
                    "height": image.height,
                    "mask_ratio": observed_ratio,
                }
            )
            display = image.copy()
            display.thumbnail((420, 300))
            panel = Image.new("RGB", (440, 350), "white")
            panel.paste(display, ((440 - display.width) // 2, 42))
            ImageDraw.Draw(panel).text(
                (8, 8), f"{source['Plant']} | {source['Disease']}", fill="black"
            )
            panels.append(panel)
    expected_count = len(PLANTS) * images_per_plant
    if len(rows) != expected_count or len({row["image_sha256"] for row in rows}) != expected_count:
        raise ValueError("unexpected or duplicate Task 11A.3 selection")
    (destination / "Metadata.csv").write_bytes(metadata_payload)
    with (destination / "manifest.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    columns = 2 if images_per_plant == 1 else 4
    rows_count = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (440 * columns, 350 * rows_count), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 440, (index // columns) * 350))
    sheet.save(destination / "audit_sheet.jpg", quality=95)
    report = {
        "version": "task11a3-plantseg-damage-null-report-1",
        "state": "completed",
        "record_id": RECORD_ID,
        "archive_url": ARCHIVE_URL,
        "archive_size": ARCHIVE_SIZE,
        "archive_md5": ARCHIVE_MD5,
        "record_license": RECORD_LICENSE,
        "source_split": "Validation",
        "plants": list(PLANTS),
        "images_per_plant": images_per_plant,
        "image_count": len(rows),
        "prior_content_overlap": 0,
        "visual_gate": "PENDING_MANUAL_AUDIT",
    }
    write_json_new(destination / "dataset_report.json", report)
    signed = ["Metadata.csv", "manifest.jsonl", "audit_sheet.jpg", "dataset_report.json"]
    signed += [f"images/{Path(row['image']).name}" for row in rows]
    signed += [f"masks/{Path(row['mask']).name}" for row in rows]
    with (destination / "completion.sha256").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Task 11A.3 PlantSeg nulls")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task10b-feature-rows", type=Path, required=True)
    parser.add_argument("--task11a2-manifest", type=Path, required=True)
    parser.add_argument("--images-per-plant", type=int, choices=(1, 3), required=True)
    arguments = parser.parse_args()
    report = prepare_dataset(
        output_root=arguments.output_root,
        task10b_feature_rows=arguments.task10b_feature_rows,
        task11a2_manifest=arguments.task11a2_manifest,
        images_per_plant=arguments.images_per_plant,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
