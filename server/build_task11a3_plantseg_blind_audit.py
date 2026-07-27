"""Build a model-blind PlantSeg source-quality audit from the verified archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from prepare_task11a3_plantseg_damage_null import (
    ARCHIVE_MD5,
    ARCHIVE_SIZE,
    RECORD_ID,
    parse_resolution,
)

ALLOWED_LICENSES = {"CC-BY-NC", "CC0"}
MEDIA_SUFFIXES = (".jpg", ".jpeg", ".png")


def direct_media_url(value: str) -> bool:
    return urlsplit(str(value).strip()).path.lower().endswith(MEDIA_SUFFIXES)


def metadata_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for source in rows:
        if source.get("Split") != "Validation":
            continue
        width, height = parse_resolution(source.get("Resolution", ""))
        ratio = float(source.get("Mask ratio", "nan"))
        if min(width, height) < 224 or not 0.02 <= ratio <= 0.40:
            continue
        if source.get("License") not in ALLOWED_LICENSES:
            continue
        if not direct_media_url(source.get("URL", "")):
            continue
        selected.append(dict(source))
    return sorted(selected, key=lambda row: (row["Name"], row["Label file"]))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_rgb(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as loaded:
        return loaded.convert("RGB")


def _load_mask(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as loaded:
        return loaded.convert("L")


def _audit_panel(image: Image.Image, mask: Image.Image, audit_id: str) -> Image.Image:
    canvas = Image.new("RGB", (900, 360), "white")
    left = image.copy()
    left.thumbnail((430, 310))
    canvas.paste(left, ((440 - left.width) // 2, 40 + (310 - left.height) // 2))
    overlay = image.copy()
    red = Image.new("RGB", overlay.size, (255, 0, 0))
    alpha = mask.point(lambda value: 105 if value > 0 else 0)
    overlay.paste(red, mask=alpha)
    overlay.thumbnail((430, 310))
    canvas.paste(overlay, (450 + (440 - overlay.width) // 2, 40 + (310 - overlay.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"audit_id={audit_id} | original", fill="black")
    draw.text((460, 10), "official mask overlay", fill="black")
    return canvas


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_blind_audit(
    *, archive_path: Path, prior_smoke_manifest: Path, output_root: Path, page_size: int = 20
) -> dict:
    if page_size < 1:
        raise ValueError("page_size must be positive")
    archive_path = Path(archive_path)
    if archive_path.stat().st_size != ARCHIVE_SIZE:
        raise ValueError("PlantSeg archive size mismatch")
    destination = Path(output_root)
    ensure_new_directory(destination)
    sheets_root = destination / "contact_sheets"
    sheets_root.mkdir()
    prior_hashes = {str(row["image_sha256"]) for row in _jsonl(prior_smoke_manifest)}

    all_rows: list[dict] = []
    panels: dict[str, Image.Image] = {}
    with zipfile.ZipFile(archive_path) as archive:
        metadata_payload = archive.read("plantseg/Metadata.csv")
        metadata = list(
            csv.DictReader(io.TextIOWrapper(io.BytesIO(metadata_payload), encoding="utf-8-sig"))
        )
        selected = metadata_candidates(metadata)
        for source in selected:
            image_member = f"plantseg/images/val/{source['Name']}"
            mask_member = f"plantseg/annotations/val/{source['Label file']}"
            image_payload = archive.read(image_member)
            mask_payload = archive.read(mask_member)
            image = _load_rgb(image_payload)
            mask = _load_mask(mask_payload)
            expected_size = parse_resolution(source["Resolution"])
            if image.size != expected_size or mask.size != expected_size:
                raise ValueError(f"dimension mismatch: {source['Name']}")
            observed_ratio = float((np.asarray(mask) > 0).mean())
            expected_ratio = float(source["Mask ratio"])
            tolerance = max(1e-6, 1.0 / (image.width * image.height))
            if abs(observed_ratio - expected_ratio) > tolerance:
                raise ValueError(f"mask ratio mismatch: {source['Name']}")
            image_sha = _sha256(image_payload)
            mask_sha = _sha256(mask_payload)
            row_id = _sha256(
                f"task11a3-blind-row|{RECORD_ID}|{source['Name']}|{image_sha}".encode()
            )[:24]
            all_rows.append(
                {
                    "row_id": row_id,
                    "source_name": source["Name"],
                    "source_mask_name": source["Label file"],
                    "plant": source["Plant"],
                    "disease": source["Disease"],
                    "license": source["License"],
                    "source_url": source["URL"],
                    "width": image.width,
                    "height": image.height,
                    "mask_ratio": observed_ratio,
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "prior_smoke": image_sha in prior_hashes,
                }
            )
            panels.setdefault(image_sha, _audit_panel(image, mask, image_sha[:16]))

    by_sha: dict[str, list[dict]] = defaultdict(list)
    by_url: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_sha[row["image_sha256"]].append(row)
        by_url[row["source_url"]].append(row)
    audit_rows = []
    exclusions = []
    for image_sha, group in sorted(by_sha.items()):
        ordered = sorted(group, key=lambda row: row["source_name"])
        if image_sha in prior_hashes:
            exclusions.extend({**row, "exclusion": "prior_smoke"} for row in ordered)
            continue
        canonical = ordered[0]
        audit_rows.append(
            {
                "audit_id": image_sha[:16],
                "image_sha256": image_sha,
                "mask_sha256": canonical["mask_sha256"],
                "row_id": canonical["row_id"],
            }
        )
        exclusions.extend({**row, "exclusion": "exact_content_duplicate"} for row in ordered[1:])

    for page_index, offset in enumerate(range(0, len(audit_rows), page_size), start=1):
        page_rows = audit_rows[offset : offset + page_size]
        sheet = Image.new("RGB", (900, 360 * len(page_rows)), "white")
        for index, row in enumerate(page_rows):
            sheet.paste(panels[row["image_sha256"]], (0, index * 360))
            row["contact_sheet"] = f"contact_sheets/page_{page_index:03d}.jpg"
            row["contact_sheet_row"] = index + 1
        sheet.save(sheets_root / f"page_{page_index:03d}.jpg", quality=92)

    private_rows = sorted(all_rows, key=lambda row: row["row_id"])
    audit_rows = sorted(audit_rows, key=lambda row: row["audit_id"])
    exclusions = sorted(exclusions, key=lambda row: (row["exclusion"], row["row_id"]))
    _write_jsonl(destination / "candidate_index_private.jsonl", private_rows)
    _write_jsonl(destination / "audit_manifest.jsonl", audit_rows)
    _write_jsonl(destination / "excluded_rows.jsonl", exclusions)
    with (destination / "review_template.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "audit_id", "real_photo", "lesion_visible", "no_visible_pest",
                "no_dominant_text", "no_collage", "mask_valid", "decision", "notes",
            ]
        )
        writer.writerows([[row["audit_id"], "", "", "", "", "", "", "", ""] for row in audit_rows])

    report = {
        "version": "task11a3-plantseg-blind-source-audit-1",
        "state": "completed",
        "decision": "PENDING_MANUAL_AUDIT",
        "archive_size": archive_path.stat().st_size,
        "archive_md5": ARCHIVE_MD5,
        "metadata_validation_rows": sum(row.get("Split") == "Validation" for row in metadata),
        "metadata_eligible_direct_rows": len(all_rows),
        "unique_content": len(by_sha),
        "prior_smoke_unique": len({row["image_sha256"] for row in all_rows if row["prior_smoke"]}),
        "unseen_unique_audit_rows": len(audit_rows),
        "exact_duplicate_excess_rows": sum(max(0, len(group) - 1) for group in by_sha.values()),
        "duplicate_source_url_groups": sum(len(group) > 1 for group in by_url.values()),
        "plants": dict(sorted(Counter(row["plant"] for row in all_rows).items())),
        "disease_count": len({row["disease"] for row in all_rows}),
        "contact_sheet_count": (len(audit_rows) + page_size - 1) // page_size,
        "model_outputs_read": False,
    }
    write_json_new(destination / "audit_report.json", report)
    signed = [
        "candidate_index_private.jsonl", "audit_manifest.jsonl", "excluded_rows.jsonl",
        "review_template.csv", "audit_report.json",
    ] + [str(path.relative_to(destination)).replace("\\", "/") for path in sorted(sheets_root.glob("*.jpg"))]
    with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prior-smoke-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(build_blind_audit(
        archive_path=args.archive,
        prior_smoke_manifest=args.prior_smoke_manifest,
        output_root=args.output_root,
        page_size=args.page_size,
    ), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
