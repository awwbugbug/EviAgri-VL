"""Create a label-blind, native-resolution human review bundle for Task 11A.3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new

CRITERIA = (
    "real_photo",
    "lesion_visible",
    "no_visible_pest",
    "no_dominant_text",
    "no_collage",
    "mask_valid",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def review_decision(values: dict[str, str]) -> str:
    normalized = [str(values.get(key, "")).strip().upper() for key in CRITERIA]
    if any(value == "FAIL" for value in normalized):
        return "REJECT"
    if all(value == "PASS" for value in normalized):
        return "PASS"
    return "UNCERTAIN"


def validate_review_rows(rows: list[dict[str, str]], expected_ids: set[str]) -> dict:
    ids = [str(row.get("audit_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("review rows must contain every audit_id exactly once")
    reviewers = {str(row.get("reviewer_id", "")).strip() for row in rows}
    if "" in reviewers or len(reviewers) != 1:
        raise ValueError("one non-empty reviewer_id is required per review file")
    counts = {"PASS": 0, "REJECT": 0, "UNCERTAIN": 0}
    for row in rows:
        for criterion in CRITERIA:
            if str(row.get(criterion, "")).strip().upper() not in {"PASS", "FAIL", "UNCERTAIN"}:
                raise ValueError(f"invalid review value for {criterion}")
        counts[review_decision(row)] += 1
    return {"reviewer_id": next(iter(reviewers)), "counts": counts}


def _write_review_template(path: Path, audit_ids: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audit_id", "reviewer_id", *CRITERIA, "notes"])
        writer.writerows([[audit_id, "", *("" for _ in CRITERIA), ""] for audit_id in audit_ids])


def _html(audit_ids: list[str]) -> str:
    cards = []
    for audit_id in audit_ids:
        cards.append(
            f'<section><h2>{audit_id}</h2><div><img src="images/{audit_id}.jpg" '
            f'alt="{audit_id} original"><img src="overlays/{audit_id}.jpg" '
            f'alt="{audit_id} official mask overlay"></div></section>'
        )
    return """<!doctype html><meta charset="utf-8"><title>Task11A3 blind review</title>
<style>body{font-family:system-ui;margin:24px;background:#eee}section{background:white;margin:0 0 24px;padding:16px}h2{font:16px monospace}section div{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;max-height:72vh;object-fit:contain;background:#222}</style>
<h1>Task 11A.3 model-blind human review</h1>
<p>Review the native original and official-mask overlay. Do not consult disease labels, source metadata, or router outputs. Record PASS/FAIL/UNCERTAIN in an independent reviewer CSV.</p>
""" + "\n".join(cards)


def build_review_bundle(*, archive_path: Path, audit_root: Path, output_root: Path) -> dict:
    archive_path = Path(archive_path)
    audit_root = Path(audit_root)
    destination = Path(output_root)
    ensure_new_directory(destination)
    images_root = destination / "images"
    masks_root = destination / "masks"
    overlays_root = destination / "overlays"
    for root in (images_root, masks_root, overlays_root):
        root.mkdir()
    audit_rows = _read_jsonl(audit_root / "audit_manifest.jsonl")
    private_rows = _read_jsonl(audit_root / "candidate_index_private.jsonl")
    private_by_row = {row["row_id"]: row for row in private_rows}
    if len(private_by_row) != len(private_rows):
        raise ValueError("private candidate row IDs are not unique")
    output_rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for audit in sorted(audit_rows, key=lambda row: row["audit_id"]):
            private = private_by_row.get(audit["row_id"])
            if private is None or private["image_sha256"] != audit["image_sha256"]:
                raise ValueError("audit/private manifest mismatch")
            image_payload = archive.read(f"plantseg/images/val/{private['source_name']}")
            mask_payload = archive.read(f"plantseg/annotations/val/{private['source_mask_name']}")
            if hashlib.sha256(image_payload).hexdigest() != audit["image_sha256"]:
                raise ValueError("archive image hash mismatch")
            if hashlib.sha256(mask_payload).hexdigest() != audit["mask_sha256"]:
                raise ValueError("archive mask hash mismatch")
            audit_id = audit["audit_id"]
            with Image.open(io.BytesIO(image_payload)) as loaded:
                image = loaded.convert("RGB")
            with Image.open(io.BytesIO(mask_payload)) as loaded:
                mask = loaded.convert("L")
            image.save(images_root / f"{audit_id}.jpg", quality=96)
            mask.save(masks_root / f"{audit_id}.png")
            overlay = image.copy()
            red = Image.new("RGB", image.size, (255, 0, 0))
            overlay.paste(red, mask=mask.point(lambda value: 105 if value > 0 else 0))
            overlay.save(overlays_root / f"{audit_id}.jpg", quality=94)
            output_rows.append(
                {
                    "audit_id": audit_id,
                    "image": f"images/{audit_id}.jpg",
                    "mask": f"masks/{audit_id}.png",
                    "overlay": f"overlays/{audit_id}.jpg",
                    "source_image_sha256": audit["image_sha256"],
                    "source_mask_sha256": audit["mask_sha256"],
                }
            )
    with (destination / "review_manifest.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit_ids = [row["audit_id"] for row in output_rows]
    _write_review_template(destination / "reviewer_a.csv", audit_ids)
    _write_review_template(destination / "reviewer_b.csv", audit_ids)
    (destination / "review.html").write_text(_html(audit_ids), encoding="utf-8", newline="\n")
    report = {
        "version": "task11a3-human-review-bundle-1",
        "state": "completed",
        "audit_rows": len(output_rows),
        "criteria": list(CRITERIA),
        "required_reviewers": 2,
        "router_outputs_read": False,
        "labels_exposed": False,
        "decision": "PENDING_INDEPENDENT_HUMAN_REVIEW",
    }
    write_json_new(destination / "bundle_report.json", report)
    signed = ["review_manifest.jsonl", "reviewer_a.csv", "reviewer_b.csv", "review.html", "bundle_report.json"]
    for folder in (images_root, masks_root, overlays_root):
        signed.extend(str(path.relative_to(destination)).replace("\\", "/") for path in sorted(folder.iterdir()))
    with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_review_bundle(
        archive_path=args.archive, audit_root=args.audit_root, output_root=args.output_root
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
