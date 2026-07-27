"""Materialize the frozen 290-image PlantSeg router audit from the verified archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


ARCHIVE_SIZE = 1_057_281_724
ARCHIVE_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
FINAL_AUDIT_SHA256 = "56d8256949315a420cc57fd71bfeb72105020eb5900de500f2c520cb6ae9efb2"
AUDIT_MANIFEST_SHA256 = "55e9f80146dc1f20c306f9e3bbee790189bc83636bdf80a4fd7463e4e0ee8bb9"
PRIVATE_INDEX_SHA256 = "5e3dfc1ab183f6e1ef684f163dd347342f4cec8324daac12762e38bdbd486246"
EXPECTED_KEEP = 290


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hashes(path: Path, field: str) -> set[str]:
    return {str(row[field]) for row in _read_jsonl(path)}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_router_dataset(
    *, archive_path: Path, audit_manifest_path: Path, private_index_path: Path,
    final_audit_path: Path, final_report_path: Path, task10b_feature_rows: Path,
    task11a2_feature_rows: Path, output_root: Path,
    expected_archive_size: int = ARCHIVE_SIZE,
    expected_archive_md5: str = ARCHIVE_MD5,
    expected_final_audit_sha256: str = FINAL_AUDIT_SHA256,
    expected_audit_manifest_sha256: str = AUDIT_MANIFEST_SHA256,
    expected_private_index_sha256: str = PRIVATE_INDEX_SHA256,
    expected_keep: int = EXPECTED_KEEP,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    audit_manifest_path = Path(audit_manifest_path)
    private_index_path = Path(private_index_path)
    final_audit_path = Path(final_audit_path)
    final_report_path = Path(final_report_path)
    if archive_path.stat().st_size != expected_archive_size or _md5(archive_path) != expected_archive_md5:
        raise ValueError("PlantSeg archive identity mismatch")
    if sha256_file(final_audit_path) != expected_final_audit_sha256:
        raise ValueError("final human audit SHA256 mismatch")
    if sha256_file(audit_manifest_path) != expected_audit_manifest_sha256:
        raise ValueError("blind audit manifest SHA256 mismatch")
    if sha256_file(private_index_path) != expected_private_index_sha256:
        raise ValueError("private candidate index SHA256 mismatch")
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    if (
        final_report.get("human_audit_status") != "COMPLETE"
        or final_report.get("decision") != "READY_FOR_FROZEN_ROUTER"
        or int(final_report.get("strict_real_null_candidates", -1)) != expected_keep
    ):
        raise ValueError("final human audit report is not router-ready")
    final_rows = _read_jsonl(final_audit_path)
    keep_ids = {str(row["audit_id"]) for row in final_rows if row.get("strict_real_null_eligible") is True}
    if len(final_rows) != 338 or len(keep_ids) != expected_keep:
        raise ValueError("unexpected final human audit cardinality")
    audit_by_id = {str(row["audit_id"]): row for row in _read_jsonl(audit_manifest_path)}
    private_by_row = {str(row["row_id"]): row for row in _read_jsonl(private_index_path)}
    if len(audit_by_id) != 338 or not keep_ids <= set(audit_by_id):
        raise ValueError("final audit does not align with blind audit manifest")
    prior_hashes = _hashes(Path(task10b_feature_rows), "source_image_sha256")
    prior_hashes |= _hashes(Path(task11a2_feature_rows), "image_sha256")
    destination = Path(output_root)
    ensure_new_directory(destination)
    images_root, masks_root = destination / "images", destination / "masks"
    images_root.mkdir()
    masks_root.mkdir()
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for audit_id in sorted(keep_ids):
            audit = audit_by_id[audit_id]
            private = private_by_row.get(str(audit["row_id"]))
            if private is None:
                raise ValueError(f"missing private provenance for {audit_id}")
            image_member = f"plantseg/images/val/{private['source_name']}"
            mask_member = f"plantseg/annotations/val/{private['source_mask_name']}"
            image_payload, mask_payload = archive.read(image_member), archive.read(mask_member)
            image_sha, mask_sha = hashlib.sha256(image_payload).hexdigest(), hashlib.sha256(mask_payload).hexdigest()
            if image_sha != audit["image_sha256"] or image_sha != private["image_sha256"]:
                raise ValueError(f"source image SHA256 mismatch: {audit_id}")
            if mask_sha != audit["mask_sha256"] or mask_sha != private["mask_sha256"]:
                raise ValueError(f"source mask SHA256 mismatch: {audit_id}")
            if image_sha in prior_hashes:
                raise ValueError(f"router audit overlaps prior train/audit content: {audit_id}")
            prior_hashes.add(image_sha)
            image_suffix = Path(str(private["source_name"])).suffix.lower()
            mask_suffix = Path(str(private["source_mask_name"])).suffix.lower()
            if image_suffix not in {".jpg", ".jpeg", ".png"} or mask_suffix != ".png":
                raise ValueError(f"unexpected archive member extension: {audit_id}")
            image_path = images_root / f"{audit_id}{image_suffix}"
            mask_path = masks_root / f"{audit_id}.png"
            image_path.write_bytes(image_payload)
            mask_path.write_bytes(mask_payload)
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size or image.size != (int(private["width"]), int(private["height"])):
                    raise ValueError(f"image/mask geometry mismatch: {audit_id}")
            rows.append({
                "id": audit_id,
                "condition": "external_damage_real_null",
                "plant": str(private["plant"]),
                "disease": str(private["disease"]),
                "license": str(private["license"]),
                "mask_ratio": float(private["mask_ratio"]),
                "image_sha256": image_sha,
                "mask_sha256": mask_sha,
                "image": str(image_path),
                "mask": str(mask_path),
                "input_contract": "pixels_only",
            })
    if len(rows) != expected_keep or len({row["image_sha256"] for row in rows}) != expected_keep:
        raise ValueError("router dataset cardinality or uniqueness mismatch")
    manifest_path = destination / "manifest.jsonl"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "version": "task11a3-router-dataset-1",
        "state": "completed",
        "decision": "READY_FOR_FROZEN_FEATURE_EXTRACTION",
        "image_count": len(rows),
        "unique_image_count": len({row["image_sha256"] for row in rows}),
        "prior_content_overlap": 0,
        "input_contract": "pixels_only",
        "source_split": "Validation",
        "final_audit_sha256": sha256_file(final_audit_path),
        "audit_manifest_sha256": sha256_file(audit_manifest_path),
        "private_index_sha256": sha256_file(private_index_path),
        "manifest_sha256": sha256_file(manifest_path),
        "raw_data_deleted": False,
    }
    write_json_new(destination / "dataset_report.json", report)
    signed = ["manifest.jsonl", "dataset_report.json"]
    signed += [str(path.relative_to(destination)).replace("\\", "/") for path in sorted(images_root.iterdir())]
    signed += [str(path.relative_to(destination)).replace("\\", "/") for path in sorted(masks_root.iterdir())]
    with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--private-index", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--task10b-feature-rows", type=Path, required=True)
    parser.add_argument("--task11a2-feature-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_router_dataset(
        archive_path=args.archive, audit_manifest_path=args.audit_manifest,
        private_index_path=args.private_index, final_audit_path=args.final_audit,
        final_report_path=args.final_report, task10b_feature_rows=args.task10b_feature_rows,
        task11a2_feature_rows=args.task11a2_feature_rows, output_root=args.output_root,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
