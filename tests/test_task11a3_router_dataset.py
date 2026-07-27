import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_task11a3_router_dataset import (
    AUDIT_MANIFEST_SHA256,
    FINAL_AUDIT_SHA256,
    PRIVATE_INDEX_SHA256,
    build_router_dataset,
)
from task10_audit_common import sha256_file


def test_frozen_source_hashes_are_complete_sha256_values():
    assert AUDIT_MANIFEST_SHA256 == "55e9f80146dc1f20c306f9e3bbee790189bc83636bdf80a4fd7463e4e0ee8bb9"
    assert all(len(value) == 64 for value in (
        AUDIT_MANIFEST_SHA256, FINAL_AUDIT_SHA256, PRIVATE_INDEX_SHA256,
    ))


def _image_bytes(mode: str, fmt: str) -> bytes:
    output = io.BytesIO()
    Image.new(mode, (16, 16), 1 if mode == "L" else (20, 80, 40)).save(output, format=fmt)
    return output.getvalue()


def test_router_dataset_materializes_only_human_keep_rows(tmp_path):
    image, mask = _image_bytes("RGB", "JPEG"), _image_bytes("L", "PNG")
    archive = tmp_path / "plantseg.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("plantseg/images/val/x.jpg", image)
        z.writestr("plantseg/annotations/val/x.png", mask)
    final_rows = [
        {"audit_id": f"id{i:03d}", "strict_real_null_eligible": i == 0}
        for i in range(338)
    ]
    final_audit = tmp_path / "final.jsonl"
    final_audit.write_text("".join(json.dumps(row) + "\n" for row in final_rows), encoding="utf-8")
    final_report = tmp_path / "report.json"
    final_report.write_text(json.dumps({
        "human_audit_status": "COMPLETE", "decision": "READY_FOR_FROZEN_ROUTER",
        "strict_real_null_candidates": 1,
    }), encoding="utf-8")
    audit_manifest = tmp_path / "audit.jsonl"
    audit_rows = [
        {"audit_id": f"id{i:03d}", "row_id": f"row{i:03d}",
         "image_sha256": hashlib.sha256(image).hexdigest() if i == 0 else f"image{i}",
         "mask_sha256": hashlib.sha256(mask).hexdigest() if i == 0 else f"mask{i}"}
        for i in range(338)
    ]
    audit_manifest.write_text("".join(json.dumps(row) + "\n" for row in audit_rows), encoding="utf-8")
    private = tmp_path / "private.jsonl"
    private.write_text(json.dumps({
        "row_id": "row000", "source_name": "x.jpg", "source_mask_name": "x.png",
        "image_sha256": hashlib.sha256(image).hexdigest(), "mask_sha256": hashlib.sha256(mask).hexdigest(),
        "width": 16, "height": 16, "plant": "Apple", "disease": "test",
        "license": "CC0", "mask_ratio": 1.0,
    }) + "\n", encoding="utf-8")
    empty_rows = tmp_path / "empty.jsonl"
    empty_rows.write_text("", encoding="utf-8")

    report = build_router_dataset(
        archive_path=archive, audit_manifest_path=audit_manifest,
        private_index_path=private, final_audit_path=final_audit,
        final_report_path=final_report, task10b_feature_rows=empty_rows,
        task11a2_feature_rows=empty_rows, output_root=tmp_path / "out",
        expected_archive_size=archive.stat().st_size,
        expected_archive_md5=hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest(),
        expected_final_audit_sha256=sha256_file(final_audit),
        expected_audit_manifest_sha256=sha256_file(audit_manifest),
        expected_private_index_sha256=sha256_file(private), expected_keep=1,
    )

    assert report["image_count"] == 1
    assert report["prior_content_overlap"] == 0
    row = json.loads((tmp_path / "out" / "manifest.jsonl").read_text())
    assert row["id"] == "id000" and row["input_contract"] == "pixels_only"
    assert Path(row["image"]).read_bytes() == image
