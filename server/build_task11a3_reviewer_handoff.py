"""Build an isolated Task 11A.3 handoff bundle for one independent reviewer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_handoff(*, source_root: Path, output_root: Path, reviewer_id: str) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    reviewer_id = reviewer_id.strip()
    if not reviewer_id:
        raise ValueError("reviewer_id is required")
    ensure_new_directory(output_root)
    manifest = _read_jsonl(source_root / "review_manifest.jsonl")
    audit_ids = [str(row["audit_id"]) for row in manifest]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("review manifest audit IDs are not unique")
    (output_root / "images").mkdir()
    (output_root / "overlays").mkdir()
    transfer_modes: set[str] = set()
    signed: list[str] = []
    for audit_id in audit_ids:
        for folder in ("images", "overlays"):
            relative = Path(folder) / f"{audit_id}.jpg"
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            transfer_modes.add(_link_or_copy(source, output_root / relative))
            signed.append(relative.as_posix())
    for source_name, output_name in (
        ("review.html", "review.html"),
        ("review_manifest.jsonl", "review_manifest.jsonl"),
        ("reviewer_b.csv", "reviewer_b.csv"),
    ):
        shutil.copy2(source_root / source_name, output_root / output_name)
        signed.append(output_name)
    instructions = f"""# Task 11A.3 Reviewer B independent review

Open `review.html`. For every audit ID, inspect the original image on the left and the red mask overlay on the right.

Record only your own judgment in `reviewer_b.csv`, with reviewer_id `{reviewer_id}`. For each criterion use PASS, FAIL, or UNCERTAIN:

- real_photo: real field or natural photo
- lesion_visible: visible plant lesion or disease symptom
- no_visible_pest: no recognizable insect or pest body
- no_dominant_text: no prominent text, logo, or watermark
- no_collage: one coherent photograph, not a collage or multi-panel graphic
- mask_valid: red region adequately marks the visible lesion

Any FAIL means REJECT. Use UNCERTAIN when genuinely unsure. Do not consult disease labels, filenames, source metadata, AI flags, or Reviewer A's decisions. Return the completed CSV without changing audit IDs or row order.
"""
    (output_root / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8", newline="\n")
    signed.append("INSTRUCTIONS.md")
    report = {
        "version": "task11a3-independent-reviewer-handoff-1",
        "state": "completed",
        "reviewer_id": reviewer_id,
        "audit_rows": len(audit_ids),
        "labels_exposed": False,
        "router_outputs_read": False,
        "reviewer_a_results_included": False,
        "ai_pretriage_included": False,
        "transfer_modes": sorted(transfer_modes),
        "decision": "PENDING_REVIEWER_B",
    }
    write_json_new(output_root / "handoff_report.json", report)
    signed.append("handoff_report.json")
    with (output_root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        for name in sorted(signed):
            handle.write(f"{sha256_file(output_root / name)}  {name}\n")
    forbidden = [path.name for path in output_root.iterdir() if "reviewer_a" in path.name.lower() or "ai_" in path.name.lower()]
    if forbidden:
        raise ValueError(f"forbidden information in reviewer handoff: {forbidden}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build_handoff(
        source_root=args.source_root,
        output_root=args.output_root,
        reviewer_id=args.reviewer_id,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
