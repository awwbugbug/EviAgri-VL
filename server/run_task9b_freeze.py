"""Server entry point for building and freezing the Task 9B v2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_task9b_v2 import build_dataset
from task9b_split import assign_components, locked_exclusion
from validate_task9b_freeze import validate_freeze


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(row: dict[str, Any]) -> dict[str, Any]:
    image = Path(row["image"])
    diagnosis = row["target"]["diagnosis"]
    valid_objects = row["metadata"]["all_valid_objects"]
    return {
        "image_id": str(row["metadata"]["image_id"]),
        "image_sha256": _sha256(image),
        "image_path": str(image),
        "class_id": int(diagnosis["pest_id"]),
        "class_name": str(diagnosis["pest_name"]),
        "bbox": list(row["target"]["evidence_bbox"]),
        "present_class_ids": sorted({int(obj["pest_id"]) for obj in valid_objects}),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(
    source_dir: Path,
    locked_manifest_path: Path,
    reviewed_pairs_path: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing existing Task 9B destination: {output_root}")
    source_paths = sorted(source_dir.glob("*_evidence_positive.jsonl"))
    if len(source_paths) != 3:
        raise ValueError(f"expected three positive source manifests, found {len(source_paths)}")
    source_rows = [row for path in source_paths for row in _read_jsonl(path)]
    records = [source_record(row) for row in source_rows]
    if len({row["image_id"] for row in records}) != len(records):
        raise ValueError("source positive records are not image-unique")

    locked_manifest = _read_jsonl(locked_manifest_path)
    exclusion = locked_exclusion(locked_manifest, _sha256(locked_manifest_path))
    review = json.loads(reviewed_pairs_path.read_text(encoding="utf-8"))
    reviewed_pairs = review.get("reviewed_candidates", [])
    if len(reviewed_pairs) != int(review.get("reviewed_count", -1)):
        raise ValueError("near-duplicate review is incomplete")
    split = assign_components(records, reviewed_pairs, exclusion, seed)

    build = build_dataset(
        records,
        split["assignment"],
        output_root,
        seed=seed,
        component_by_image_id=split["component_by_image_id"],
    )
    private = output_root / "private"
    _write_json(private / "locked_exclusion.json", exclusion)
    _write_json(
        private / "split_manifest.json",
        {
            "seed": seed,
            "split_image_counts": split["split_image_counts"],
            "excluded_components": split["excluded_components"],
            "assignment": split["assignment"],
            "component_by_image_id": split["component_by_image_id"],
        },
    )
    _write_json(
        private / "source_inputs.json",
        {
            "positive_manifests": {
                path.name: _sha256(path) for path in source_paths
            },
            "locked_manifest_sha256": _sha256(locked_manifest_path),
            "reviewed_pairs_sha256": _sha256(reviewed_pairs_path),
            "reviewed_pair_count": len(reviewed_pairs),
            "high_confidence_pair_count": sum(bool(row.get("high_confidence")) for row in reviewed_pairs),
        },
    )
    report = validate_freeze(output_root, locked_exclusion=exclusion)
    return {"build": build["summary"], "freeze": report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    arguments = parser.parse_args()
    result = run(
        arguments.source_dir,
        arguments.locked_manifest,
        arguments.reviewed_pairs,
        arguments.output_root,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
