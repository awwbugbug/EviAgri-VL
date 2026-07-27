"""Build the fresh, family-safe Task14A annotation-oracle protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from task10b_protocol import FROZEN_SELECTED_CLASS_IDS


SEEDS = (17, 29, 43)
POSITIVE_QUOTAS = {"probe_train": 4, "probe_val": 1, "probe_test": 2}
NULL_COUNT = 32
PASSED = "PASSED_PROTOCOL"
BLOCKED = "BLOCKED_PROTOCOL"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"task14a-oracle-v1|{namespace}|{value}".encode("utf-8")
    ).hexdigest()


def _provenance_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = str(row.get("source_image_id", ""))
        value = {
            "source_image_sha256": str(row.get("source_image_sha256", "")),
            "near_duplicate_component_id": str(
                row.get("near_duplicate_component_id", "")
            ),
        }
        if not image_id or len(value["source_image_sha256"]) != 64:
            raise ValueError("invalid positive provenance row")
        if not value["near_duplicate_component_id"]:
            raise ValueError("positive provenance lacks near-duplicate component")
        if image_id in result and result[image_id] != value:
            raise ValueError(f"inconsistent positive provenance: {image_id}")
        result[image_id] = value
    return result


def _used_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        for key in ("id", "source_image_id"):
            value = str(row.get(key, ""))
            if value:
                values.add(value)
    return values


def _valid_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("positive row has invalid evidence bbox")
    box = [float(item) for item in value]
    if box[2] <= box[0] or box[3] <= box[1] or min(box) < 0:
        raise ValueError("positive row has degenerate evidence bbox")
    return box


def select_protocol(
    *,
    positive_rows: Iterable[dict[str, Any]],
    provenance_rows: Iterable[dict[str, Any]],
    selected_classes: Iterable[dict[str, Any]],
    prior_positive_rows: Iterable[dict[str, Any]],
    plantseg_rows: Iterable[dict[str, Any]],
    prior_used_rows: Iterable[dict[str, Any]],
    locked_ids: set[str],
    locked_sha256: set[str],
) -> dict[str, Any]:
    """Select rows without opening an image or loading a model."""
    class_bands = {
        int(row["class_id"]): str(row.get("class_band", row.get("band", "")))
        for row in selected_classes
    }
    expected_classes = set(FROZEN_SELECTED_CLASS_IDS)
    if set(class_bands) != expected_classes or any(
        band not in {"head", "medium", "tail"} for band in class_bands.values()
    ):
        raise ValueError("Task14A frozen 16-class catalog mismatch")

    prior_positive = list(prior_positive_rows)
    excluded_components = {
        str(row.get("near_duplicate_component_id", "")) for row in prior_positive
    } - {""}
    excluded_sha256 = {
        str(row.get("source_image_sha256", "")) for row in prior_positive
    } - {""}
    excluded_source_ids = {
        str(row.get("source_image_id", "")) for row in prior_positive
    } - {""}
    if not excluded_components or not excluded_sha256:
        raise ValueError("prior positive boundary lacks SHA256 or components")

    provenance = _provenance_index(provenance_rows)
    exclusions = Counter()
    seen_source_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for row in positive_rows:
        if str(row.get("source_split", "")) != "trainval":
            continue
        class_id = int(row.get("target", {}).get("diagnosis", {}).get("pest_id", -1))
        if class_id not in class_bands:
            continue
        source_id = str(row.get("metadata", {}).get("image_id", ""))
        if not source_id or source_id in seen_source_ids:
            raise ValueError(f"duplicate or empty positive source ID: {source_id}")
        seen_source_ids.add(source_id)
        if source_id not in provenance:
            raise ValueError(f"missing positive provenance: {source_id}")
        private = provenance[source_id]
        digest = private["source_image_sha256"]
        component = private["near_duplicate_component_id"]
        if source_id in locked_ids or digest in locked_sha256:
            exclusions["task8_hashed_boundary"] += 1
            continue
        if (
            source_id in excluded_source_ids
            or digest in excluded_sha256
            or component in excluded_components
        ):
            exclusions["prior_task10_family_boundary"] += 1
            continue
        image = str(row.get("image", ""))
        if not image:
            raise ValueError("positive row lacks image path")
        candidates.append(
            {
                "id": source_id,
                "source_image_id": source_id,
                "source_image_sha256": digest,
                "near_duplicate_component_id": component,
                "image": image,
                "class_id": class_id,
                "class_band": class_bands[class_id],
                "evidence_bbox": _valid_bbox(
                    row.get("target", {}).get("evidence_bbox")
                ),
                "target_type": "positive",
                "region_source": "ip102_gt_bbox",
            }
        )

    component_classes: dict[str, set[int]] = defaultdict(set)
    component_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        component = str(row["near_duplicate_component_id"])
        component_classes[component].add(int(row["class_id"]))
        component_rows[component].append(row)
    multiclass = {
        component for component, classes in component_classes.items() if len(classes) != 1
    }
    representatives: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for component, rows in component_rows.items():
        if component in multiclass:
            continue
        representative = min(
            rows,
            key=lambda row: (
                _stable_rank("representative", str(row["source_image_id"])),
                str(row["source_image_id"]),
            ),
        )
        representatives[int(representative["class_id"])].append(representative)

    required_per_class = sum(POSITIVE_QUOTAS.values())
    availability = {
        str(class_id): len(representatives[class_id])
        for class_id in sorted(expected_classes)
    }
    deficient = {
        class_id: count
        for class_id, count in availability.items()
        if count < required_per_class
    }
    if deficient:
        return {
            "status": BLOCKED,
            "manifest": [],
            "report": {
                "status": BLOCKED,
                "reason": "insufficient_fresh_positive_components",
                "availability": availability,
                "deficient": deficient,
                "task8_locked_content_read": False,
                "model_loaded": False,
            },
        }

    manifest: list[dict[str, Any]] = []
    for class_id in sorted(expected_classes):
        ranked = sorted(
            representatives[class_id],
            key=lambda row: (
                _stable_rank(
                    f"component:{class_id}", str(row["near_duplicate_component_id"])
                ),
                str(row["near_duplicate_component_id"]),
            ),
        )[:required_per_class]
        offset = 0
        for split in ("probe_train", "probe_val", "probe_test"):
            count = POSITIVE_QUOTAS[split]
            manifest.extend({**row, "probe_split": split} for row in ranked[offset : offset + count])
            offset += count

    used_ids = _used_ids(prior_used_rows)
    plantseg_candidates = []
    for row in plantseg_rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in used_ids:
            continue
        image_sha = str(row.get("image_sha256", ""))
        mask_sha = str(row.get("mask_sha256", ""))
        if len(image_sha) != 64 or len(mask_sha) != 64:
            raise ValueError("PlantSeg row lacks file hashes")
        ratio = float(row.get("mask_ratio", -1.0))
        if not (0.0 < ratio <= 1.0):
            raise ValueError("PlantSeg row has invalid mask ratio")
        plantseg_candidates.append(
            {
                "id": identifier,
                "image": str(row.get("image", "")),
                "source_image_sha256": image_sha,
                "mask": str(row.get("mask", "")),
                "mask_sha256": mask_sha,
                "mask_ratio": ratio,
                "plant": str(row.get("plant", "")),
                "disease": str(row.get("disease", "")),
                "class_id": None,
                "class_band": None,
                "probe_split": "null_test",
                "target_type": "real_null",
                "region_source": "plantseg_damage_mask",
            }
        )
    plantseg_candidates.sort(key=lambda row: (float(row["mask_ratio"]), str(row["id"])))
    if len(plantseg_candidates) < NULL_COUNT:
        return {
            "status": BLOCKED,
            "manifest": [],
            "report": {
                "status": BLOCKED,
                "reason": "insufficient_fresh_plantseg",
                "available": len(plantseg_candidates),
                "task8_locked_content_read": False,
                "model_loaded": False,
            },
        }
    indices = [int(index * (len(plantseg_candidates) - 1) / (NULL_COUNT - 1)) for index in range(NULL_COUNT)]
    if len(set(indices)) != NULL_COUNT:
        raise ValueError("PlantSeg quantile selection is not unique")
    manifest.extend(plantseg_candidates[index] for index in indices)
    manifest.sort(
        key=lambda row: (
            str(row["probe_split"]),
            -1 if row["class_id"] is None else int(row["class_id"]),
            str(row["id"]),
        )
    )

    counts = Counter(str(row["probe_split"]) for row in manifest)
    expected_counts = {"probe_train": 64, "probe_val": 16, "probe_test": 32, "null_test": 32}
    components_by_split = {
        split: {
            str(row["near_duplicate_component_id"])
            for row in manifest
            if row["target_type"] == "positive" and row["probe_split"] == split
        }
        for split in POSITIVE_QUOTAS
    }
    overlap = set()
    split_names = list(POSITIVE_QUOTAS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap.update(components_by_split[left] & components_by_split[right])
    identifiers = [str(row["id"]) for row in manifest]
    if dict(counts) != expected_counts or len(set(identifiers)) != len(identifiers) or overlap:
        raise ValueError("Task14A cardinality, ID, or family-isolation failure")
    if any(
        str(row.get("near_duplicate_component_id", "")) in excluded_components
        for row in manifest
        if row["target_type"] == "positive"
    ):
        raise ValueError("Task14A reused a Task10B near-duplicate component")

    report = {
        "version": "task14a-oracle-protocol-report-1",
        "status": PASSED,
        "row_count": len(manifest),
        "rows_by_split": expected_counts,
        "positive_class_count": len(expected_classes),
        "fresh_positive_components_by_class": availability,
        "eligible_fresh_plantseg": len(plantseg_candidates),
        "selected_plantseg_mask_ratio": {
            "minimum": min(float(row["mask_ratio"]) for row in manifest if row["target_type"] == "real_null"),
            "maximum": max(float(row["mask_ratio"]) for row in manifest if row["target_type"] == "real_null"),
        },
        "excluded": dict(sorted(exclusions.items())),
        "excluded_task10_component_count": len(excluded_components),
        "excluded_multiclass_component_count": len(multiclass),
        "prior_used_id_count": len(used_ids),
        "cross_split_component_overlap": len(overlap),
        "model_loaded": False,
        "task8_locked_content_read": False,
    }
    return {"status": PASSED, "manifest": manifest, "report": report}


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def build_from_paths(
    *,
    positive_paths: Iterable[Path],
    provenance_path: Path,
    selected_classes_path: Path,
    prior_positive_path: Path,
    plantseg_path: Path,
    prior_used_paths: Iterable[Path],
    locked_exclusion_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    positive_files = [Path(path) for path in positive_paths]
    used_files = [Path(path) for path in prior_used_paths]
    input_paths = [
        *positive_files,
        Path(provenance_path),
        Path(selected_classes_path),
        Path(prior_positive_path),
        Path(plantseg_path),
        *used_files,
        Path(locked_exclusion_path),
    ]
    if not positive_files or not used_files or any(not path.is_file() for path in input_paths):
        raise ValueError("Task14A protocol input is missing")
    root = Path(output_root)
    ensure_new_directory(root)
    (root / "status.json").write_text('{"state":"running"}\n', encoding="utf-8")
    try:
        locked = json.loads(Path(locked_exclusion_path).read_text(encoding="utf-8"))
        result = select_protocol(
            positive_rows=[row for path in positive_files for row in read_jsonl(path)],
            provenance_rows=read_jsonl(Path(provenance_path)),
            selected_classes=json.loads(Path(selected_classes_path).read_text(encoding="utf-8")),
            prior_positive_rows=read_jsonl(Path(prior_positive_path)),
            plantseg_rows=read_jsonl(Path(plantseg_path)),
            prior_used_rows=[row for path in used_files for row in read_jsonl(path)],
            locked_ids={str(value) for value in locked.get("image_ids", [])},
            locked_sha256={str(value) for value in locked.get("image_sha256", [])},
        )
        write_json_new(root / "input_sha256.json", {str(path): sha256_file(path) for path in input_paths})
        write_json_new(
            root / "config.snapshot.json",
            {
                "version": "task14a-oracle-protocol-config-1",
                "positive_quotas_per_class": POSITIVE_QUOTAS,
                "plantseg_null_count": NULL_COUNT,
                "seeds": list(SEEDS),
                "selection": "fresh_component_rank_and_mask_ratio_quantiles",
            },
        )
        if result["status"] != PASSED:
            write_json_new(root / "block_report.json", result["report"])
            (root / "status.json").write_text('{"state":"blocked"}\n', encoding="utf-8")
            return result["report"]
        _write_jsonl_new(root / "manifest.jsonl", result["manifest"])
        write_json_new(root / "protocol_report.json", result["report"])
        signed = ["manifest.jsonl", "protocol_report.json", "input_sha256.json", "config.snapshot.json"]
        with (root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed:
                handle.write(f"{sha256_file(root / name)}  {name}\n")
        (root / "status.json").write_text('{"state":"completed"}\n', encoding="utf-8")
        return result["report"]
    except Exception as exc:
        write_json_new(root / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        (root / "status.json").write_text('{"state":"failed"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-path", action="append", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--selected-classes", type=Path, required=True)
    parser.add_argument("--prior-positive", type=Path, required=True)
    parser.add_argument("--plantseg", type=Path, required=True)
    parser.add_argument("--prior-used", action="append", type=Path, required=True)
    parser.add_argument("--locked-exclusion", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_from_paths(
        positive_paths=args.positive_path,
        provenance_path=args.provenance,
        selected_classes_path=args.selected_classes,
        prior_positive_path=args.prior_positive,
        plantseg_path=args.plantseg,
        prior_used_paths=args.prior_used,
        locked_exclusion_path=args.locked_exclusion,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
