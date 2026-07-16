"""Fail-closed metadata protocol for the Task 10B v2 linear probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


PASSED_PROTOCOL = "PASSED_PROTOCOL"
BLOCKED_CLASS_QUOTA = "BLOCKED_CLASS_QUOTA"
DEFAULT_SPLIT_QUOTAS = {"train": 12, "val": 3, "dev": 5}
DEFAULT_BAND_QUOTAS = {"head": 6, "medium": 5, "tail": 5}
FROZEN_SELECTED_CLASS_IDS = (9, 10, 16, 17, 22, 24, 45, 50, 64, 68, 71, 82, 83, 87, 99, 101)


def _stable_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"task10b-v2|{namespace}|{value}".encode("utf-8")
    ).hexdigest()


def _provenance_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = str(row.get("source_image_id", ""))
        value = {
            "source_image_sha256": str(row.get("source_image_sha256", "")),
            "near_duplicate_component_id": str(
                row.get("near_duplicate_component_id", "")
            ),
        }
        if not image_id or len(value["source_image_sha256"]) != 64:
            raise ValueError("invalid source provenance row")
        if not value["near_duplicate_component_id"]:
            raise ValueError("missing near-duplicate component")
        if image_id in index and index[image_id] != value:
            raise ValueError(f"inconsistent source provenance: {image_id}")
        index[image_id] = value
    return index


def _class_catalog(
    selected_classes: Iterable[dict[str, Any]],
    class_bands: Mapping[str, str],
) -> dict[int, str]:
    catalog: dict[int, str] = {}
    for row in selected_classes:
        class_id = int(row["class_id"])
        declared = str(row["band"])
        mapped = str(class_bands.get(str(class_id), ""))
        if not mapped or declared != mapped:
            raise ValueError(f"class band mismatch: {class_id}")
        if class_id in catalog:
            raise ValueError(f"duplicate selected class: {class_id}")
        catalog[class_id] = declared
    if not catalog:
        raise ValueError("empty frozen class catalog")
    return catalog


def _overlap_count(manifest: list[dict[str, Any]], key: str) -> int:
    values: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        values[str(row["split"])].add(str(row[key]))
    splits = sorted(values)
    overlap = set()
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap.update(values[left] & values[right])
    return len(overlap)


def build_protocol(
    *,
    positive_rows: Iterable[dict[str, Any]],
    provenance_rows: Iterable[dict[str, Any]],
    used_sha256: set[str],
    locked_ids: set[str],
    locked_sha256: set[str],
    selected_classes: Iterable[dict[str, Any]],
    class_bands: Mapping[str, str],
    split_quotas: Mapping[str, int],
    band_quotas: Mapping[str, int],
) -> dict[str, Any]:
    """Build a deterministic protocol without opening images or loading a model."""
    quotas = {str(key): int(value) for key, value in split_quotas.items()}
    required_splits = {"train", "val", "dev"}
    if set(quotas) != required_splits or any(value <= 0 for value in quotas.values()):
        raise ValueError("split quotas must define positive train/val/dev counts")
    required_per_class = sum(quotas.values())
    required_bands = {str(key): int(value) for key, value in band_quotas.items()}
    if set(required_bands) != {"head", "medium", "tail"}:
        raise ValueError("band quotas must define head/medium/tail")

    catalog = _class_catalog(selected_classes, class_bands)
    provenance = _provenance_index(provenance_rows)
    raw_counts = Counter()
    excluded = Counter()
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in positive_rows:
        if str(row.get("source_split", "")) != "trainval":
            excluded["non_trainval"] += 1
            continue
        image_id = str(row.get("metadata", {}).get("image_id", ""))
        class_id = int(row.get("target", {}).get("diagnosis", {}).get("pest_id", -1))
        if class_id not in catalog:
            continue
        raw_counts[class_id] += 1
        if not image_id or image_id in seen_ids:
            raise ValueError(f"duplicate or empty positive source ID: {image_id}")
        seen_ids.add(image_id)
        if image_id not in provenance:
            raise ValueError(f"missing source provenance: {image_id}")
        private = provenance[image_id]
        digest = private["source_image_sha256"]
        if image_id in locked_ids or digest in locked_sha256:
            excluded["task8_locked_boundary"] += 1
            continue
        if digest in used_sha256:
            excluded["task9_used_sha256"] += 1
            continue
        candidates.append(
            {
                "image": str(row.get("image", "")),
                "source_image_id": image_id,
                "source_image_sha256": digest,
                "near_duplicate_component_id": private[
                    "near_duplicate_component_id"
                ],
                "class_id": class_id,
                "class_band": catalog[class_id],
            }
        )

    component_classes: dict[str, set[int]] = defaultdict(set)
    component_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        component = row["near_duplicate_component_id"]
        component_classes[component].add(int(row["class_id"]))
        component_rows[component].append(row)
    multiclass_components = {
        component for component, classes in component_classes.items() if len(classes) != 1
    }

    representatives: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for component, rows in component_rows.items():
        if component in multiclass_components:
            continue
        class_id = int(rows[0]["class_id"])
        representative = min(
            rows,
            key=lambda row: (
                _stable_rank("representative", row["source_image_id"]),
                row["source_image_id"],
            ),
        )
        representatives[class_id].append(representative)

    eligible_by_band = Counter()
    eligible_classes: dict[str, list[int]] = defaultdict(list)
    availability = {}
    for class_id, band in catalog.items():
        count = len(representatives[class_id])
        availability[str(class_id)] = {
            "band": band,
            "raw_trainval": raw_counts[class_id],
            "eligible_components": count,
        }
        if count >= required_per_class:
            eligible_by_band[band] += 1
            eligible_classes[band].append(class_id)

    report = {
        "version": "task10b-v2-protocol-1",
        "model_loaded": False,
        "required_per_class": required_per_class,
        "required_by_band": dict(sorted(required_bands.items())),
        "eligible_by_band": {
            band: int(eligible_by_band[band]) for band in ("head", "medium", "tail")
        },
        "class_availability": availability,
        "excluded": dict(sorted(excluded.items())),
        "excluded_multiclass_components": len(multiclass_components),
        "task8_locked_content_read": False,
    }
    config = {
        "version": "task10b-v2-config-1",
        "split_quotas": dict(sorted(quotas.items())),
        "band_quotas": dict(sorted(required_bands.items())),
        "class_selection": "raw_trainval_desc_then_class_id_asc",
        "component_policy": "one_representative_from_mono_class_component",
    }

    infeasible = any(
        eligible_by_band[band] < required for band, required in required_bands.items()
    )
    if infeasible:
        report["status"] = BLOCKED_CLASS_QUOTA
        return {
            "status": BLOCKED_CLASS_QUOTA,
            "manifest": [],
            "selected_classes": [],
            "report": report,
            "config": config,
        }

    chosen: list[int] = []
    for band in ("head", "medium", "tail"):
        ranked = sorted(
            eligible_classes[band],
            key=lambda class_id: (-raw_counts[class_id], class_id),
        )
        chosen.extend(ranked[: required_bands[band]])

    manifest: list[dict[str, Any]] = []
    selected_output = []
    for class_id in chosen:
        band = catalog[class_id]
        ranked = sorted(
            representatives[class_id],
            key=lambda row: (
                _stable_rank(f"component:{class_id}", row["near_duplicate_component_id"]),
                row["near_duplicate_component_id"],
            ),
        )[:required_per_class]
        selected_output.append(
            {
                "class_id": class_id,
                "class_band": band,
                "raw_trainval": raw_counts[class_id],
                "eligible_components": len(representatives[class_id]),
            }
        )
        offset = 0
        for split in ("train", "val", "dev"):
            for source in ranked[offset : offset + quotas[split]]:
                identifier = _stable_rank(
                    "manifest",
                    f"{split}|{class_id}|{source['source_image_id']}",
                )[:32]
                manifest.append(
                    {
                        "id": identifier,
                        "split": split,
                        **source,
                    }
                )
            offset += quotas[split]

    manifest.sort(key=lambda row: (row["split"], int(row["class_id"]), row["id"]))
    rows_by_split = Counter(str(row["split"]) for row in manifest)
    overlap = {
        "near_duplicate_component": _overlap_count(
            manifest, "near_duplicate_component_id"
        ),
        "source_image_sha256": _overlap_count(manifest, "source_image_sha256"),
    }
    expected_rows = sum(required_bands.values()) * required_per_class
    if len(manifest) != expected_rows or any(overlap.values()):
        raise ValueError("internal protocol cardinality or overlap failure")

    report.update(
        {
            "status": PASSED_PROTOCOL,
            "selected_class_count": len(chosen),
            "rows_by_split": dict(sorted(rows_by_split.items())),
            "row_count": len(manifest),
            "overlap": overlap,
        }
    )
    return {
        "status": PASSED_PROTOCOL,
        "manifest": manifest,
        "selected_classes": sorted(selected_output, key=lambda row: int(row["class_id"])),
        "report": report,
        "config": config,
    }


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def write_protocol(
    result: dict[str, Any], output_root: Path, input_paths: Iterable[Path]
) -> None:
    destination = Path(output_root)
    ensure_new_directory(destination)
    inputs = {str(path): sha256_file(Path(path)) for path in input_paths}
    write_json_new(destination / "input_sha256.json", inputs)
    write_json_new(destination / "config.snapshot.json", result["config"])
    if result["status"] != PASSED_PROTOCOL:
        write_json_new(destination / "block_report.json", result["report"])
        write_json_new(
            destination / "status.json",
            {"state": "blocked", "stage": "protocol", "reason": result["status"]},
        )
        return

    _write_jsonl_new(destination / "manifest.jsonl", result["manifest"])
    write_json_new(destination / "selected_classes.json", result["selected_classes"])
    write_json_new(destination / "protocol_report.json", result["report"])
    write_json_new(destination / "status.json", {"state": "completed", "stage": "protocol"})
    signed = [
        "manifest.jsonl",
        "selected_classes.json",
        "protocol_report.json",
        "input_sha256.json",
        "config.snapshot.json",
    ]
    with (destination / "completion.sha256").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_protocol_from_paths(
    *,
    positive_paths: Iterable[Path],
    provenance_path: Path,
    used_provenance_path: Path,
    locked_exclusion_path: Path,
    selected_classes_path: Path,
    class_bands_path: Path,
    output_root: Path,
    expected_selected_class_ids: Iterable[int] = FROZEN_SELECTED_CLASS_IDS,
) -> dict[str, Any]:
    positive_files = [Path(path) for path in positive_paths]
    provenance_file = Path(provenance_path)
    used_file = Path(used_provenance_path)
    locked_file = Path(locked_exclusion_path)
    selected_file = Path(selected_classes_path)
    bands_file = Path(class_bands_path)
    input_paths = [
        *positive_files,
        provenance_file,
        used_file,
        locked_file,
        selected_file,
        bands_file,
    ]
    if not positive_files or any(not path.is_file() for path in input_paths):
        raise ValueError("Task 10B protocol input is missing")

    positives = [row for path in positive_files for row in _read_jsonl(path)]
    used_rows = _read_jsonl(used_file)
    used_sha256 = {
        str(row["source_image_sha256"])
        for row in used_rows
        if row.get("source_image_sha256")
    }
    locked = json.loads(locked_file.read_text(encoding="utf-8"))
    selected_classes = json.loads(selected_file.read_text(encoding="utf-8"))
    class_bands = json.loads(bands_file.read_text(encoding="utf-8"))
    result = build_protocol(
        positive_rows=positives,
        provenance_rows=_read_jsonl(provenance_file),
        used_sha256=used_sha256,
        locked_ids={str(value) for value in locked.get("image_ids", [])},
        locked_sha256={str(value) for value in locked.get("image_sha256", [])},
        selected_classes=selected_classes,
        class_bands=class_bands,
        split_quotas=DEFAULT_SPLIT_QUOTAS,
        band_quotas=DEFAULT_BAND_QUOTAS,
    )
    if result["status"] == PASSED_PROTOCOL:
        observed = sorted(int(row["class_id"]) for row in result["selected_classes"])
        expected = sorted(int(value) for value in expected_selected_class_ids)
        if observed != expected:
            raise ValueError(
                f"frozen selected class mismatch: observed={observed}, expected={expected}"
            )
    write_protocol(result, Path(output_root), input_paths)
    return result["report"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen Task 10B v2 protocol")
    parser.add_argument("--positive-path", action="append", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--used-provenance", type=Path, required=True)
    parser.add_argument("--locked-exclusion", type=Path, required=True)
    parser.add_argument("--selected-classes", type=Path, required=True)
    parser.add_argument("--class-bands", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_protocol_from_paths(
        positive_paths=arguments.positive_path,
        provenance_path=arguments.provenance,
        used_provenance_path=arguments.used_provenance,
        locked_exclusion_path=arguments.locked_exclusion,
        selected_classes_path=arguments.selected_classes,
        class_bands_path=arguments.class_bands,
        output_root=arguments.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
