"""Deterministic A/B/C preparation primitives for Task 9D."""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from task9b_protocol import build_prompt
from task9b_v21_exact_match import match_all_strata


QUERY_PATTERN = re.compile(r"queried pest '(.+?)'")
ROLES = ("positive", "semantic_negative", "visual_counterfactual")


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"task9d:{seed}:{value}".encode()).hexdigest()


def select_families(
    provenance: Iterable[dict[str, Any]], split: str, count: int, *, seed: int
) -> list[str]:
    positives = [row for row in provenance if row.get("split") == split and row.get("role") == "positive"]
    if count <= 0 or count > len(positives):
        raise ValueError(f"cannot select {count} families from {len(positives)} {split} positives")
    by_class: dict[int, list[str]] = defaultdict(list)
    for row in positives:
        by_class[int(row["query_class_id"])].append(str(row["family_id"]))
    for class_id in by_class:
        by_class[class_id].sort(key=lambda value: _rank(seed, value))
    selected: list[str] = []
    offset = 0
    classes = sorted(by_class)
    while len(selected) < count:
        progressed = False
        for class_id in classes:
            if offset < len(by_class[class_id]):
                selected.append(by_class[class_id][offset])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise AssertionError("class-stratified selector exhausted unexpectedly")
        offset += 1
    return sorted(selected)


def _prompt_query(model: dict[str, Any]) -> str:
    text = model["messages"][1]["content"][1]["text"]
    match = QUERY_PATTERN.search(text)
    if match is None:
        raise ValueError("cannot parse query name from frozen model prompt")
    return match.group(1)


def _set_prompt(model: dict[str, Any], template_id: str, query_name: str) -> None:
    model["messages"][1]["content"][1]["text"] = build_prompt(template_id, query_name)


def build_variant_schedule(
    model_by_id: Mapping[str, dict[str, Any]],
    provenance: Iterable[dict[str, Any]],
    family_ids: Iterable[str],
    variant: str,
    *,
    seed: int,
    target_rows: int,
    locked_image_ids: set[str] | None = None,
    locked_image_sha256: set[str] | None = None,
) -> dict[str, Any]:
    if variant not in {"A", "B", "C"}:
        raise ValueError(f"unknown variant: {variant}")
    families = sorted(set(map(str, family_ids)))
    selected = [copy.deepcopy(row) for row in provenance if str(row.get("family_id")) in set(families)]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_family[str(row["family_id"])].append(row)
    if set(by_family) != set(families):
        raise ValueError("selected family IDs are missing from provenance")
    expected = Counter({role: 1 for role in ROLES})
    if any(Counter(row["role"] for row in rows) != expected for rows in by_family.values()):
        raise ValueError("selected family does not have exact 1:1:1 source roles")
    locked_ids = locked_image_ids or set()
    locked_sha = locked_image_sha256 or set()
    for row in selected:
        if str(row.get("source_image_id")) in locked_ids or str(row.get("source_image_sha256")) in locked_sha:
            raise ValueError("locked source overlap in Task 9D family pool")

    class_names: dict[int, str] = {}
    matching_families = []
    for family_id in families:
        positive = next(row for row in by_family[family_id] if row["role"] == "positive")
        class_id = int(positive["query_class_id"])
        class_names[class_id] = _prompt_query(model_by_id[str(positive["id"])])
        template = str(positive["template_id"]) if variant == "C" else "train_neutral_0"
        matching_families.append({
            "family_id": family_id,
            "split": str(positive["split"]),
            "template_id": template,
            "positive_query_class_id": class_id,
            "present_class_ids": list(positive["present_class_ids"]),
        })
    matching = match_all_strata(matching_families)

    repaired_models: dict[str, dict[str, Any]] = {}
    repaired_rows: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_roles = {"positive", "semantic_negative"} if variant == "A" else set(ROLES)
    for family_id in families:
        family_template = next(
            row["template_id"] for row in matching_families if row["family_id"] == family_id
        )
        for row in by_family[family_id]:
            if row["role"] not in allowed_roles:
                continue
            row["template_id"] = family_template
            if row["role"] == "semantic_negative":
                row["query_class_id"] = int(matching["assignment"][family_id])
            query_id = int(row["query_class_id"])
            if query_id not in class_names:
                raise ValueError(f"missing class name for query {query_id}")
            model = copy.deepcopy(model_by_id[str(row["id"])])
            _set_prompt(model, family_template, class_names[query_id])
            repaired_models[str(row["id"])] = model
            repaired_rows[(family_id, str(row["role"]))] = row

    base_units = [
        [repaired_rows[(family_id, role)] for role in ROLES if (family_id, role) in repaired_rows]
        for family_id in families
    ]
    base_count = sum(map(len, base_units))
    if target_rows < base_count:
        raise ValueError("target_rows cannot be smaller than one complete family pass")
    schedule_rows = [row for unit in base_units for row in unit]
    remaining = target_rows - base_count
    unit_size = 2 if variant == "A" else 3
    if remaining % unit_size:
        raise ValueError("target_rows must add complete family role units")
    ranked_units = sorted(base_units, key=lambda rows: _rank(seed, str(rows[0]["family_id"])))
    cursor = 0
    while remaining:
        unit = ranked_units[cursor % len(ranked_units)]
        schedule_rows.extend(unit)
        remaining -= len(unit)
        cursor += 1
    if len(schedule_rows) != target_rows:
        raise AssertionError("training schedule length mismatch")

    schedule = [
        {
            "schedule_index": index,
            "id": str(row["id"]),
            "family_id": str(row["family_id"]),
            "role": str(row["role"]),
            "template_id": str(row["template_id"]),
            "model": repaired_models[str(row["id"])],
        }
        for index, row in enumerate(schedule_rows)
    ]
    return {
        "variant": variant,
        "family_ids": families,
        "schedule": schedule,
        "template_ids": [row["template_id"] for row in schedule],
        "role_counts": dict(Counter(row["role"] for row in schedule)),
        "matching_max_tv": max(item["total_variation"] for item in matching["strata"]),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_bands(provenance: list[dict[str, Any]]) -> dict[str, str]:
    counts = Counter(int(row["query_class_id"]) for row in provenance
                     if row["split"] == "train" and row["role"] == "positive")
    ordered = sorted(counts, key=lambda class_id: (-counts[class_id], class_id))
    result = {}
    for index, class_id in enumerate(ordered):
        fraction = index / max(1, len(ordered))
        result[str(class_id)] = "head" if fraction < 1 / 3 else "medium" if fraction < 2 / 3 else "tail"
    return result


def prepare_task9d(
    source_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 20260716,
    train_families: int = 512,
    val_families: int = 192,
    dev_families: int = 512,
    challenge_families: int = 128,
    train_rows: int = 1536,
) -> dict[str, Any]:
    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing existing Task 9D preparation output: {output_root}")
    freeze = json.loads((source_root / "freeze_report.json").read_text(encoding="utf-8"))
    if freeze.get("passed") is not True:
        raise ValueError("Task 9D source v2.1 freeze is not passed")
    paths = {
        "train": source_root / "model/train.jsonl",
        "val": source_root / "model/val.jsonl",
        "dev": source_root / "dev_audit/model.jsonl",
    }
    ordered_models = {split: _read_jsonl(path) for split, path in paths.items()}
    model_by_id = {str(row["id"]): row for rows in ordered_models.values() for row in rows}
    provenance = _read_jsonl(source_root / "private/provenance.jsonl")
    if set(model_by_id) != {str(row["id"]) for row in provenance}:
        raise ValueError("Task 9D source model/provenance IDs are not aligned")
    locked = json.loads((source_root / "private/locked_exclusion.json").read_text(encoding="utf-8"))
    locked_ids = set(map(str, locked.get("image_ids", [])))
    locked_sha = set(map(str, locked.get("image_sha256", [])))

    selections = {
        "train": select_families(provenance, "train", train_families, seed=seed),
        "val": select_families(provenance, "val", val_families, seed=seed),
        "dev": select_families(provenance, "dev", dev_families, seed=seed),
    }
    selections["challenge"] = sorted(
        selections["dev"], key=lambda family: _rank(seed + 1, family)
    )[:challenge_families]
    if challenge_families <= 0 or challenge_families > len(selections["dev"]):
        raise ValueError("challenge_families must be within selected dev families")

    schedules = {
        variant: build_variant_schedule(
            model_by_id, provenance, selections["train"], variant, seed=seed,
            target_rows=train_rows, locked_image_ids=locked_ids, locked_image_sha256=locked_sha,
        ) for variant in "ABC"
    }
    shared_val = build_variant_schedule(
        model_by_id, provenance, selections["val"], "B", seed=seed,
        target_rows=val_families * 3, locked_image_ids=locked_ids, locked_image_sha256=locked_sha,
    )
    selected_all = set(selections["train"] + selections["val"] + selections["dev"])
    selected_provenance = [row for row in provenance if str(row["family_id"]) in selected_all]
    for row in selected_provenance:
        if str(row.get("source_image_id")) in locked_ids or str(row.get("source_image_sha256")) in locked_sha:
            raise ValueError("locked source overlap in Task 9D selected provenance")
    component_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected_provenance:
        component_splits[str(row["near_duplicate_component_id"])].add(str(row["split"]))
    crossing = {key: sorted(value) for key, value in component_splits.items() if len(value) > 1}
    if crossing:
        raise ValueError(f"near-duplicate component crosses Task 9D splits: {list(crossing.items())[:3]}")

    # All feasibility and leakage checks above complete before the immutable output is created.
    output_root.mkdir(parents=True)
    for variant, schedule in schedules.items():
        _write_jsonl(output_root / f"variants/{variant}/train_schedule.jsonl", schedule["schedule"])
        _write_jsonl(output_root / f"variants/{variant}/val.jsonl", shared_val["schedule"])
        _write_json(output_root / f"variants/{variant}/schedule_report.json", {
            "variant": variant, "family_count": len(schedule["family_ids"]),
            "row_count": len(schedule["schedule"]), "role_counts": schedule["role_counts"],
            "matching_max_tv": schedule["matching_max_tv"],
        })
    dev_rows = [row for row in ordered_models["dev"]
                if str(next(item["family_id"] for item in provenance if item["id"] == row["id"])) in set(selections["dev"])]
    _write_jsonl(output_root / "dev_audit/source_model.jsonl", dev_rows)
    _write_jsonl(output_root / "private/provenance.jsonl", selected_provenance)
    _write_json(output_root / "private/selections.json", selections)
    _write_json(output_root / "private/locked_exclusion.json", locked)
    _write_json(output_root / "class_bands.json", _class_bands(provenance))

    selected_ids = {str(row["id"]) for row in selected_provenance}
    image_references = set()
    for identifier in selected_ids:
        reference = str(model_by_id[identifier]["messages"][1]["content"][0]["image"])
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source model image reference is not opaque and relative")
        image_references.add(relative)
    for relative in sorted(image_references):
        source = source_root / relative
        destination = output_root / relative
        if not source.is_file():
            raise ValueError(f"missing selected source image: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)

    report = {
        "version": "task9d-preparation-v1", "passed": True, "seed": seed,
        "source_root": str(source_root),
        "families": {key: len(value) for key, value in selections.items()},
        "train_rows": train_rows, "val_rows": val_families * 3,
        "same_family_pool_across_variants": True, "locked_overlap": 0,
        "near_duplicate_split_crossings": 0, "hardlinked_images": len(image_references),
        "task8_locked_set_read": False,
    }
    _write_json(output_root / "freeze_report.json", report)
    manifest_paths = sorted(path for path in output_root.rglob("*") if path.is_file())
    manifest = {str(path.relative_to(output_root)).replace("\\", "/"): _sha256(path)
                for path in manifest_paths}
    _write_json(output_root / "protocol_manifest.json", manifest)
    completion_paths = sorted(path for path in output_root.rglob("*")
                              if path.is_file() and path.name != "completion.sha256")
    (output_root / "completion.sha256").write_text(
        "".join(f"{_sha256(path)}  {str(path.relative_to(output_root)).replace(os.sep, '/')}\n"
                for path in completion_paths), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_task9d(args.source_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
