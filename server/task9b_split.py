from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable


SOURCE_ID = re.compile(r"^ip102det_(?:train|val|test)_(.+?)_positive$")


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def connected_components(
    image_ids: Iterable[str], reviewed_pairs: Iterable[dict[str, Any]]
) -> dict[str, str]:
    values = {str(value) for value in image_ids}
    union = _UnionFind(sorted(values))
    for pair in reviewed_pairs:
        if not bool(pair.get("high_confidence", True)):
            continue
        left, right = str(pair.get("left_id", "")), str(pair.get("right_id", ""))
        if left in values and right in values:
            union.union(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for value in sorted(values):
        groups[union.find(value)].append(value)
    component_ids = {
        root: hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:24]
        for root, members in groups.items()
    }
    return {value: component_ids[union.find(value)] for value in sorted(values)}


def locked_exclusion(
    family_manifest: Iterable[dict[str, Any]], source_manifest_sha256: str
) -> dict[str, Any]:
    image_ids: set[str] = set()
    image_sha256: set[str] = set()
    for row in family_manifest:
        match = SOURCE_ID.fullmatch(str(row.get("source_id", "")))
        if match is None:
            raise ValueError("locked family source_id is not a recognized positive source")
        image_ids.add(match.group(1))
        digest = str(row.get("source_image_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("locked family source image SHA256 is invalid")
        image_sha256.add(digest)
    if not image_ids:
        raise ValueError("locked family manifest is empty")
    return {
        "version": "task9b-blinded-locked-exclusion-1",
        "source_manifest_sha256": source_manifest_sha256,
        "image_ids": sorted(image_ids),
        "image_sha256": sorted(image_sha256),
    }


def _stable_rank(seed: int, component_id: str) -> str:
    return hashlib.sha256(f"task9b-split:{seed}:{component_id}".encode("utf-8")).hexdigest()


def assign_components(
    records: Iterable[dict[str, Any]],
    reviewed_pairs: Iterable[dict[str, Any]],
    exclusions: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        image_id = str(row.get("image_id", ""))
        if not image_id:
            raise ValueError("record image_id must be non-empty")
        if image_id in by_id:
            raise ValueError(f"duplicate image_id: {image_id}")
        by_id[image_id] = dict(row)
    components = connected_components(by_id, reviewed_pairs)
    component_members: dict[str, list[str]] = defaultdict(list)
    for image_id, component_id in components.items():
        component_members[component_id].append(image_id)
    excluded_ids = {str(value) for value in exclusions.get("image_ids", [])}
    excluded_sha = {str(value) for value in exclusions.get("image_sha256", [])}
    excluded_components = {
        component_id
        for component_id, members in component_members.items()
        if any(
            image_id in excluded_ids
            or str(by_id[image_id].get("image_sha256", "")) in excluded_sha
            for image_id in members
        )
    }
    strata: dict[int, list[str]] = defaultdict(list)
    for component_id, members in component_members.items():
        if component_id in excluded_components:
            continue
        class_ids = [int(by_id[image_id]["class_id"]) for image_id in members]
        strata[min(class_ids)].append(component_id)
    component_split: dict[str, str] = {
        component_id: "excluded" for component_id in excluded_components
    }
    for class_id in sorted(strata):
        ranked = sorted(strata[class_id], key=lambda value: _stable_rank(seed, value))
        for index, component_id in enumerate(ranked):
            slot = index % 10
            component_split[component_id] = "dev" if slot == 0 else "val" if slot == 1 else "train"
    assignment = {
        image_id: component_split[component_id]
        for image_id, component_id in sorted(components.items())
    }
    split_counts = defaultdict(int)
    for split in assignment.values():
        split_counts[split] += 1
    return {
        "seed": seed,
        "assignment": assignment,
        "component_by_image_id": components,
        "component_split": dict(sorted(component_split.items())),
        "split_image_counts": dict(sorted(split_counts.items())),
        "excluded_components": sorted(excluded_components),
    }
