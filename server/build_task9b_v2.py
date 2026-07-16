"""Build the sanitized, three-row-family Task 9B v2 protocol dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from task9b_protocol import (
    DEV_TEMPLATE_IDS,
    TRAIN_TEMPLATE_IDS,
    SYSTEM_PROMPT,
    build_prompt,
    build_target,
    length_bucket_for_family,
    opaque_id,
    serialize_target,
)
from task9b_transforms import apply_transform, transform_kind_for_index


ROLES = ("positive", "semantic_negative", "visual_counterfactual")


def _jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_opaque(image: Image.Image, image_dir: Path, *, use_jpeg: bool = False) -> tuple[str, str]:
    # Encode first, then name by the bytes actually supplied to the model.
    suffix = ".jpg" if use_jpeg else ".png"
    staging = image_dir / f".staging{suffix}"
    if use_jpeg:
        image.convert("RGB").save(staging, format="JPEG", quality=95, subsampling=0)
    else:
        image.convert("RGB").save(staging, format="PNG", optimize=False)
    digest = _sha256(staging)
    destination = image_dir / f"{digest}{suffix}"
    if destination.exists():
        staging.unlink()
    else:
        staging.replace(destination)
    return f"images/{destination.name}", digest


def _materialize_original(source: Path, image_dir: Path) -> tuple[str, str, bool]:
    digest = _sha256(source)
    use_jpeg = source.suffix.lower() in {".jpg", ".jpeg"}
    suffix = ".jpg" if use_jpeg else ".png"
    destination = image_dir / f"{digest}{suffix}"
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return f"images/{destination.name}", digest, use_jpeg


def _select_absent_class(
    family_id: str,
    present_class_ids: set[int],
    class_catalog: Mapping[int, str],
    seed: int,
) -> tuple[int, str]:
    choices = sorted(class_id for class_id in class_catalog if class_id not in present_class_ids)
    if not choices:
        raise ValueError(f"no absent semantic query exists for {family_id}")
    digest = hashlib.sha256(f"task9b-neg:{seed}:{family_id}".encode("utf-8")).digest()
    class_id = choices[int.from_bytes(digest[:8], "big") % len(choices)]
    return class_id, class_catalog[class_id]


def _messages(image_ref: str, prompt: str, target_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]


def build_dataset(
    records: Iterable[dict[str, Any]],
    assignment: Mapping[str, str],
    output_root: str | Path,
    *,
    seed: int,
    component_by_image_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build immutable model JSONL and a private provenance sidecar.

    ``records`` must be clean positive detection records.  No source path or
    family role is copied to model-visible JSONL.
    """
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing existing freeze destination: {output_root}")
    output_root.mkdir(parents=True)
    image_dir = output_root / "images"
    image_dir.mkdir()

    rows = sorted((dict(row) for row in records), key=lambda row: str(row["image_id"]))
    if not rows:
        raise ValueError("no records supplied")
    class_catalog = {int(row["class_id"]): str(row["class_name"]) for row in rows}
    if len(class_catalog) < 2:
        raise ValueError("at least two classes are required for semantic negatives")

    visible_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "dev": []}
    provenance: list[dict[str, Any]] = []
    split_family_index = Counter()

    for row in rows:
        source_id = str(row["image_id"])
        split = str(assignment.get(source_id, "excluded"))
        if split == "excluded":
            continue
        if split not in visible_by_split:
            raise ValueError(f"unsupported split {split!r} for {source_id}")
        family_id = hashlib.sha256(f"task9b-family:{seed}:{source_id}".encode("utf-8")).hexdigest()[:32]
        surface = "dev" if split == "dev" else "train"
        templates = DEV_TEMPLATE_IDS if surface == "dev" else TRAIN_TEMPLATE_IDS
        family_index = split_family_index[split]
        split_family_index[split] += 1
        template_id = templates[family_index % len(templates)]
        length_bucket = length_bucket_for_family(family_id)
        transform_kind = transform_kind_for_index(family_index, surface)

        image_path = Path(row["image_path"])
        with Image.open(image_path) as loaded:
            source_image = loaded.convert("RGB")
        original_ref, original_derived_sha, source_is_jpeg = _materialize_original(image_path, image_dir)
        transformed = apply_transform(surface, transform_kind, source_image, row["bbox"], seed=seed + family_index)
        transformed_ref, transformed_sha = _save_opaque(
            transformed, image_dir, use_jpeg=source_is_jpeg
        )

        present_ids = {int(value) for value in row.get("present_class_ids", [row["class_id"]])}
        negative_id, negative_name = _select_absent_class(family_id, present_ids, class_catalog, seed)
        positive_id, positive_name = int(row["class_id"]), str(row["class_name"])
        specifications = (
            ("positive", original_ref, positive_id, positive_name, True, None),
            ("semantic_negative", original_ref, negative_id, negative_name, False, "real_null"),
            ("visual_counterfactual", transformed_ref, positive_id, positive_name, False, "synthetic_null"),
        )

        for role, image_ref, query_id, query_name, is_positive, null_source in specifications:
            target = build_target(
                is_positive,
                list(row["bbox"]) if is_positive else None,
                positive_id if is_positive else None,
                positive_name if is_positive else None,
            )
            target_text = serialize_target(target, length_bucket)
            identifier = opaque_id(seed, family_id, role)
            visible_by_split[split].append(
                {
                    "id": identifier,
                    "messages": _messages(image_ref, build_prompt(template_id, query_name), target_text),
                }
            )
            provenance.append(
                {
                    "id": identifier,
                    "family_id": family_id,
                    "role": role,
                    "evidence_present": is_positive,
                    "null_source": null_source,
                    "split": split,
                    "source_image_id": source_id,
                    "source_image_sha256": str(row["image_sha256"]),
                    "near_duplicate_component_id": str(
                        (component_by_image_id or {}).get(source_id, source_id)
                    ),
                    "derived_image_sha256": transformed_sha if role == "visual_counterfactual" else original_derived_sha,
                    "present_class_ids": sorted(present_ids),
                    "query_class_id": query_id,
                    "template_id": template_id,
                    "length_bucket": length_bucket,
                    "transform_id": transform_kind if role == "visual_counterfactual" else None,
                }
            )

    model_files = {
        "train": output_root / "model" / "train.jsonl",
        "val": output_root / "model" / "val.jsonl",
        "dev": output_root / "dev_audit" / "model.jsonl",
    }
    for split, path in model_files.items():
        _jsonl_write(path, visible_by_split[split])
    provenance_file = output_root / "private" / "provenance.jsonl"
    _jsonl_write(provenance_file, provenance)
    summary = {
        "version": "task9b-v2-protocol-1",
        "seed": seed,
        "families_by_split": dict(sorted(split_family_index.items())),
        "rows_by_split": {split: len(values) for split, values in visible_by_split.items()},
        "roles": dict(Counter(row["role"] for row in provenance)),
        "null_sources": {str(key): value for key, value in Counter(row["null_source"] for row in provenance).items()},
    }
    summary_path = output_root / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "output_root": str(output_root),
        "model_files": {key: str(path) for key, path in model_files.items()},
        "provenance_file": str(provenance_file),
        "summary_file": str(summary_path),
        "summary": summary,
    }
