"""Held-out, paired evaluation protocol for Task 9D."""

from __future__ import annotations

import hashlib
import io
import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFilter

from task9b_protocol import build_prompt


BLANK_RGB = (91, 107, 123)
BLUR_RADIUS_FRACTION = 0.16
SHUFFLE_GRID = 7
UNSEEN_TEMPLATE_IDS = ("task9d_unseen_neutral_alpha", "task9d_unseen_neutral_beta")
QUERY_PATTERN = re.compile(r"queried pest '(.+?)'")
_COMMON = (
    " Use pixels only. Return one JSON object with keys in this exact order: "
    "evidence_present, evidence_region, visible_attributes, diagnosis, reliability. "
    "When evidence is insufficient, use evidence_present=false, evidence_region=null, "
    "and diagnosis.status set to uncertain or abstain without a pest, species, or stage."
)


def _stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"task9d-eval:{key}".encode()).digest()[:8], "big")


def _shuffle_patches(image: Image.Image, key: str) -> Image.Image:
    source = image.convert("RGB")
    width, height = source.size
    cell_width, cell_height = width // SHUFFLE_GRID, height // SHUFFLE_GRID
    if cell_width < 1 or cell_height < 1:
        raise ValueError(f"image {source.size} is too small for {SHUFFLE_GRID}x{SHUFFLE_GRID} shuffle")
    boxes = [
        (column * cell_width, row * cell_height,
         (column + 1) * cell_width, (row + 1) * cell_height)
        for row in range(SHUFFLE_GRID) for column in range(SHUFFLE_GRID)
    ]
    order = list(range(len(boxes)))
    random.Random(_stable_seed(key)).shuffle(order)
    if order == list(range(len(boxes))):
        order = order[1:] + order[:1]
    result = source.copy()
    patches = [source.crop(box) for box in boxes]
    for destination, source_index in zip(boxes, order):
        result.paste(patches[source_index], destination[:2])
    return result


def apply_heldout_transform(image: Image.Image, condition: str, *, key: str) -> Image.Image:
    source = image.convert("RGB")
    if condition == "original":
        return source.copy()
    if condition == "blank":
        return Image.new("RGB", source.size, BLANK_RGB)
    if condition == "blur":
        radius = max(source.size) * BLUR_RADIUS_FRACTION
        return source.filter(ImageFilter.GaussianBlur(radius=radius))
    if condition == "shuffle":
        return _shuffle_patches(source, key)
    raise ValueError(f"unknown held-out condition: {condition}")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _write_content_addressed(image: Image.Image, output_dir: Path) -> tuple[Path, str]:
    payload = _png_bytes(image)
    digest = hashlib.sha256(payload).hexdigest()
    path = output_dir / f"{digest}.png"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-address collision at {path}")
    else:
        path.write_bytes(payload)
    return path, digest


def build_paired_conditions(
    families: Iterable[dict[str, Any]], output_dir: str | Path
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family in sorted(families, key=lambda row: str(row["family_id"])):
        family_id = str(family["family_id"])
        if not family_id or family_id in seen:
            raise ValueError("challenge family IDs must be unique and non-empty")
        seen.add(family_id)
        source_path = Path(str(family["image_path"]))
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")
        pair_id = hashlib.sha256(f"task9d-pair:{family_id}".encode()).hexdigest()[:24]
        for condition in ("original", "blank", "blur", "shuffle"):
            transformed = apply_heldout_transform(source, condition, key=family_id)
            path, digest = _write_content_addressed(transformed, output_dir)
            rows.append({
                **family,
                "pair_id": pair_id,
                "condition": condition,
                "image_path": str(path),
                "image_sha256": digest,
                "transform_protocol": None if condition == "original" else "task9d-heldout-v1",
            })
    return rows


def _query_name(model: dict[str, Any]) -> str:
    text = str(model["messages"][1]["content"][1]["text"])
    match = QUERY_PATTERN.search(text)
    if match is None:
        raise ValueError("cannot extract query name for Task 9D evaluation")
    return match.group(1)


def _unseen_prompt(view: str, name: str) -> str:
    if view == "unseen_alpha":
        prefix = f"Inspect the visual content and judge support for the queried pest '{name}'."
    elif view == "unseen_beta":
        prefix = f"From image evidence alone, verify whether the queried pest '{name}' is visible."
    else:
        raise ValueError(f"unknown unseen Task 9D prompt view: {view}")
    return prefix + _COMMON


def _eval_id(source_id: str, view: str, condition: str) -> str:
    return hashlib.sha256(f"task9d-eval:{source_id}:{view}:{condition}".encode()).hexdigest()[:32]


def _absolute_image(model: dict[str, Any], prepared_root: Path) -> str:
    reference = Path(str(model["messages"][1]["content"][0]["image"]))
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError("prepared evaluation source image must be an opaque relative path")
    resolved = (prepared_root / reference).resolve()
    if not resolved.is_file():
        raise ValueError(f"missing prepared evaluation image: {resolved}")
    return str(resolved)


def build_eval_manifest_from_records(
    model_by_id: dict[str, dict[str, Any]],
    provenance: list[dict[str, Any]],
    challenge_family_ids: Iterable[str],
    prepared_root: str | Path,
    transform_output: str | Path,
) -> list[dict[str, Any]]:
    prepared_root = Path(prepared_root)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in provenance:
        by_family.setdefault(str(row["family_id"]), []).append(row)
    challenge = set(map(str, challenge_family_ids))
    if not challenge or not challenge.issubset(by_family):
        raise ValueError("challenge families must be a non-empty subset of dev provenance")
    rows: list[dict[str, Any]] = []
    positive_core: dict[str, dict[str, Any]] = {}
    for family_id in sorted(by_family):
        family_rows = by_family[family_id]
        roles = {str(row["role"]) for row in family_rows}
        if roles != {"positive", "semantic_negative", "visual_counterfactual"}:
            raise ValueError(f"dev family {family_id} lacks exact 1:1:1 roles")
        for private in sorted(family_rows, key=lambda row: str(row["role"])):
            source_id = str(private["id"])
            model = copy.deepcopy(model_by_id[source_id])
            name = _query_name(model)
            model["messages"][1]["content"][0]["image"] = _absolute_image(model, prepared_root)
            model["messages"][1]["content"][1]["text"] = build_prompt("dev_neutral_0", name)
            role = str(private["role"])
            condition = {"positive": "original", "semantic_negative": "semantic_null",
                         "visual_counterfactual": "source_visual_null"}[role]
            gt_bbox = None
            if role == "positive":
                target = json.loads(model["messages"][2]["content"][0]["text"])
                gt_bbox = target.get("evidence_region")
            row = {
                "id": _eval_id(source_id, "canonical", condition),
                "source_id": source_id, "family_id": family_id, "role": role,
                "condition": condition, "prompt_view": "canonical",
                "query_class_id": int(private["query_class_id"]), "gt_bbox": gt_bbox,
                "messages": model["messages"][:2],
            }
            rows.append(row)
            if role == "positive":
                positive_core[family_id] = row

    for family_id in sorted(challenge):
        canonical = positive_core[family_id]
        source_model = model_by_id[str(canonical["source_id"])]
        name = _query_name(source_model)
        for view in ("native_0", "native_1", "native_2", "unseen_alpha", "unseen_beta"):
            row = copy.deepcopy(canonical)
            row["id"] = _eval_id(str(row["source_id"]), view, "original")
            row["prompt_view"] = view
            if view.startswith("native_"):
                row["messages"][1]["content"][1]["text"] = build_prompt(
                    f"train_neutral_{view[-1]}", name
                )
            else:
                row["messages"][1]["content"][1]["text"] = _unseen_prompt(view, name)
            rows.append(row)

    transform_inputs = [{
        "family_id": family_id,
        "image_path": positive_core[family_id]["messages"][1]["content"][0]["image"],
        "query_class_id": positive_core[family_id]["query_class_id"],
        "source_id": positive_core[family_id]["source_id"],
    } for family_id in sorted(challenge)]
    transformed = build_paired_conditions(transform_inputs, transform_output)
    for item in transformed:
        if item["condition"] == "original":
            continue
        canonical = copy.deepcopy(positive_core[str(item["family_id"])])
        canonical.update({
            "id": _eval_id(str(canonical["source_id"]), "canonical", str(item["condition"])),
            "role": "visual_counterfactual", "condition": str(item["condition"]),
            "gt_bbox": None, "pair_id": item["pair_id"], "image_sha256": item["image_sha256"],
        })
        canonical["messages"][1]["content"][0]["image"] = item["image_path"]
        rows.append(canonical)
    identifiers = [str(row["id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Task 9D evaluation IDs are not unique")
    return sorted(rows, key=lambda row: str(row["id"]))


def build_eval_manifest(prepared_root: str | Path) -> dict[str, Any]:
    prepared_root = Path(prepared_root)
    output = prepared_root / "evaluation_protocol"
    if output.exists():
        raise FileExistsError(f"refusing existing Task 9D evaluation protocol: {output}")
    models = [json.loads(line) for line in
              (prepared_root / "dev_audit/source_model.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    model_by_id = {str(row["id"]): row for row in models}
    provenance = [json.loads(line) for line in
                  (prepared_root / "private/provenance.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_ids = set(model_by_id)
    dev_provenance = [row for row in provenance if str(row["id"]) in dev_ids]
    selections = json.loads((prepared_root / "private/selections.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    rows = build_eval_manifest_from_records(
        model_by_id, dev_provenance, selections["challenge"], prepared_root, output / "images"
    )
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "version": "task9d-eval-protocol-v1", "passed": True, "rows": len(rows),
        "dev_families": len(selections["dev"]), "challenge_families": len(selections["challenge"]),
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "prompt_views": sorted({str(row["prompt_view"]) for row in rows}),
        "heldout_transforms": {"blank_rgb": BLANK_RGB, "blur_radius_fraction": BLUR_RADIUS_FRACTION,
                               "shuffle_grid": SHUFFLE_GRID},
        "task8_locked_set_read": False,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
