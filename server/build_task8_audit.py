from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from task8_protocol import GROUPS, build_prompt, protocol_hash


CONDITIONS = (
    "original_correct",
    "original_wrong_query",
    "shuffled_image",
    "strong_blur",
    "blank_image",
    "no_target_image",
)


def _diagnosis(record: dict[str, Any]) -> dict[str, Any]:
    target = record.get("target", {})
    diagnosis = target.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise ValueError(f"positive record has no diagnosis object: {record.get('id')}")
    return diagnosis


def select_families(
    records: list[dict[str, Any]],
    per_class: int,
    seed: int,
    max_classes: int | None = None,
    exclude_image_stems: set[str] | None = None,
) -> list[dict[str, Any]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    excluded = exclude_image_stems or set()
    for record in records:
        if record.get("task_type") != "pest_evidence_grounding":
            continue
        if Path(str(record.get("image", ""))).stem in excluded:
            continue
        if not Path(str(record.get("image", ""))).is_file():
            raise ValueError(f"missing audit source image: {record.get('image')}")
        pest_id = int(_diagnosis(record)["pest_id"])
        grouped[pest_id].append(record)
    class_ids = sorted(grouped)
    if max_classes is not None:
        if max_classes < 2:
            raise ValueError("max_classes must be at least two")
        if max_classes < len(class_ids):
            class_ids = sorted(random.Random(seed).sample(class_ids, max_classes))
    selected: list[dict[str, Any]] = []
    for pest_id in class_ids:
        candidates = sorted(grouped[pest_id], key=lambda row: str(row["id"]))
        if len(candidates) < per_class:
            raise ValueError(
                f"class {pest_id} has {len(candidates)} records, needs {per_class}"
            )
        rng = random.Random(f"{seed}:{pest_id}")
        selected.extend(sorted(rng.sample(candidates, per_class), key=lambda row: str(row["id"])))
    if len(grouped) < 2:
        raise ValueError("Task 8 requires at least two pest classes")
    return selected


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _save_content_addressed(
    image: Image.Image, temporary_images: Path, final_images: Path
) -> tuple[str, str]:
    data = _png_bytes(image)
    digest = _sha256_bytes(data)
    temporary_path = temporary_images / f"{digest}.png"
    if not temporary_path.exists():
        temporary_path.write_bytes(data)
    return str(final_images / temporary_path.name), digest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partner_rows(
    families: list[dict[str, Any]], index: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    own_class = int(_diagnosis(families[index])["pest_id"])
    candidates = [
        row for row in families if int(_diagnosis(row)["pest_id"]) != own_class
    ]
    rng = random.Random(f"{seed}:{families[index]['id']}:partners")
    shuffled = candidates[rng.randrange(len(candidates))]
    remaining = [row for row in candidates if row["id"] != shuffled["id"]]
    no_target = remaining[rng.randrange(len(remaining))] if remaining else shuffled
    return shuffled, no_target


def build_audit_dataset(
    records: list[dict[str, Any]],
    output_dir: Path,
    per_class: int,
    seed: int,
    max_classes: int | None = None,
    exclude_image_stems: set[str] | None = None,
    exclusion_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
    if output_dir.exists():
        output_dir.rmdir()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"temporary audit output already exists: {temporary}")
    temporary_images = temporary / "images"
    final_images = output_dir / "images"
    temporary_images.mkdir(parents=True)
    try:
        families = select_families(
            records,
            per_class=per_class,
            seed=seed,
            max_classes=max_classes,
            exclude_image_stems=exclude_image_stems,
        )
        family_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for index, record in enumerate(families):
            diagnosis = _diagnosis(record)
            pest_id = int(diagnosis["pest_id"])
            pest_name = str(diagnosis["pest_name"])
            family_id = hashlib.sha256(f"task8:{record['id']}".encode()).hexdigest()[:24]
            shuffled_record, no_target_record = _partner_rows(families, index, seed)
            wrong_diagnosis = _diagnosis(no_target_record)
            with Image.open(record["image"]) as source_file:
                source = source_file.convert("RGB")
            size = source.size
            source_png = _png_bytes(source)
            original_sha = _sha256_bytes(source_png)
            family_rows.append(
                {
                    "family_id": family_id,
                    "source_id": record["id"],
                    "source_image_sha256": original_sha,
                    "pest_id": pest_id,
                    "pest_name": pest_name,
                    "gt_bbox": record["target"]["evidence_bbox"],
                    "width": size[0],
                    "height": size[1],
                }
            )
            for condition in CONDITIONS:
                query_id = pest_id
                query_name = pest_name
                actual_id = pest_id
                actual_name = pest_name
                derived_from = record["id"]
                transformed = source.copy()
                if condition == "original_wrong_query":
                    query_id = int(wrong_diagnosis["pest_id"])
                    query_name = str(wrong_diagnosis["pest_name"])
                elif condition in {"shuffled_image", "no_target_image"}:
                    partner = shuffled_record if condition == "shuffled_image" else no_target_record
                    partner_diagnosis = _diagnosis(partner)
                    with Image.open(partner["image"]) as partner_file:
                        transformed = partner_file.convert("RGB").resize(size, Image.Resampling.BICUBIC)
                    actual_id = int(partner_diagnosis["pest_id"])
                    actual_name = str(partner_diagnosis["pest_name"])
                    derived_from = partner["id"]
                elif condition == "strong_blur":
                    transformed = source.filter(
                        ImageFilter.GaussianBlur(radius=max(size) / 20.0)
                    )
                elif condition == "blank_image":
                    transformed = Image.new("RGB", size, color=(127, 127, 127))
                image_path, image_sha = _save_content_addressed(
                    transformed, temporary_images, final_images
                )
                is_positive = condition == "original_correct"
                audit_id = hashlib.sha256(f"{family_id}:{condition}".encode()).hexdigest()[:32]
                audit_rows.append(
                    {
                        "audit_id": audit_id,
                        "family_id": family_id,
                        "condition": condition,
                        "image": image_path,
                        "image_sha256": image_sha,
                        "source_image_sha256": original_sha,
                        "source_id": record["id"],
                        "derived_from_source_id": derived_from,
                        "query_pest_id": query_id,
                        "query_pest_name": query_name,
                        "actual_image_pest_id": actual_id,
                        "actual_image_pest_name": actual_name,
                        "expected_evidence_present": is_positive,
                        "gt_bbox": record["target"]["evidence_bbox"] if is_positive else None,
                        "width": size[0],
                        "height": size[1],
                    }
                )
        jobs = [
            {
                **row,
                "job_id": f"{group}:{row['audit_id']}",
                "group": group,
                "prompt": build_prompt(
                    group,
                    row["query_pest_name"],
                    row["condition"],
                    queried_pest_id=row["query_pest_id"],
                ),
                "protocol_hash": protocol_hash(group),
            }
            for row in audit_rows
            for group in GROUPS
        ]
        _write_jsonl(temporary / "family_manifest.jsonl", family_rows)
        _write_jsonl(temporary / "audit_manifest.jsonl", audit_rows)
        _write_jsonl(temporary / "inference_jobs.jsonl", jobs)
        summary = {
            "seed": seed,
            "per_class": per_class,
            "max_classes": max_classes,
            "classes": len({_diagnosis(row)["pest_id"] for row in families}),
            "families": len(families),
            "audit_rows": len(audit_rows),
            "inference_jobs": len(jobs),
            "conditions": list(CONDITIONS),
            "groups": list(GROUPS),
            "excluded_test_image_stems": len(exclude_image_stems or set()),
            "exclusion_manifest_sha256": exclusion_manifest_sha256,
        }
        (temporary / "build_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        manifest_files = sorted(
            path for path in temporary.rglob("*") if path.is_file()
        )
        (temporary / "manifest.sha256").write_text(
            "".join(
                f"{_file_sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in manifest_files
            ),
            encoding="utf-8",
        )
        temporary.replace(output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task 8 paired counterfactual audit")
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-class", required=True, type=int)
    parser.add_argument("--max-classes", type=int)
    parser.add_argument("--seed", default=20260715, type=int)
    parser.add_argument("--exclusion-review", type=Path)
    args = parser.parse_args()
    excluded: set[str] = set()
    exclusion_sha = None
    if args.exclusion_review is not None:
        exclusion = json.loads(args.exclusion_review.read_text(encoding="utf-8"))
        if exclusion.get("missing_count") != 0:
            raise ValueError("exclusion review has unresolved missing images")
        excluded = set(
            exclusion.get("contaminated_image_ids_by_split", {}).get("test", [])
        )
        exclusion_sha = _file_sha256(args.exclusion_review)
    summary = build_audit_dataset(
        _load_jsonl(args.test_jsonl),
        args.output_dir,
        args.per_class,
        args.seed,
        args.max_classes,
        excluded,
        exclusion_sha,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
