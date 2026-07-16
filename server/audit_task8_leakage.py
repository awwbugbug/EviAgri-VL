from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


OPAQUE_STEM = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_NEUTRAL_PROMPT_TOKENS = (
    "positive sample",
    "null sample",
    "task_type",
    "split=",
    ".jpg",
    "/root/",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(path: Path) -> int:
    with Image.open(path) as source:
        pixels = list(source.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _near_duplicate_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for index, item in enumerate(items):
        value = item["dhash"]
        for chunk_index, (offset, width) in enumerate(chunks):
            key = (chunk_index, (value >> offset) & ((1 << width) - 1))
            for other in buckets[key]:
                if items[other]["split"] != item["split"]:
                    candidates.add((other, index))
            buckets[key].append(index)
    rows = []
    for left_index, right_index in sorted(candidates):
        left, right = items[left_index], items[right_index]
        distance = hamming_distance(left["dhash"], right["dhash"])
        if left["sha256"] != right["sha256"] and distance <= 4:
            rows.append(
                {
                    "left_split": left["split"],
                    "left_id": left["id"],
                    "right_split": right["split"],
                    "right_id": right["id"],
                    "dhash_distance": distance,
                }
            )
    return rows


def _template_summary(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for group in sorted({job["group"] for job in jobs}):
        prompts = [str(job["prompt"]) for job in jobs if job["group"] == group]
        result[group] = {
            "rows": len(prompts),
            "unique_prompt_sha256": len(
                {hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts}
            ),
            "length_min": min(map(len, prompts), default=0),
            "length_max": max(map(len, prompts), default=0),
        }
    return result


def _answer_summary(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    lengths: dict[str, list[int]] = defaultdict(list)
    for rows in split_rows.values():
        for row in rows:
            task = str(row.get("task_type", "unknown"))
            messages = row.get("messages") or []
            if len(messages) < 2:
                continue
            content = messages[1].get("content") or []
            text = content[0].get("text", "") if content else ""
            lengths[task].append(len(text))
    return {
        task: {
            "count": len(values),
            "length_min": min(values),
            "length_max": max(values),
            "length_mean": sum(values) / len(values),
        }
        for task, values in sorted(lengths.items())
        if values
    }


def audit_task8_leakage(
    split_rows: dict[str, list[dict[str, Any]]],
    audit_rows: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    hard_failures: list[str] = []
    warnings: list[str] = []
    audit_ids = [str(row.get("audit_id")) for row in audit_rows]
    duplicates = sorted({value for value in audit_ids if audit_ids.count(value) > 1})
    if duplicates:
        hard_failures.append(f"duplicate audit_id: {duplicates[:10]}")

    for row in audit_rows:
        path = Path(str(row.get("image", "")))
        if (
            path.parent.name != "images"
            or not OPAQUE_STEM.fullmatch(path.stem)
            or str(row.get("actual_image_pest_name", "")).lower() in str(path).lower()
        ):
            hard_failures.append(f"derived image path is not opaque: {path}")
        if not path.is_file():
            hard_failures.append(f"derived image is missing: {path}")
        elif _sha256_file(path) != row.get("image_sha256"):
            hard_failures.append(f"derived image SHA256 mismatch: {row.get('audit_id')}")

    jobs_by_audit: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for job in jobs:
        jobs_by_audit[str(job.get("audit_id"))][str(job.get("group"))] = job
        if job.get("group") in {"B1", "B2"}:
            lowered = str(job.get("prompt", "")).lower()
            for token in FORBIDDEN_NEUTRAL_PROMPT_TOKENS:
                if token in lowered:
                    hard_failures.append(
                        f"forbidden prompt token in {job.get('job_id')}: {token}"
                    )
    for audit_id in sorted(set(audit_ids)):
        grouped = jobs_by_audit.get(audit_id, {})
        if set(grouped) != {"B0", "B1", "B2", "B3"}:
            hard_failures.append(f"missing B0-B3 jobs for audit_id: {audit_id}")
            continue
        b1, b2 = grouped["B1"], grouped["B2"]
        comparable = ("prompt", "protocol_hash", "image", "image_sha256")
        if any(b1.get(key) != b2.get(key) for key in comparable):
            hard_failures.append(f"B1/B2 same-protocol mismatch: {audit_id}")

    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    source_id_to_splits: dict[str, set[str]] = defaultdict(set)
    image_items: list[dict[str, Any]] = []
    cache: dict[Path, tuple[str, int]] = {}
    perceptual_seen: set[tuple[str, Path]] = set()
    for split, rows in split_rows.items():
        for row in rows:
            path = Path(str(row.get("image", "")))
            if not path.is_file():
                hard_failures.append(f"missing split image: {split}:{path}")
                continue
            if path not in cache:
                cache[path] = (_sha256_file(path), dhash64(path))
            sha256, perceptual = cache[path]
            hash_to_splits[sha256].add(split)
            source_id_to_splits[path.stem].add(split)
            perceptual_key = (split, path)
            if perceptual_key not in perceptual_seen:
                perceptual_seen.add(perceptual_key)
                image_items.append(
                    {
                        "split": split,
                        "id": path.stem,
                        "sha256": sha256,
                        "dhash": perceptual,
                    }
                )
    exact_cross_split = sorted(
        digest for digest, splits in hash_to_splits.items() if len(splits) > 1
    )
    source_cross_split = sorted(
        source_id for source_id, splits in source_id_to_splits.items() if len(splits) > 1
    )
    if exact_cross_split:
        hard_failures.append(
            f"exact image duplicate across splits: {exact_cross_split[:10]}"
        )
    if source_cross_split:
        hard_failures.append(
            f"source image id across splits: {source_cross_split[:10]}"
        )
    near_duplicates = _near_duplicate_candidates(image_items)
    if near_duplicates:
        warnings.append(
            f"cross-split perceptual near-duplicate candidates require review: {len(near_duplicates)}"
        )
    return {
        "passed": not hard_failures,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "exact_cross_split_sha256": exact_cross_split,
        "source_ids_crossing_splits": source_cross_split,
        "perceptual_unique_images": len(image_items),
        "near_duplicate_candidates": near_duplicates,
        "template_summary": _template_summary(jobs),
        "answer_summary": _answer_summary(split_rows),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Task 8 for hidden leakage")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--inference-jobs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_task8_leakage(
        {
            "train": _load_jsonl(args.train_jsonl),
            "val": _load_jsonl(args.val_jsonl),
            "test": _load_jsonl(args.test_jsonl),
        },
        _load_jsonl(args.audit_manifest),
        _load_jsonl(args.inference_jobs),
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
