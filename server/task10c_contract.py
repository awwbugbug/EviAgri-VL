"""Frozen Task 10C C0 protocol derived from the signed Task 10B v2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_MANIFEST_SHA256 = "84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90"
CLASS_IDS = (9, 10, 16, 17, 22, 24, 45, 50, 64, 68, 71, 82, 83, 87, 99, 101)
SPLIT_COUNTS = {"train": 192, "val": 48, "dev": 80}
PER_CLASS = {"train": 12, "val": 3, "dev": 5}
SYSTEM_PROMPT = "You are an agricultural pest image classifier. Follow the requested output format exactly."
TRAIN_PROMPT = "Identify the insect pest shown in this image. Return exactly one JSON object containing its canonical IP102 class ID."
UNSEEN_PROMPT = "Which insect pest category is visible in this image? Reply with one JSON object containing only its canonical IP102 identifier."
PARSER_VERSION = "task10c-strict-pest-json-v1"
_CANONICAL_PATTERN = re.compile(r"^IP\d{3}$")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def canonical_pest_id(class_id: int) -> str:
    class_id = int(class_id)
    if class_id not in CLASS_IDS:
        raise ValueError(f"class is outside frozen Task 10C set: {class_id}")
    return f"IP{class_id:03d}"


def strict_parse_pest_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {
            "syntax_valid": False,
            "schema_valid": False,
            "pest_id": None,
            "error": "invalid_json",
        }
    compact = isinstance(value, dict) and raw == json.dumps(value, separators=(",", ":"))
    pest_id = value.get("pest_id") if isinstance(value, dict) else None
    schema_valid = bool(
        compact
        and list(value) == ["pest_id"]
        and isinstance(pest_id, str)
        and _CANONICAL_PATTERN.fullmatch(pest_id)
        and pest_id in {f"IP{item:03d}" for item in CLASS_IDS}
    )
    return {
        "syntax_valid": True,
        "schema_valid": schema_valid,
        "pest_id": pest_id if schema_valid else None,
        "error": None if schema_valid else "schema_or_canonical_format",
    }


def _cross_split_overlap(rows: list[dict[str, Any]], key: str) -> int:
    by_split = {
        split: {str(row[key]) for row in rows if row["split"] == split}
        for split in SPLIT_COUNTS
    }
    return sum(
        len(by_split[left] & by_split[right])
        for index, left in enumerate(SPLIT_COUNTS)
        for right in list(SPLIT_COUNTS)[index + 1 :]
    )


def _model_envelope(row: dict[str, Any]) -> dict[str, Any]:
    class_id = int(row["class_id"])
    target = json.dumps(
        {"pest_id": canonical_pest_id(class_id)},
        separators=(",", ":"),
    )
    model = {
        "id": str(row["id"]),
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(row["image"])},
                    {"type": "text", "text": TRAIN_PROMPT},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
    }
    return {
        "id": str(row["id"]),
        "image": str(row["image"]),
        "class_id": class_id,
        "class_band": str(row["class_band"]),
        "split": str(row["split"]),
        "source_image_id": str(row["source_image_id"]),
        "source_image_sha256": str(row["source_image_sha256"]),
        "near_duplicate_component_id": str(row["near_duplicate_component_id"]),
        "model": model,
    }


def build_task10c_protocol(
    rows: list[dict[str, Any]],
    manifest_sha256: str,
    *,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"manifest SHA256 mismatch: expected={expected_manifest_sha256} actual={manifest_sha256}"
        )
    required = {
        "id", "image", "class_id", "class_band", "split", "source_image_id",
        "source_image_sha256", "near_duplicate_component_id",
    }
    if any(required - set(row) for row in rows):
        raise ValueError("Task 10C manifest row is missing required fields")
    normalized = sorted(
        ({**row, "class_id": int(row["class_id"])} for row in rows),
        key=lambda row: (str(row["split"]), int(row["class_id"]), str(row["source_image_sha256"]), str(row["id"])),
    )
    if set(int(row["class_id"]) for row in normalized) != set(CLASS_IDS):
        raise ValueError("Task 10C frozen class set mismatch")
    split_counts = Counter(str(row["split"]) for row in normalized)
    if dict(split_counts) != SPLIT_COUNTS:
        raise ValueError(f"split row count mismatch: {dict(split_counts)}")
    for split, quota in PER_CLASS.items():
        counts = Counter(int(row["class_id"]) for row in normalized if row["split"] == split)
        if counts != Counter({class_id: quota for class_id in CLASS_IDS}):
            raise ValueError(f"per-class quota mismatch for {split}: {dict(counts)}")
    source_overlap = _cross_split_overlap(normalized, "source_image_sha256")
    component_overlap = _cross_split_overlap(normalized, "near_duplicate_component_id")
    if source_overlap:
        raise ValueError(f"cross-split source SHA overlap: {source_overlap}")
    if component_overlap:
        raise ValueError(f"cross-split near-duplicate overlap: {component_overlap}")
    if len({str(row["source_image_sha256"]) for row in normalized}) != len(normalized):
        raise ValueError("Task 10C manifest repeats a source image")
    if len({str(row["near_duplicate_component_id"]) for row in normalized}) != len(normalized):
        raise ValueError("Task 10C manifest repeats a near-duplicate component")

    envelopes = [_model_envelope(row) for row in normalized]
    by_split = {
        split: [row for row in envelopes if row["split"] == split]
        for split in SPLIT_COUNTS
    }
    smoke_train: list[dict[str, Any]] = []
    smoke_dev: list[dict[str, Any]] = []
    for class_id in CLASS_IDS:
        class_train = [row for row in by_split["train"] if row["class_id"] == class_id]
        class_dev = [row for row in by_split["dev"] if row["class_id"] == class_id]
        smoke_train.extend(class_train[:4])
        smoke_dev.extend(class_dev[:1])
    report = {
        "version": "task10c-c0-preflight-v1",
        "passed": True,
        "manifest_sha256": manifest_sha256,
        "class_ids": list(CLASS_IDS),
        "rows_by_split": dict(sorted(split_counts.items())),
        "per_class_by_split": dict(PER_CLASS),
        "source_overlap": source_overlap,
        "component_overlap": component_overlap,
        "smoke_train_count": len(smoke_train),
        "smoke_dev_count": len(smoke_dev),
        "task8_locked_content_read": False,
        "official_test_references": 0,
        "model_loaded": False,
    }
    return {
        "report": report,
        "train": by_split["train"],
        "val": by_split["val"],
        "dev": by_split["dev"],
        "smoke_train": smoke_train,
        "smoke_dev": smoke_dev,
    }


def _model_hash_report(model_path: Path) -> list[dict[str, Any]]:
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"Task 10C model directory has no safetensors: {model_path}")
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in shards
    ]


def _completion(output: Path, names: list[str]) -> None:
    (output / "completion.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def run_protocol(
    manifest_path: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    *,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    manifest_path, model_path, output = Path(manifest_path), Path(model_path), Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Task 10C protocol: {output}")
    output.mkdir(parents=True)
    try:
        actual_manifest_sha = _sha256(manifest_path)
        result = build_task10c_protocol(
            _read_jsonl(manifest_path),
            actual_manifest_sha,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        model_hashes = _model_hash_report(model_path)
        for name in ("train", "val", "dev", "smoke_train", "smoke_dev"):
            _write_jsonl(output / f"{name}.jsonl", result[name])
        _write_json(output / "preflight_report.json", result["report"])
        _write_json(output / "model_files.sha256.json", model_hashes)
        _write_json(output / "config.snapshot.json", {
            "version": "task10c-c0-config-v1",
            "manifest": str(manifest_path),
            "manifest_sha256": actual_manifest_sha,
            "model_path": str(model_path),
            "system_prompt": SYSTEM_PROMPT,
            "train_prompt": TRAIN_PROMPT,
            "unseen_prompt": UNSEEN_PROMPT,
            "parser_version": PARSER_VERSION,
            "vision": {"min_pixels": 200704, "max_pixels": 401408},
            "training": {
                "seeds": [17, 29, 43], "optimizer_steps": 8,
                "batch_size": 1, "gradient_accumulation_steps": 8,
                "learning_rate": 0.0001,
            },
        })
        status = {"state": "completed", "stage": "c0", "passed": True}
        _write_json(output / "status.json", status)
        names = [
            "train.jsonl", "val.jsonl", "dev.jsonl", "smoke_train.jsonl",
            "smoke_dev.jsonl", "preflight_report.json", "config.snapshot.json",
            "model_files.sha256.json", "status.json",
        ]
        _completion(output, names)
        return status
    except Exception as exc:
        _write_json(output / "status.json", {"state": "blocked", "stage": "c0"})
        _write_json(output / "failure.json", {
            "state": "blocked", "stage": "c0", "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Task 10C C0 protocol")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_protocol(args.manifest, args.model_path, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
