import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from static_qlora_config import load_training_config


SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing source JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"record at {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def choose_null(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"requested {count} null records but only {len(records)} are available")
    return sorted(records, key=lambda row: stable_rank(seed, str(row.get("id", ""))))[:count]


def validate_unique_existing_images(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in records:
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("every record must have a non-empty string id")
        if record_id in seen:
            raise ValueError(f"duplicate record id: {record_id}")
        seen.add(record_id)
        image = row.get("image")
        if not isinstance(image, str) or not Path(image).is_file():
            raise ValueError(f"missing image for record {record_id}: {image}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _source_paths(source: Path) -> dict[str, dict[str, Path]]:
    return {
        split: {
            "positive": source / "vlm_sft" / f"{split}_evidence_positive.jsonl",
            "null": source / "hallucination" / f"{split}_prompt_conflict.jsonl",
        }
        for split in SPLITS
    }


def build_mix(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    source = Path(config["source_data_root"])
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output_root}")

    paths = _source_paths(source)
    positive = {split: read_jsonl(paths[split]["positive"]) for split in SPLITS}
    null = {split: read_jsonl(paths[split]["null"]) for split in SPLITS}
    expected_positive = config["data"]["train_positive"]
    if len(positive["train"]) != expected_positive:
        raise ValueError(
            f"expected {expected_positive} train positive records, found {len(positive['train'])}"
        )
    selected_train_null = choose_null(null["train"], config["data"]["train_null"], config["seed"])

    bundle: dict[str, list[dict[str, Any]]] = {
        "train": positive["train"] + selected_train_null,
        "val": positive["val"] + null["val"],
        "test": positive["test"] + null["test"],
    }
    for offset, split in enumerate(SPLITS, start=1):
        bundle[split].sort(key=lambda row: stable_rank(config["seed"] + offset, row["id"]))
    validate_unique_existing_images([row for split in SPLITS for row in bundle[split]])

    counts = {
        "train": {
            "positive": len(positive["train"]),
            "null": len(selected_train_null),
            "total": len(bundle["train"]),
        },
        "val": {
            "positive": len(positive["val"]),
            "null": len(null["val"]),
            "total": len(bundle["val"]),
        },
        "test": {
            "positive": len(positive["test"]),
            "null": len(null["test"]),
            "total": len(bundle["test"]),
        },
    }
    source_sha256 = {
        str(path.relative_to(source)).replace("\\", "/"): sha256_file(path)
        for split_paths in paths.values()
        for path in split_paths.values()
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for split in SPLITS:
            _write_jsonl(temporary / f"{split}.jsonl", bundle[split])
        output_sha256 = {
            f"{split}.jsonl": sha256_file(temporary / f"{split}.jsonl") for split in SPLITS
        }
        manifest = {
            "version": "static_qlora_v1",
            "seed": config["seed"],
            "source_data_root": str(source),
            "counts": counts,
            "selected_train_null_ids": [row["id"] for row in selected_train_null],
            "source_sha256": dict(sorted(source_sha256.items())),
            "output_sha256": output_sha256,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checksum_paths = [temporary / f"{split}.jsonl" for split in SPLITS] + [temporary / "manifest.json"]
        (temporary / "sha256sum.txt").write_text(
            "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
            encoding="utf-8",
        )
        if output_root.exists():
            output_root.rmdir()
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Static QLoRA v1 JSONL splits")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    summary = build_mix(config, Path(config["mixed_data_root"]))
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
