from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@lru_cache(maxsize=4)
def _dct_basis(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64) + 0.5
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    basis = np.cos(math.pi * frequencies * positions / size)
    basis[0] *= math.sqrt(1.0 / size)
    basis[1:] *= math.sqrt(2.0 / size)
    return basis


def _gray_array(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )


def phash64(path: Path) -> int:
    pixels = _gray_array(Path(path), 32)
    basis = _dct_basis(32)
    low = np.round((basis @ pixels @ basis.T)[:8, :8], decimals=6)
    median = float(np.median(low.reshape(-1)[1:]))
    value = 0
    for coefficient in low.reshape(-1):
        value = (value << 1) | int(coefficient > median)
    return value


def structural_correlation(left: Path, right: Path) -> float:
    left_pixels = _gray_array(Path(left), 64).reshape(-1)
    right_pixels = _gray_array(Path(right), 64).reshape(-1)
    left_pixels -= left_pixels.mean()
    right_pixels -= right_pixels.mean()
    denominator = float(np.linalg.norm(left_pixels) * np.linalg.norm(right_pixels))
    if denominator == 0:
        return float(np.array_equal(left_pixels, right_pixels))
    return float(np.dot(left_pixels, right_pixels) / denominator)


def _path_index(split_rows: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for split, rows in split_rows.items():
        for row in rows:
            path = Path(str(row.get("image", "")))
            key = (split, path.stem)
            previous = index.get(key)
            if previous is not None and previous != path:
                raise ValueError(f"ambiguous image stem within split: {split}:{path.stem}")
            index[key] = path
    return index


def review_candidates(
    split_rows: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    phash_threshold: int = 8,
    correlation_threshold: float = 0.90,
) -> dict[str, Any]:
    index = _path_index(split_rows)
    phash_cache: dict[Path, int] = {}
    reviewed: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate in candidates:
        left_key = (str(candidate["left_split"]), str(candidate["left_id"]))
        right_key = (str(candidate["right_split"]), str(candidate["right_id"]))
        left = index.get(left_key)
        right = index.get(right_key)
        if left is None or right is None or not left.is_file() or not right.is_file():
            missing.append(f"{left_key}->{right_key}")
            continue
        if left not in phash_cache:
            phash_cache[left] = phash64(left)
        if right not in phash_cache:
            phash_cache[right] = phash64(right)
        distance = (phash_cache[left] ^ phash_cache[right]).bit_count()
        correlation = structural_correlation(left, right)
        reviewed.append(
            {
                **candidate,
                "phash_distance": distance,
                "structural_correlation": correlation,
                "high_confidence": distance <= phash_threshold
                and correlation >= correlation_threshold,
            }
        )
    high_confidence = [row for row in reviewed if row["high_confidence"]]
    contaminated: dict[str, set[str]] = {
        split: set() for split in sorted(split_rows)
    }
    for row in high_confidence:
        contaminated[str(row["left_split"])].add(str(row["left_id"]))
        contaminated[str(row["right_split"])].add(str(row["right_id"]))
    return {
        "candidate_count": len(candidates),
        "reviewed_count": len(reviewed),
        "missing_count": len(missing),
        "missing": missing,
        "phash_threshold": phash_threshold,
        "correlation_threshold": correlation_threshold,
        "high_confidence_count": len(high_confidence),
        "requires_manual_resolution": bool(high_confidence or missing),
        "contaminated_image_ids_by_split": {
            split: sorted(image_ids) for split, image_ids in contaminated.items()
        },
        "high_confidence_candidates": high_confidence,
        "reviewed_candidates": reviewed,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-stage Task 8 near-duplicate review")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--leakage-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    leakage = json.loads(args.leakage_report.read_text(encoding="utf-8"))
    result = review_candidates(
        {
            "train": _load_jsonl(args.train_jsonl),
            "val": _load_jsonl(args.val_jsonl),
            "test": _load_jsonl(args.test_jsonl),
        },
        leakage.get("near_duplicate_candidates", []),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_count",
                    "reviewed_count",
                    "missing_count",
                    "high_confidence_count",
                    "requires_manual_resolution",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
