"""Small fail-closed primitives shared by Task 10 forensic audits."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    """Write UTF-8 JSON while refusing to replace an existing file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_new_directory(path: Path) -> None:
    """Create a directory only when no file-system entry already exists."""
    Path(path).mkdir(parents=True, exist_ok=False)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def family_bootstrap_delta(
    values: list[tuple[str, float, float]],
    repetitions: int = 1000,
    seed: int = 20260717,
) -> dict[str, float | int | str]:
    """Estimate a paired family-level mean delta and percentile interval."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    by_family = {str(family_id): (float(left), float(right)) for family_id, left, right in values}
    families = sorted(by_family)
    if not families or len(families) != len(values):
        raise ValueError("family bootstrap requires one non-empty paired row per family")

    rng = random.Random(seed)
    observed = mean(by_family[family][0] - by_family[family][1] for family in families)
    samples = []
    for _ in range(repetitions):
        draw = [families[rng.randrange(len(families))] for _ in families]
        samples.append(mean(by_family[family][0] - by_family[family][1] for family in draw))
    samples.sort()
    return {
        "estimate": observed,
        "low": _percentile(samples, 0.025),
        "high": _percentile(samples, 0.975),
        "repetitions": repetitions,
        "seed": seed,
        "unit": "family_id",
    }
