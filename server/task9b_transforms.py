"""Deterministic, disjoint counterfactual transforms for Task 9B.

The train and development registries deliberately use different transform
names and parameter domains.  Neither reproduces the Task 8 confirmation
perturbations (0.05-radius blur, uniform RGB 127 blank, or whole-image class
substitution).
"""

from __future__ import annotations

import hashlib
import random
from typing import Mapping, Sequence

from PIL import Image, ImageFilter


TRAIN_TRANSFORMS: dict[str, dict[str, list[float] | list[int]]] = {
    "train_blur": {"radius_fraction": [0.08, 0.11]},
    "train_patch_permute": {"grids": [3, 4]},
    "train_evidence_exclusion_crop": {"margin_fraction": [0.03, 0.08]},
    "train_low_information": {"mean_range": [96, 118], "noise_range": [3, 8]},
}

DEV_TRANSFORMS: dict[str, dict[str, list[float] | list[int]]] = {
    "dev_blur": {"radius_fraction": [0.14, 0.18]},
    "dev_patch_permute": {"grids": [5, 6]},
    "dev_evidence_exclusion_crop": {"margin_fraction": [0.12, 0.18]},
    "dev_low_information": {"mean_range": [145, 168], "noise_range": [9, 14]},
}


def _registry(surface: str) -> Mapping[str, dict]:
    if surface == "train":
        return TRAIN_TRANSFORMS
    if surface == "dev":
        return DEV_TRANSFORMS
    raise ValueError(f"unknown surface: {surface!r}")


def _rng(surface: str, kind: str, seed: int, image: Image.Image, bbox: Sequence[float]) -> random.Random:
    material = f"{surface}|{kind}|{seed}|{image.size}|{tuple(bbox)}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def transform_kind_for_index(index: int, surface: str) -> str:
    """Rotate deterministically and exactly evenly over complete cycles."""
    if index < 0:
        raise ValueError("index must be non-negative")
    kinds = tuple(_registry(surface))
    return kinds[index % len(kinds)]


def _patch_permute(image: Image.Image, grid: int, rng: random.Random) -> Image.Image:
    width, height = image.size
    work_width = max(grid, (width // grid) * grid)
    work_height = max(grid, (height // grid) * grid)
    work = image.resize((work_width, work_height), Image.Resampling.BICUBIC)
    tile_width, tile_height = work_width // grid, work_height // grid
    tiles = [
        work.crop((x * tile_width, y * tile_height, (x + 1) * tile_width, (y + 1) * tile_height))
        for y in range(grid)
        for x in range(grid)
    ]
    order = list(range(len(tiles)))
    rng.shuffle(order)
    if order == list(range(len(tiles))):
        order = order[1:] + order[:1]
    output = Image.new("RGB", work.size)
    for destination, source in enumerate(order):
        x, y = destination % grid, destination // grid
        output.paste(tiles[source], (x * tile_width, y * tile_height))
    return output.resize(image.size, Image.Resampling.BICUBIC)


def _evidence_exclusion_crop(
    image: Image.Image, bbox: Sequence[float], margin_fraction: float
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    margin_x, margin_y = width * margin_fraction, height * margin_fraction
    candidates = [
        (0, 0, max(0, int(x1 - margin_x)), height),
        (min(width, int(x2 + margin_x)), 0, width, height),
        (0, 0, width, max(0, int(y1 - margin_y))),
        (0, min(height, int(y2 + margin_y)), width, height),
    ]
    viable = [box for box in candidates if box[2] - box[0] >= 2 and box[3] - box[1] >= 2]
    if not viable:
        # The target fills the frame: retain no recognizable evidence.
        return image.resize((1, 1), Image.Resampling.BOX).resize(image.size, Image.Resampling.NEAREST)
    crop = max(viable, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    return image.crop(crop).resize(image.size, Image.Resampling.BICUBIC)


def _low_information(image: Image.Image, mean: int, amplitude: int, rng: random.Random) -> Image.Image:
    pixels = []
    for _ in range(image.width * image.height):
        pixels.append(tuple(max(0, min(255, mean + rng.randint(-amplitude, amplitude))) for _ in range(3)))
    output = Image.new("RGB", image.size)
    output.putdata(pixels)
    return output


def apply_transform(
    surface: str,
    kind: str,
    image: Image.Image,
    bbox: Sequence[float],
    *,
    seed: int,
) -> Image.Image:
    """Apply a registered transform without mutating the source image."""
    registry = _registry(surface)
    if kind not in registry:
        raise ValueError(f"transform {kind!r} is not registered for {surface!r}")
    image = image.convert("RGB")
    rng = _rng(surface, kind, seed, image, bbox)
    parameters = registry[kind]

    if kind.endswith("_blur"):
        low, high = parameters["radius_fraction"]
        radius = max(image.size) * rng.uniform(float(low), float(high))
        return image.filter(ImageFilter.GaussianBlur(radius))
    if kind.endswith("_patch_permute"):
        return _patch_permute(image, rng.choice(parameters["grids"]), rng)
    if kind.endswith("_evidence_exclusion_crop"):
        low, high = parameters["margin_fraction"]
        return _evidence_exclusion_crop(image, bbox, rng.uniform(float(low), float(high)))
    if kind.endswith("_low_information"):
        mean = rng.randint(*[int(value) for value in parameters["mean_range"]])
        amplitude = rng.randint(*[int(value) for value in parameters["noise_range"]])
        return _low_information(image, mean, amplitude, rng)
    raise AssertionError(f"unimplemented transform: {kind}")
