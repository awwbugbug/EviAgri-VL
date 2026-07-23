"""Frozen helpers for the Task 11A confidence-aware taxonomy router."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import balanced_accuracy_score


SEEDS = (17, 29, 43)
CONDITIONS = ("blank", "blur", "shuffle")


def build_stress_rows(source_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in source_rows:
        split = str(row.get("split"))
        if split not in {"val", "dev"}:
            continue
        for condition in ("blank", "blur"):
            output.append({**row, "condition": condition, "stress_seed": 0})
        for seed in SEEDS:
            output.append({**row, "condition": "shuffle", "stress_seed": seed})
    if not output:
        raise ValueError("Task 11A stress rows are empty")
    return output


def _shuffle_seed(row_id: str, split: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"task11a|{split}|{seed}|{row_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def transform_image(
    image: Image.Image,
    *,
    condition: str,
    split: str,
    row_id: str,
    seed: int,
) -> Image.Image:
    rgb = image.convert("RGB")
    if split not in {"val", "dev"}:
        raise ValueError(f"unsupported Task 11A split: {split}")
    if condition == "blank":
        value = 127 if split == "val" else 114
        return Image.new("RGB", rgb.size, (value, value, value))
    if condition == "blur":
        radius = 6 if split == "val" else 10
        return rgb.filter(ImageFilter.GaussianBlur(radius=radius))
    if condition != "shuffle":
        raise ValueError(f"unsupported Task 11A condition: {condition}")
    grid = 6 if split == "val" else 8
    xs = np.linspace(0, rgb.width, grid + 1, dtype=np.int64)
    ys = np.linspace(0, rgb.height, grid + 1, dtype=np.int64)
    patches = [
        rgb.crop((int(xs[x]), int(ys[y]), int(xs[x + 1]), int(ys[y + 1])))
        for y in range(grid)
        for x in range(grid)
    ]
    rng = np.random.default_rng(_shuffle_seed(row_id, split, seed))
    order = rng.permutation(len(patches))
    shuffled = Image.new("RGB", rgb.size)
    for target, source in enumerate(order):
        y, x = divmod(target, grid)
        patch = patches[int(source)].resize(
            (int(xs[x + 1] - xs[x]), int(ys[y + 1] - ys[y])),
            resample=Image.Resampling.BILINEAR,
        )
        shuffled.paste(patch, (int(xs[x]), int(ys[y])))
    return shuffled


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite matrix")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def select_threshold(
    original_confidence: np.ndarray,
    original_correct: np.ndarray,
    null_confidence: np.ndarray,
) -> dict[str, float]:
    original = np.asarray(original_confidence, dtype=np.float64)
    correct = np.asarray(original_correct, dtype=bool)
    null = np.asarray(null_confidence, dtype=np.float64)
    if original.shape != correct.shape or original.ndim != 1 or null.ndim != 1:
        raise ValueError("threshold arrays have incompatible shapes")
    if not len(original) or not len(null):
        raise ValueError("threshold selection requires original and null samples")
    truth = np.concatenate([correct, np.zeros(len(null), dtype=bool)]).astype(int)
    confidence = np.concatenate([original, null])
    best = None
    for value in np.linspace(0.0, 1.0, 101):
        predicted = (confidence >= value).astype(int)
        score = float(balanced_accuracy_score(truth, predicted))
        candidate = (score, -float(value))
        if best is None or candidate > best[0]:
            best = (candidate, float(value), score)
    assert best is not None
    return {"threshold": best[1], "balanced_accuracy": best[2]}


def render_router_json(prediction: int | None) -> dict[str, Any]:
    if prediction is None:
        return {
            "decision": "abstain",
            "diagnosis": {"pest_id": None},
            "evidence_present": False,
            "evidence_region": None,
        }
    return {
        "decision": "diagnosis",
        "diagnosis": {"pest_id": int(prediction)},
        "evidence_present": True,
        "evidence_region": None,
    }


def validate_router_json(payload: dict[str, Any]) -> bool:
    if list(payload) != [
        "decision",
        "diagnosis",
        "evidence_present",
        "evidence_region",
    ]:
        return False
    if not isinstance(payload.get("diagnosis"), dict):
        return False
    decision = payload.get("decision")
    pest_id = payload["diagnosis"].get("pest_id")
    if decision == "abstain":
        return (
            payload.get("evidence_present") is False
            and payload.get("evidence_region") is None
            and pest_id is None
        )
    return (
        decision == "diagnosis"
        and payload.get("evidence_present") is True
        and payload.get("evidence_region") is None
        and isinstance(pest_id, int)
        and not isinstance(pest_id, bool)
    )


def serialized_payload(payload: dict[str, Any]) -> str:
    if not validate_router_json(payload):
        raise ValueError("invalid Task 11A router payload")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decide_seed(metrics: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "original_macro_f1_delta_ge_minus_0_03": float(
            metrics["confidence_original"]["macro_f1_delta"]
        )
        >= -0.03,
        "original_coverage_ge_0_70": float(metrics["confidence_original"]["coverage"])
        >= 0.70,
        "blank_null_fpr_lt_0_10": float(metrics["null"]["blank_fpr"]) < 0.10,
        "blur_null_fpr_lt_0_10": float(metrics["null"]["blur_fpr"]) < 0.10,
        "shuffle_null_fpr_lt_0_25": float(metrics["null"]["shuffle_fpr"]) < 0.25,
        "json_contract_all_one": all(
            float(metrics["json_contract"][key]) == 1.0
            for key in (
                "syntax_validity",
                "schema_validity",
                "semantic_consistency",
                "task_compliance",
            )
        ),
    }
    return {"gates": gates, "passed": all(gates.values())}
