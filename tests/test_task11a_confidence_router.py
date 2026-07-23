import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task11a_confidence_router import (
    build_stress_rows,
    decide_seed,
    render_router_json,
    select_threshold,
    temperature_scale,
    transform_image,
    validate_router_json,
)


def test_build_stress_rows_separates_val_and_dev_parameters():
    rows = [
        {"id": "t", "split": "train"},
        {"id": "v", "split": "val"},
        {"id": "d", "split": "dev"},
    ]
    stress = build_stress_rows(rows)
    assert len(stress) == 10
    assert {row["split"] for row in stress} == {"val", "dev"}
    assert sum(row["condition"] == "shuffle" for row in stress) == 6


def test_transform_image_is_deterministic_and_split_specific():
    image = Image.fromarray(np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3))
    val_blank = transform_image(
        image, condition="blank", split="val", row_id="x", seed=0
    )
    dev_blank = transform_image(
        image, condition="blank", split="dev", row_id="x", seed=0
    )
    assert np.asarray(val_blank)[0, 0, 0] == 127
    assert np.asarray(dev_blank)[0, 0, 0] == 114
    first = transform_image(
        image, condition="shuffle", split="dev", row_id="x", seed=17
    )
    second = transform_image(
        image, condition="shuffle", split="dev", row_id="x", seed=17
    )
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_temperature_scale_and_threshold_are_finite_and_deterministic():
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6]])
    scaled = temperature_scale(probabilities, 2.0)
    assert np.allclose(scaled.sum(axis=1), 1.0)
    result = select_threshold(
        np.asarray([0.9, 0.4]),
        np.asarray([True, False]),
        np.asarray([0.2, 0.3]),
    )
    assert result["threshold"] == pytest.approx(0.41)
    assert result["balanced_accuracy"] == 1.0


def test_router_renderer_is_semantically_closed():
    accepted = render_router_json(9)
    refused = render_router_json(None)
    assert validate_router_json(accepted)
    assert validate_router_json(refused)
    broken = dict(refused)
    broken["diagnosis"] = {"pest_id": 9}
    assert not validate_router_json(broken)


def test_decision_is_fail_closed():
    metrics = {
        "confidence_original": {"macro_f1_delta": -0.02, "coverage": 0.8},
        "null": {"blank_fpr": 0.0, "blur_fpr": 0.09, "shuffle_fpr": 0.2},
        "json_contract": {
            "syntax_validity": 1.0,
            "schema_validity": 1.0,
            "semantic_consistency": 1.0,
            "task_compliance": 1.0,
        },
    }
    assert decide_seed(metrics)["passed"] is True
    metrics["null"]["blur_fpr"] = 0.10
    assert decide_seed(metrics)["passed"] is False
