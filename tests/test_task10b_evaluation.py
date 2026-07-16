import sys
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task10b_probe import (
    bootstrap_pooled_macro_f1,
    decide_task10b,
    evaluate_seed,
    run_evaluation,
)
from extract_task10b_features import write_feature_outputs


def _separable_rows_and_features():
    rows = []
    features = []
    bands = {0: "head", 1: "medium", 2: "tail"}
    for split, count in (("train", 12), ("val", 3), ("dev", 5)):
        for class_id in range(3):
            for index in range(count):
                vector = np.zeros(3, dtype=np.float32)
                vector[class_id] = 1.0
                rows.append(
                    {
                        "id": f"{split}-{class_id}-{index}",
                        "split": split,
                        "class_id": class_id,
                        "class_band": bands[class_id],
                        "feature_index": len(features),
                    }
                )
                features.append(vector)
    return np.asarray(features, dtype=np.float32), rows


def test_evaluate_seed_uses_frozen_classifier_and_reports_controls():
    features, rows = _separable_rows_and_features()

    result = evaluate_seed(features, rows, seed=17)

    assert result["classifier_config"] == {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 17,
        "solver": "lbfgs",
    }
    assert result["dev"]["accuracy"] == pytest.approx(1.0)
    assert result["dev"]["macro_f1"] == pytest.approx(1.0)
    assert result["dev"]["band_macro_f1"] == {
        "head": pytest.approx(1.0),
        "medium": pytest.approx(1.0),
        "tail": pytest.approx(1.0),
    }
    assert 0.0 <= result["permutation_control_dev_macro_f1"] <= 1.0
    assert 0.0 <= result["no_image_dev_macro_f1"] <= 1.0
    assert result["visual_gain_macro_f1"] == pytest.approx(
        result["dev"]["macro_f1"] - result["no_image_dev_macro_f1"]
    )
    assert len(result["predictions"]) == 15
    assert {row["split"] for row in result["predictions"]} == {"dev"}


def test_label_permutation_is_deterministic_per_seed():
    features, rows = _separable_rows_and_features()

    first = evaluate_seed(features, rows, seed=29)
    second = evaluate_seed(features, rows, seed=29)

    assert first == second


def test_bootstrap_pooled_macro_f1_is_deterministic_and_family_based():
    rows = [
        {"id": f"d{i}", "split": "dev", "class_id": i % 2}
        for i in range(20)
    ]
    predictions = {
        17: {row["id"]: row["class_id"] for row in rows},
        29: {row["id"]: row["class_id"] for row in rows},
        43: {row["id"]: 0 for row in rows},
    }

    first = bootstrap_pooled_macro_f1(rows, predictions, repetitions=1000, seed=20260717)
    second = bootstrap_pooled_macro_f1(rows, predictions, repetitions=1000, seed=20260717)

    assert first == second
    assert first["repetitions"] == 1000
    assert first["unit"] == "source_image_id"
    assert 0.0 <= first["low"] <= first["estimate"] <= first["high"] <= 1.0


def _seed_metric(macro_f1: float, permutation: float = 0.08):
    return {
        "dev": {"macro_f1": macro_f1},
        "permutation_control_dev_macro_f1": permutation,
        "visual_gain_macro_f1": 0.20,
    }


def test_decide_task10b_applies_all_preregistered_pass_conditions():
    metrics = {
        17: _seed_metric(0.31),
        29: _seed_metric(0.29),
        43: _seed_metric(0.27),
    }
    bootstrap = {"estimate": 0.29, "low": 0.13, "high": 0.42, "repetitions": 1000}

    decision = decide_task10b(
        metrics,
        bootstrap,
        overlap={"source_image_sha256": 0, "near_duplicate_component": 0},
    )

    assert decision["status"] == "PASS"
    assert decision["authorize_task10c_planning"] is True
    assert decision["authorize_task10c_execution"] is False
    assert decision["mean_macro_f1"] == pytest.approx(0.29)
    assert decision["worst_seed_macro_f1"] == pytest.approx(0.27)


@pytest.mark.parametrize(
    "metrics, bootstrap, overlap, failed_condition",
    [
        (
            {17: _seed_metric(0.24), 29: _seed_metric(0.24), 43: _seed_metric(0.24)},
            {"estimate": 0.24, "low": 0.13, "high": 0.35},
            {"source_image_sha256": 0, "near_duplicate_component": 0},
            "mean_macro_f1_ge_0_25",
        ),
        (
            {17: _seed_metric(0.30), 29: _seed_metric(0.28), 43: _seed_metric(0.19)},
            {"estimate": 0.26, "low": 0.13, "high": 0.35},
            {"source_image_sha256": 0, "near_duplicate_component": 0},
            "worst_seed_macro_f1_ge_0_20",
        ),
        (
            {17: _seed_metric(0.30), 29: _seed_metric(0.28), 43: _seed_metric(0.27)},
            {"estimate": 0.28, "low": 0.12, "high": 0.35},
            {"source_image_sha256": 0, "near_duplicate_component": 0},
            "bootstrap_low_gt_0_125",
        ),
        (
            {
                17: _seed_metric(0.30, 0.11),
                29: _seed_metric(0.28, 0.11),
                43: _seed_metric(0.27, 0.11),
            },
            {"estimate": 0.28, "low": 0.13, "high": 0.35},
            {"source_image_sha256": 0, "near_duplicate_component": 0},
            "permutation_mean_le_0_10",
        ),
        (
            {17: _seed_metric(0.30), 29: _seed_metric(0.28), 43: _seed_metric(0.27)},
            {"estimate": 0.28, "low": 0.13, "high": 0.35},
            {"source_image_sha256": 1, "near_duplicate_component": 0},
            "zero_split_overlap",
        ),
    ],
)
def test_decide_task10b_fails_each_gate(metrics, bootstrap, overlap, failed_condition):
    decision = decide_task10b(metrics, bootstrap, overlap=overlap)

    assert decision["status"] == "FAIL"
    assert decision["conditions"][failed_condition] is False
    assert decision["authorize_task10c_planning"] is False
    assert decision["authorize_task10c_execution"] is False


def test_run_evaluation_writes_signed_three_seed_outputs_and_refuses_overwrite(tmp_path):
    features, rows = _separable_rows_and_features()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    feature_root = tmp_path / "features"
    write_feature_outputs(
        matrix=features,
        feature_rows=rows,
        output_root=feature_root,
        manifest_path=manifest,
        config={"min_pixels": 200704, "max_pixels": 401408},
        model_identity={"config.json": "a" * 64},
    )
    output = tmp_path / "evaluation"

    report = run_evaluation(feature_root=feature_root, output_root=output, repetitions=100)

    assert set(report["seed_metrics"]) == {"17", "29", "43"}
    assert report["bootstrap"]["repetitions"] == 100
    assert (output / "task10b_decision_report.json").is_file()
    assert (output / "completion.sha256").is_file()
    for seed in (17, 29, 43):
        assert (output / f"seed_{seed}_predictions.jsonl").is_file()
        assert (output / f"seed_{seed}_metrics.json").is_file()
    with pytest.raises(FileExistsError):
        run_evaluation(feature_root=feature_root, output_root=output, repetitions=100)
