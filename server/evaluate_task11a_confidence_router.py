"""Evaluate the frozen Task 11A confidence-aware taxonomy router."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

from evaluate_task10b_probe import _classifier, _fit, _split_arrays
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from task11a_confidence_router import (
    CONDITIONS,
    SEEDS,
    decide_seed,
    render_router_json,
    select_threshold,
    serialized_payload,
    temperature_scale,
    validate_router_json,
)


EXPECTED_BASE_FEATURE_SHA256 = "5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_completion(root: Path) -> None:
    completion = root / "completion.sha256"
    if not completion.is_file():
        raise ValueError(f"missing completion SHA256: {root}")
    for line in completion.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        target = root / relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"completion SHA256 mismatch: {target}")


def _fit_temperature(probabilities: np.ndarray, truth: np.ndarray, classes: np.ndarray) -> float:
    class_index = {int(label): index for index, label in enumerate(classes)}
    indices = np.asarray([class_index[int(value)] for value in truth], dtype=np.int64)

    def objective(log_temperature: float) -> float:
        scaled = temperature_scale(probabilities, float(np.exp(log_temperature)))
        return float(-np.log(np.clip(scaled[np.arange(len(indices)), indices], 1e-12, 1.0)).mean())

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("temperature fitting failed")
    return float(np.exp(result.x))


def _classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    rows: list[dict[str, Any]],
    labels: list[int],
) -> dict[str, Any]:
    bands = {}
    for band in ("head", "medium", "tail"):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["class_band"] == band],
            dtype=np.int64,
        )
        band_labels = sorted(
            {int(row["class_id"]) for row in rows if row["class_band"] == band}
        )
        bands[band] = float(
            f1_score(
                truth[indices],
                prediction[indices],
                labels=band_labels,
                average="macro",
                zero_division=0,
            )
        )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "band_macro_f1": bands,
    }


def _stress_arrays(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    split: str,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    stress_seed = seed if condition == "shuffle" else 0
    selected = [
        row
        for row in rows
        if row["split"] == split
        and row["condition"] == condition
        and int(row["stress_seed"]) == stress_seed
    ]
    if not selected:
        raise ValueError(f"missing stress rows: {split}/{condition}/{stress_seed}")
    indices = np.asarray([int(row["feature_index"]) for row in selected], dtype=np.int64)
    return matrix[indices], selected


def evaluate_seed(
    base_features: np.ndarray,
    base_rows: list[dict[str, Any]],
    stress_features: np.ndarray,
    stress_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    x_train, y_train, _ = _split_arrays(base_features, base_rows, "train")
    x_val, y_val, _ = _split_arrays(base_features, base_rows, "val")
    x_dev, y_dev, dev_rows = _split_arrays(base_features, base_rows, "dev")
    labels = sorted({int(value) for value in y_train})
    classifier = _classifier(seed)
    _fit(classifier, x_train, y_train)
    val_probabilities = classifier.predict_proba(x_val)
    temperature = _fit_temperature(val_probabilities, y_val, classifier.classes_)
    val_scaled = temperature_scale(val_probabilities, temperature)
    val_prediction = classifier.classes_[val_scaled.argmax(axis=1)]

    val_null_confidence = []
    for condition in CONDITIONS:
        values, _rows = _stress_arrays(
            stress_features, stress_rows, split="val", condition=condition, seed=seed
        )
        val_null_confidence.extend(
            temperature_scale(classifier.predict_proba(values), temperature).max(axis=1)
        )
    threshold_report = select_threshold(
        val_scaled.max(axis=1), val_prediction == y_val, np.asarray(val_null_confidence)
    )
    threshold = float(threshold_report["threshold"])

    dev_scaled = temperature_scale(classifier.predict_proba(x_dev), temperature)
    forced_prediction = classifier.classes_[dev_scaled.argmax(axis=1)].astype(np.int64)
    accepted = dev_scaled.max(axis=1) >= threshold
    confidence_prediction = np.where(accepted, forced_prediction, -1)
    forced_metrics = _classification_metrics(y_dev, forced_prediction, dev_rows, labels)
    confidence_metrics = _classification_metrics(
        y_dev, confidence_prediction, dev_rows, labels
    )
    confidence_metrics.update(
        {
            "coverage": float(accepted.mean()),
            "selective_accuracy": float(
                (forced_prediction[accepted] == y_dev[accepted]).mean()
            )
            if accepted.any()
            else 0.0,
            "macro_f1_delta": float(
                confidence_metrics["macro_f1"] - forced_metrics["macro_f1"]
            ),
        }
    )

    prediction_rows: list[dict[str, Any]] = []
    for row, truth, prediction, keep, confidence in zip(
        dev_rows,
        y_dev,
        forced_prediction,
        accepted,
        dev_scaled.max(axis=1),
        strict=True,
    ):
        payload = render_router_json(int(prediction) if keep else None)
        prediction_rows.append(
            {
                "id": str(row["id"]),
                "source_image_id": str(row["source_image_id"]),
                "condition": "original",
                "class_id": int(truth),
                "confidence": float(confidence),
                "accepted": bool(keep),
                "payload": serialized_payload(payload),
            }
        )

    null_metrics: dict[str, float] = {}
    all_null_acceptance = []
    for condition in CONDITIONS:
        values, condition_rows = _stress_arrays(
            stress_features, stress_rows, split="dev", condition=condition, seed=seed
        )
        scaled = temperature_scale(classifier.predict_proba(values), temperature)
        predictions = classifier.classes_[scaled.argmax(axis=1)].astype(np.int64)
        condition_accepted = scaled.max(axis=1) >= threshold
        null_metrics[f"{condition}_fpr"] = float(condition_accepted.mean())
        all_null_acceptance.extend(condition_accepted.tolist())
        for row, prediction, keep, confidence in zip(
            condition_rows,
            predictions,
            condition_accepted,
            scaled.max(axis=1),
            strict=True,
        ):
            payload = render_router_json(int(prediction) if keep else None)
            prediction_rows.append(
                {
                    "id": str(row["id"]),
                    "source_image_id": str(row["source_image_id"]),
                    "condition": condition,
                    "class_id": int(row["class_id"]),
                    "confidence": float(confidence),
                    "accepted": bool(keep),
                    "payload": serialized_payload(payload),
                }
            )
    null_metrics["overall_fpr"] = float(np.mean(all_null_acceptance))
    null_metrics["concrete_diagnosis_under_null"] = null_metrics["overall_fpr"]

    parsed = [json.loads(row["payload"]) for row in prediction_rows]
    valid = [validate_router_json(payload) for payload in parsed]
    contract = {
        "syntax_validity": 1.0,
        "schema_validity": float(np.mean(valid)),
        "semantic_consistency": float(np.mean(valid)),
        "task_compliance": float(np.mean(valid)),
    }
    metrics = {
        "seed": int(seed),
        "temperature": temperature,
        "threshold": threshold,
        "threshold_val_balanced_accuracy": threshold_report["balanced_accuracy"],
        "forced_original": forced_metrics,
        "confidence_original": confidence_metrics,
        "null": null_metrics,
        "json_contract": contract,
    }
    return {"metrics": metrics, "predictions": prediction_rows, "decision": decide_seed(metrics)}


def _bootstrap(seed_results: dict[int, dict[str, Any]], repetitions: int) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    source_ids = None
    for seed, result in seed_results.items():
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for row in result["predictions"]:
            grouped.setdefault(str(row["source_image_id"]), {})[str(row["condition"])] = row
        by_seed[seed] = grouped
        current = sorted(grouped)
        source_ids = current if source_ids is None else source_ids
        if current != source_ids or any(set(values) != {"original", *CONDITIONS} for values in grouped.values()):
            raise ValueError("bootstrap requires complete paired source conditions")
    assert source_ids is not None
    labels = sorted(
        {
            int(values["original"]["class_id"])
            for values in by_seed[SEEDS[0]].values()
        }
    )

    def statistic(indices: list[int]) -> dict[str, float]:
        deltas = []
        fprs = {condition: [] for condition in CONDITIONS}
        for seed in SEEDS:
            selected = [source_ids[index] for index in indices]
            truth = np.asarray(
                [by_seed[seed][source]["original"]["class_id"] for source in selected]
            )
            forced = np.asarray(
                [
                    json.loads(by_seed[seed][source]["original"]["payload"])["diagnosis"]["pest_id"]
                    if by_seed[seed][source]["original"]["accepted"]
                    else -1
                    for source in selected
                ]
            )
            # The delta is already reported exactly per seed; bootstrap focuses on retained F1.
            deltas.append(float(f1_score(truth, forced, labels=labels, average="macro", zero_division=0)))
            for condition in CONDITIONS:
                fprs[condition].append(
                    float(
                        np.mean(
                            [by_seed[seed][source][condition]["accepted"] for source in selected]
                        )
                    )
                )
        return {
            "confidence_macro_f1": mean(deltas),
            **{f"{condition}_fpr": mean(values) for condition, values in fprs.items()},
        }

    observed = statistic(list(range(len(source_ids))))
    rng = random.Random(20260723)
    samples = [
        statistic([rng.randrange(len(source_ids)) for _ in source_ids])
        for _ in range(repetitions)
    ]
    return {
        key: {
            "estimate": observed[key],
            "low": float(np.quantile([sample[key] for sample in samples], 0.025)),
            "high": float(np.quantile([sample[key] for sample in samples], 0.975)),
            "repetitions": repetitions,
            "unit": "source_image_id",
        }
        for key in observed
    }


def run_evaluation(
    *,
    base_feature_root: Path,
    stress_feature_root: Path,
    output_root: Path,
    repetitions: int = 1000,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    status = destination / "status.json"
    status.write_text('{"state":"running","stage":"verify"}\n', encoding="utf-8")
    try:
        base_root = Path(base_feature_root)
        stress_root = Path(stress_feature_root)
        _verify_completion(base_root)
        _verify_completion(stress_root)
        if sha256_file(base_root / "features.npy") != EXPECTED_BASE_FEATURE_SHA256:
            raise ValueError("unexpected Task 10B base feature SHA256")
        base_summary = _read_json(base_root / "run_summary.json")
        stress_summary = _read_json(stress_root / "run_summary.json")
        if base_summary["manifest_sha256"] != stress_summary["manifest_sha256"]:
            raise ValueError("base/stress source manifest SHA256 mismatch")
        base_features = np.load(base_root / "features.npy", allow_pickle=False)
        stress_features = np.load(stress_root / "features.npy", allow_pickle=False)
        base_rows = _read_jsonl(base_root / "feature_rows.jsonl")
        stress_rows = _read_jsonl(stress_root / "feature_rows.jsonl")
        if base_features.shape != (320, 2048) or stress_features.shape != (640, 2048):
            raise ValueError("unexpected Task 11A feature shape")
        seed_results = {
            seed: evaluate_seed(base_features, base_rows, stress_features, stress_rows, seed)
            for seed in SEEDS
        }
        signed = []
        compact = {}
        for seed, result in seed_results.items():
            predictions_name = f"seed_{seed}_predictions.jsonl"
            metrics_name = f"seed_{seed}_metrics.json"
            with (destination / predictions_name).open("x", encoding="utf-8", newline="\n") as handle:
                for row in result["predictions"]:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            write_json_new(destination / metrics_name, {"metrics": result["metrics"], "decision": result["decision"]})
            signed.extend([predictions_name, metrics_name])
            compact[str(seed)] = {"metrics": result["metrics"], "decision": result["decision"]}
        bootstrap = _bootstrap(seed_results, repetitions)
        decision = {
            "version": "task11a-confidence-router-decision-1",
            "status": "PASS" if all(result["decision"]["passed"] for result in seed_results.values()) else "FAIL",
            "all_seeds_passed": all(result["decision"]["passed"] for result in seed_results.values()),
            "seed_decisions": {str(seed): result["decision"] for seed, result in seed_results.items()},
            "authorize_task11b_planning": all(result["decision"]["passed"] for result in seed_results.values()),
            "authorize_large_training": False,
        }
        report = {"version": "task11a-confidence-router-evaluation-1", "seed_results": compact, "bootstrap": bootstrap, "decision": decision}
        write_json_new(destination / "metrics.json", report)
        write_json_new(destination / "task11a_decision_report.json", decision)
        write_json_new(
            destination / "run_summary.json",
            {
                "state": "completed",
                "base_features_sha256": sha256_file(base_root / "features.npy"),
                "stress_features_sha256": sha256_file(stress_root / "features.npy"),
                "source_manifest_sha256": base_summary["manifest_sha256"],
                "stress_feature_count": len(stress_rows),
                "seeds": list(SEEDS),
                "bootstrap_repetitions": repetitions,
                "decision": decision["status"],
            },
        )
        signed.extend(["metrics.json", "task11a_decision_report.json", "run_summary.json"])
        with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed:
                handle.write(f"{sha256_file(destination / name)}  {name}\n")
        status.write_text('{"state":"completed","stage":"done"}\n', encoding="utf-8")
        return report
    except Exception as exc:
        write_json_new(destination / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        status.write_text('{"state":"failed","stage":"evaluation"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 11A confidence router")
    parser.add_argument("--base-feature-root", type=Path, required=True)
    parser.add_argument("--stress-feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    arguments = parser.parse_args()
    report = run_evaluation(
        base_feature_root=arguments.base_feature_root,
        stress_feature_root=arguments.stress_feature_root,
        output_root=arguments.output_root,
        repetitions=arguments.repetitions,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
