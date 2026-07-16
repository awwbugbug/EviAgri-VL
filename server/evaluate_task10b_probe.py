"""Preregistered linear-probe evaluation for Task 10B v2."""

from __future__ import annotations

import argparse
import json
import random
import traceback
import warnings
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


SEEDS = (17, 29, 43)


def _split_arrays(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    selected = [row for row in rows if str(row.get("split")) == split]
    if not selected:
        raise ValueError(f"missing Task 10B split: {split}")
    indices = np.asarray([int(row["feature_index"]) for row in selected], dtype=np.int64)
    labels = np.asarray([int(row["class_id"]) for row in selected], dtype=np.int64)
    return features[indices], labels, selected


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    rows: list[dict[str, Any]],
    all_labels: list[int],
) -> dict[str, Any]:
    bands = {}
    for band in ("head", "medium", "tail"):
        indices = [index for index, row in enumerate(rows) if row.get("class_band") == band]
        band_labels = sorted(
            {int(row["class_id"]) for row in rows if row.get("class_band") == band}
        )
        bands[band] = (
            float(
                f1_score(
                    labels[indices],
                    predictions[indices],
                    labels=band_labels,
                    average="macro",
                    zero_division=0,
                )
            )
            if indices and band_labels
            else 0.0
        )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "band_macro_f1": bands,
        "count": int(labels.shape[0]),
    }


def _classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )


def _fit(classifier: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="scipy.optimize: The `disp` and `iprint` options",
            category=DeprecationWarning,
        )
        classifier.fit(features, labels)


def evaluate_seed(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("feature matrix/row mismatch")
    if not np.isfinite(features).all():
        raise ValueError("non-finite Task 10B features")
    x_train, y_train, _train_rows = _split_arrays(features, rows, "train")
    x_val, y_val, val_rows = _split_arrays(features, rows, "val")
    x_dev, y_dev, dev_rows = _split_arrays(features, rows, "dev")
    labels = sorted({int(value) for value in y_train})
    if set(labels) != set(int(value) for value in y_dev):
        raise ValueError("train/dev class identity mismatch")

    classifier = _classifier(seed)
    _fit(classifier, x_train, y_train)
    val_predictions = classifier.predict(x_val)
    dev_predictions = classifier.predict(x_dev)

    rng = np.random.default_rng(seed)
    permuted = _classifier(seed)
    _fit(permuted, x_train, rng.permutation(y_train))
    permutation_predictions = permuted.predict(x_dev)

    no_image = DummyClassifier(strategy="stratified", random_state=seed)
    no_image.fit(np.zeros((len(y_train), 1), dtype=np.float32), y_train)
    no_image_predictions = no_image.predict(
        np.zeros((len(y_dev), 1), dtype=np.float32)
    )

    dev_metrics = _metrics(y_dev, dev_predictions, dev_rows, labels)
    val_metrics = _metrics(y_val, val_predictions, val_rows, labels)
    permutation_macro = float(
        f1_score(
            y_dev,
            permutation_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    no_image_macro = float(
        f1_score(
            y_dev,
            no_image_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    prediction_rows = [
        {
            "id": str(row["id"]),
            "split": "dev",
            "class_id": int(truth),
            "prediction": int(prediction),
            "correct": bool(int(truth) == int(prediction)),
        }
        for row, truth, prediction in zip(dev_rows, y_dev, dev_predictions, strict=True)
    ]
    return {
        "seed": int(seed),
        "classifier_config": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": int(seed),
            "solver": "lbfgs",
        },
        "train_count": int(len(y_train)),
        "val": val_metrics,
        "dev": dev_metrics,
        "permutation_control_dev_macro_f1": permutation_macro,
        "no_image_dev_macro_f1": no_image_macro,
        "visual_gain_macro_f1": dev_metrics["macro_f1"] - no_image_macro,
        "predictions": prediction_rows,
    }


def bootstrap_pooled_macro_f1(
    rows: list[dict[str, Any]],
    predictions_by_seed: Mapping[int, Mapping[str, int]],
    repetitions: int = 1000,
    seed: int = 20260717,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    dev_rows = [row for row in rows if str(row.get("split")) == "dev"]
    if not dev_rows or len({str(row["id"]) for row in dev_rows}) != len(dev_rows):
        raise ValueError("bootstrap requires unique non-empty dev source IDs")
    labels = sorted({int(row["class_id"]) for row in dev_rows})
    truth = np.asarray([int(row["class_id"]) for row in dev_rows], dtype=np.int64)
    predictions = {}
    for model_seed, mapping in predictions_by_seed.items():
        if any(str(row["id"]) not in mapping for row in dev_rows):
            raise ValueError(f"missing dev prediction for seed {model_seed}")
        predictions[int(model_seed)] = np.asarray(
            [int(mapping[str(row["id"])]) for row in dev_rows], dtype=np.int64
        )

    def pooled(indices: np.ndarray) -> float:
        return float(
            mean(
                f1_score(
                    truth[indices],
                    values[indices],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
                for values in predictions.values()
            )
        )

    observed = pooled(np.arange(len(dev_rows), dtype=np.int64))
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        indices = np.asarray(
            [rng.randrange(len(dev_rows)) for _row in dev_rows], dtype=np.int64
        )
        samples.append(pooled(indices))
    return {
        "estimate": observed,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "unit": "source_image_id",
    }


def decide_task10b(
    seed_metrics: Mapping[int, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    *,
    overlap: Mapping[str, int],
) -> dict[str, Any]:
    if sorted(int(seed) for seed in seed_metrics) != list(SEEDS):
        raise ValueError("Task 10B decision requires seeds 17/29/43")
    macro_values = [
        float(seed_metrics[seed]["dev"]["macro_f1"]) for seed in SEEDS
    ]
    permutation_values = [
        float(seed_metrics[seed]["permutation_control_dev_macro_f1"])
        for seed in SEEDS
    ]
    visual_gain_values = [
        float(seed_metrics[seed]["visual_gain_macro_f1"]) for seed in SEEDS
    ]
    macro_mean = mean(macro_values)
    permutation_mean = mean(permutation_values)
    conditions = {
        "mean_macro_f1_ge_0_25": macro_mean >= 0.25,
        "worst_seed_macro_f1_ge_0_20": min(macro_values) >= 0.20,
        "bootstrap_low_gt_0_125": float(bootstrap["low"]) > 0.125,
        "permutation_mean_le_0_10": permutation_mean <= 0.10,
        "zero_split_overlap": all(int(value) == 0 for value in overlap.values()),
    }
    passed = all(conditions.values())
    return {
        "version": "task10b-v2-decision-1",
        "status": "PASS" if passed else "FAIL",
        "conditions": conditions,
        "mean_macro_f1": macro_mean,
        "sample_std_macro_f1": stdev(macro_values),
        "worst_seed_macro_f1": min(macro_values),
        "permutation_control_mean_macro_f1": permutation_mean,
        "mean_visual_gain_macro_f1": mean(visual_gain_values),
        "pooled_bootstrap": dict(bootstrap),
        "overlap": dict(overlap),
        "authorize_task10c_planning": passed,
        "authorize_task10c_execution": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _write_json_replace(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_feature_completion(feature_root: Path) -> None:
    completion = Path(feature_root) / "completion.sha256"
    if not completion.is_file():
        raise ValueError("missing feature completion SHA256")
    observed = set()
    for line in completion.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        target = Path(feature_root) / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"feature completion SHA256 mismatch: {relative}")
        observed.add(relative)
    expected_files = {
        "features.npy",
        "feature_rows.jsonl",
        "config.snapshot.json",
        "run_summary.json",
    }
    if observed != expected_files:
        raise ValueError("unexpected feature completion contract")


def _split_overlap(rows: list[dict[str, Any]], key: str) -> int:
    values: dict[str, set[str]] = {}
    for split in ("train", "val", "dev"):
        values[split] = {
            str(row[key]) for row in rows if row.get("split") == split and row.get(key)
        }
    return len(
        (values["train"] & values["val"])
        | (values["train"] & values["dev"])
        | (values["val"] & values["dev"])
    )


def run_evaluation(
    *,
    feature_root: Path,
    output_root: Path,
    repetitions: int = 1000,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    _write_json_replace(destination / "status.json", {"state": "running", "stage": "verify"})
    try:
        root = Path(feature_root)
        _verify_feature_completion(root)
        matrix = np.load(root / "features.npy", allow_pickle=False)
        rows = _read_jsonl(root / "feature_rows.jsonl")
        summary = _read_json(root / "run_summary.json")
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(rows)
            or int(summary.get("feature_count", -1)) != len(rows)
            or summary.get("features_sha256") != sha256_file(root / "features.npy")
            or summary.get("feature_rows_sha256") != sha256_file(root / "feature_rows.jsonl")
        ):
            raise ValueError("feature summary/cardinality/hash mismatch")
        if sorted(int(row["feature_index"]) for row in rows) != list(range(len(rows))):
            raise ValueError("feature row indices are not a permutation of matrix rows")

        overlap = {
            "source_image_sha256": _split_overlap(rows, "source_image_sha256"),
            "near_duplicate_component": _split_overlap(
                rows, "near_duplicate_component_id"
            ),
        }
        seed_results = {seed: evaluate_seed(matrix, rows, seed) for seed in SEEDS}
        predictions_by_seed = {
            seed: {
                str(row["id"]): int(row["prediction"])
                for row in result["predictions"]
            }
            for seed, result in seed_results.items()
        }
        bootstrap = bootstrap_pooled_macro_f1(
            rows,
            predictions_by_seed,
            repetitions=repetitions,
            seed=20260717,
        )
        decision = decide_task10b(seed_results, bootstrap, overlap=overlap)
        compact_metrics = {}
        signed = []
        for seed, result in seed_results.items():
            predictions_name = f"seed_{seed}_predictions.jsonl"
            metrics_name = f"seed_{seed}_metrics.json"
            _write_jsonl_new(destination / predictions_name, result["predictions"])
            compact = {key: value for key, value in result.items() if key != "predictions"}
            write_json_new(destination / metrics_name, compact)
            compact_metrics[str(seed)] = compact
            signed.extend([predictions_name, metrics_name])
        write_json_new(destination / "task10b_decision_report.json", decision)
        report = {
            "version": "task10b-v2-evaluation-1",
            "seed_metrics": compact_metrics,
            "bootstrap": bootstrap,
            "overlap": overlap,
            "decision": decision,
        }
        write_json_new(destination / "metrics.json", report)
        run_summary = {
            "version": "task10b-v2-evaluation-summary-1",
            "state": "completed",
            "feature_root": str(root),
            "features_sha256": sha256_file(root / "features.npy"),
            "feature_rows_sha256": sha256_file(root / "feature_rows.jsonl"),
            "feature_count": len(rows),
            "seeds": list(SEEDS),
            "bootstrap_repetitions": repetitions,
            "decision": decision["status"],
        }
        write_json_new(destination / "run_summary.json", run_summary)
        signed.extend(["task10b_decision_report.json", "metrics.json", "run_summary.json"])
        with (destination / "completion.sha256").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            for name in signed:
                handle.write(f"{sha256_file(destination / name)}  {name}\n")
        _write_json_replace(destination / "status.json", {"state": "completed", "stage": "done"})
        return report
    except Exception as exc:
        write_json_new(
            destination / "failure.json",
            {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()},
        )
        _write_json_replace(destination / "status.json", {"state": "failed", "stage": "evaluation"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Task 10B linear probe")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    arguments = parser.parse_args()
    report = run_evaluation(
        feature_root=arguments.feature_root,
        output_root=arguments.output_root,
        repetitions=arguments.repetitions,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
