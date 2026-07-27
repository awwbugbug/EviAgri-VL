"""Evaluate the frozen G/R/GG/GR Task14A oracle tournament."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from evaluate_task10b_probe import _classifier, _fit
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


SEEDS = (17, 29, 43)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_completion(root: Path) -> None:
    lines = (Path(root) / "completion.sha256").read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty completion manifest")
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        target = Path(root) / relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"completion SHA256 mismatch: {target}")


def paired_bootstrap(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    positive_count: int,
    null_count: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if min(positive_count, null_count, repetitions) <= 0:
        raise ValueError("invalid paired bootstrap size")
    positive = np.arange(positive_count)
    null = np.arange(null_count)
    estimate = float(statistic(positive, null))
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        p = np.asarray([rng.randrange(positive_count) for _ in range(positive_count)])
        n = np.asarray([rng.randrange(null_count) for _ in range(null_count)])
        samples.append(float(statistic(p, n)))
    return {
        "estimate": estimate,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "repetitions": repetitions,
        "unit": "fresh_image",
    }


def decide(primary_delta: dict[str, float], mode: str = "oracle") -> dict[str, Any]:
    gain = primary_delta["accuracy"] >= 1 / 32 and primary_delta["true_probability"] > 0
    safe = primary_delta["null_confidence"] <= 0 and primary_delta["confidence_auroc"] >= 0
    if mode == "replication":
        decision = "H2_REPLICATION_INCONSISTENT" if gain and safe else "H2_MEAN_REGION_RETIRED"
        authorize_tiny = False
    elif mode != "oracle":
        raise ValueError("unknown Task14 decision mode")
    elif gain and safe:
        decision = "H2_ORACLE_SUPPORTED"
        authorize_tiny = True
    elif gain:
        decision = "H2_ORACLE_UNSAFE"
        authorize_tiny = False
    else:
        decision = "H2_ORACLE_NO_GAIN"
        authorize_tiny = False
    return {
        "decision": decision,
        "positive_gain": gain,
        "reliability_safe": safe,
        "authorize_tiny_learned_selector": authorize_tiny,
        "authorize_large_training": False,
        "authorize_task8": False,
    }


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def run(
    *,
    protocol_root: Path,
    feature_root: Path,
    output_root: Path,
    repetitions: int = 1000,
    train_per_class: int = 4,
    val_per_class: int = 1,
    test_per_class: int = 2,
    null_count: int = 32,
    decision_mode: str = "oracle",
    report_version: str = "task14a-token-oracle-evaluation-1",
) -> dict[str, Any]:
    protocol = Path(protocol_root)
    features = Path(feature_root)
    output = Path(output_root)
    ensure_new_directory(output)
    (output / "status.json").write_text('{"state":"running"}\n', encoding="utf-8")
    try:
        class_count = 16
        if test_per_class * class_count != 32 or null_count != 32:
            raise ValueError("frozen Task14 decision gates require 32 positive and 32 null test images")
        verify_completion(protocol)
        verify_completion(features)
        manifest_sha = sha256_file(protocol / "manifest.jsonl")
        feature_summary = json.loads((features / "run_summary.json").read_text(encoding="utf-8"))
        if feature_summary.get("manifest_sha256") != manifest_sha:
            raise ValueError("feature/protocol manifest mismatch")
        if feature_summary.get("all_parameters_frozen") is not True or feature_summary.get("full_frame_encodings_per_image") != 1:
            raise ValueError("Task14A frozen single-encoding contract failed")

        global_values = np.load(features / "global_features.npy", allow_pickle=False)
        region_values = np.load(features / "region_features.npy", allow_pickle=False)
        rows = read_jsonl(features / "feature_rows.jsonl")
        expected_total = class_count * (train_per_class + val_per_class + test_per_class) + null_count
        if global_values.shape != (expected_total, 2048) or region_values.shape != (expected_total, 2048) or len(rows) != expected_total:
            raise ValueError("unexpected Task14A feature contract")
        if not np.isfinite(global_values).all() or not np.isfinite(region_values).all():
            raise ValueError("non-finite Task14A features")
        if [int(row["feature_index"]) for row in rows] != list(range(144)):
            raise ValueError("feature row order mismatch")

        split_names = ("probe_train", "probe_val", "probe_test", "null_test")
        masks = {split: np.asarray([str(row["probe_split"]) == split for row in rows]) for split in split_names}
        expected_counts = {
            "probe_train": class_count * train_per_class,
            "probe_val": class_count * val_per_class,
            "probe_test": class_count * test_per_class,
            "null_test": null_count,
        }
        if {split: int(mask.sum()) for split, mask in masks.items()} != expected_counts:
            raise ValueError("Task14A split cardinality mismatch")
        labels = np.asarray([int(row["class_id"]) if row["class_id"] is not None else -1 for row in rows])
        classes = sorted(set(labels[masks["probe_train"]].tolist()))
        if len(classes) != 16:
            raise ValueError("Task14A training class count mismatch")
        for class_id in classes:
            counts = [int(((labels == class_id) & masks[split]).sum()) for split in ("probe_train", "probe_val", "probe_test")]
            if counts != [train_per_class, val_per_class, test_per_class]:
                raise ValueError(f"Task14A per-class quota mismatch: {class_id}")

        representations = {
            "G": global_values,
            "R": region_values,
            "GG": np.concatenate([global_values, global_values], axis=1) / np.sqrt(2.0),
            "GR": np.concatenate([global_values, region_values], axis=1) / np.sqrt(2.0),
        }
        per_seed: dict[str, Any] = {}
        predictions: list[dict[str, Any]] = []
        paired: dict[str, list[np.ndarray]] = {
            "correct": [],
            "true_probability": [],
            "null_confidence": [],
            "gg_positive_confidence": [],
            "gr_positive_confidence": [],
            "gg_null_confidence": [],
            "gr_null_confidence": [],
        }
        test = masks["probe_test"]
        null = masks["null_test"]
        for seed in SEEDS:
            seed_metrics: dict[str, Any] = {}
            raw: dict[str, dict[str, np.ndarray]] = {}
            for condition, matrix in representations.items():
                classifier = _classifier(seed)
                _fit(classifier, matrix[masks["probe_train"]], labels[masks["probe_train"]])
                probabilities = classifier.predict_proba(matrix)
                forced = classifier.classes_[probabilities.argmax(axis=1)].astype(np.int64)
                confidence = probabilities.max(axis=1)
                class_index = {int(value): index for index, value in enumerate(classifier.classes_)}
                test_indices = np.where(test)[0]
                true_probability = np.asarray([probabilities[index, class_index[int(labels[index])]] for index in test_indices])
                seed_metrics[condition] = {
                    "accuracy": float(accuracy_score(labels[test], forced[test])),
                    "macro_f1": float(f1_score(labels[test], forced[test], labels=classes, average="macro", zero_division=0)),
                    "mean_true_probability": float(true_probability.mean()),
                    "mean_null_confidence": float(confidence[null].mean()),
                    "confidence_auroc": float(roc_auc_score(np.r_[np.ones(test.sum()), np.zeros(null.sum())], np.r_[confidence[test], confidence[null]])),
                    "validation_accuracy": (
                        None
                        if not masks["probe_val"].any()
                        else float(accuracy_score(labels[masks["probe_val"]], forced[masks["probe_val"]]))
                    ),
                }
                raw[condition] = {"forced": forced, "confidence": confidence, "true_probability": true_probability}
                for index, row in enumerate(rows):
                    predictions.append(
                        {
                            "id": str(row["id"]),
                            "probe_split": str(row["probe_split"]),
                            "target_type": str(row["target_type"]),
                            "true_class_id": None if labels[index] < 0 else int(labels[index]),
                            "seed": seed,
                            "condition": condition,
                            "predicted_class_id": int(forced[index]),
                            "confidence": float(confidence[index]),
                            "true_class_probability": None if labels[index] < 0 else float(probabilities[index, class_index[int(labels[index])]]),
                        }
                    )
            gg = raw["GG"]
            gr = raw["GR"]
            paired["correct"].append((gr["forced"][test] == labels[test]).astype(float) - (gg["forced"][test] == labels[test]).astype(float))
            paired["true_probability"].append(gr["true_probability"] - gg["true_probability"])
            paired["null_confidence"].append(gr["confidence"][null] - gg["confidence"][null])
            paired["gg_positive_confidence"].append(gg["confidence"][test])
            paired["gr_positive_confidence"].append(gr["confidence"][test])
            paired["gg_null_confidence"].append(gg["confidence"][null])
            paired["gr_null_confidence"].append(gr["confidence"][null])
            g_correct = raw["G"]["forced"][test] == labels[test]
            r_correct = raw["R"]["forced"][test] == labels[test]
            seed_metrics["G_R_complementarity"] = {
                "G_wrong_R_right": int((~g_correct & r_correct).sum()),
                "G_right_R_wrong": int((g_correct & ~r_correct).sum()),
                "both_right": int((g_correct & r_correct).sum()),
                "both_wrong": int((~g_correct & ~r_correct).sum()),
            }
            per_seed[str(seed)] = seed_metrics
            write_json_new(output / f"seed_{seed}_metrics.json", seed_metrics)

        arrays = {key: np.stack(value, axis=1) for key, value in paired.items()}
        primary_delta = {
            "accuracy": mean(per_seed[str(seed)]["GR"]["accuracy"] - per_seed[str(seed)]["GG"]["accuracy"] for seed in SEEDS),
            "macro_f1": mean(per_seed[str(seed)]["GR"]["macro_f1"] - per_seed[str(seed)]["GG"]["macro_f1"] for seed in SEEDS),
            "true_probability": float(arrays["true_probability"].mean()),
            "null_confidence": float(arrays["null_confidence"].mean()),
            "confidence_auroc": mean(per_seed[str(seed)]["GR"]["confidence_auroc"] - per_seed[str(seed)]["GG"]["confidence_auroc"] for seed in SEEDS),
        }
        bootstrap = {
            "accuracy_delta": paired_bootstrap(lambda p, n: float(arrays["correct"][p].mean()), expected_counts["probe_test"], null_count, repetitions, 20260727),
            "true_probability_delta": paired_bootstrap(lambda p, n: float(arrays["true_probability"][p].mean()), expected_counts["probe_test"], null_count, repetitions, 20260728),
            "null_confidence_delta": paired_bootstrap(lambda p, n: float(arrays["null_confidence"][n].mean()), expected_counts["probe_test"], null_count, repetitions, 20260729),
            "confidence_auroc_delta": paired_bootstrap(
                lambda p, n: float(
                    roc_auc_score(
                        np.r_[np.ones(len(p) * len(SEEDS)), np.zeros(len(n) * len(SEEDS))],
                        np.r_[arrays["gr_positive_confidence"][p].ravel(), arrays["gr_null_confidence"][n].ravel()],
                    )
                    - roc_auc_score(
                        np.r_[np.ones(len(p) * len(SEEDS)), np.zeros(len(n) * len(SEEDS))],
                        np.r_[arrays["gg_positive_confidence"][p].ravel(), arrays["gg_null_confidence"][n].ravel()],
                    )
                ),
                expected_counts["probe_test"],
                null_count,
                repetitions,
                20260730,
            ),
        }
        decision = decide(primary_delta, mode=decision_mode)
        fractions = {
            target: {
                "minimum": float(min(float(row["region_token_fraction"]) for row in rows if row["target_type"] == target)),
                "median": float(np.median([float(row["region_token_fraction"]) for row in rows if row["target_type"] == target])),
                "maximum": float(max(float(row["region_token_fraction"]) for row in rows if row["target_type"] == target)),
            }
            for target in ("positive", "real_null")
        }
        report = {
            "version": report_version,
            "sample_counts": expected_counts,
            "representations": {"G": 2048, "R": 2048, "GG": 4096, "GR": 4096},
            "per_seed": per_seed,
            "primary_GR_minus_GG": primary_delta,
            "paired_bootstrap": bootstrap,
            "region_token_fraction": fractions,
            "decision": decision,
            "training": {"qwen_frozen": True, "linear_probe_only": True, "positive_only_fit": True, "null_used_for_fit": False},
            "task8_locked_set_read": False,
        }
        _write_jsonl_new(output / "predictions.jsonl", predictions)
        write_json_new(output / "metrics.json", report)
        write_json_new(output / "decision_report.json", decision)
        write_json_new(
            output / "run_summary.json",
            {
                "state": "completed",
                "decision": decision["decision"],
                "feature_manifest_sha256": sha256_file(features / "feature_rows.jsonl"),
                "bootstrap_repetitions": repetitions,
            },
        )
        signed = [f"seed_{seed}_metrics.json" for seed in SEEDS] + ["predictions.jsonl", "metrics.json", "decision_report.json", "run_summary.json"]
        with (output / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed:
                handle.write(f"{sha256_file(output / name)}  {name}\n")
        (output / "status.json").write_text('{"state":"completed"}\n', encoding="utf-8")
        return report
    except Exception as exc:
        write_json_new(output / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        (output / "status.json").write_text('{"state":"failed"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--null-count", type=int, default=32)
    parser.add_argument("--decision-mode", choices=("oracle", "replication"), default="oracle")
    parser.add_argument("--report-version", default="task14a-token-oracle-evaluation-1")
    args = parser.parse_args()
    report = run(
        protocol_root=args.protocol_root,
        feature_root=args.feature_root,
        output_root=args.output_root,
        repetitions=args.repetitions,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        null_count=args.null_count,
        decision_mode=args.decision_mode,
        report_version=args.report_version,
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
