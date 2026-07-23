"""Evaluate the frozen Task 11A router on external PlantDoc real nulls."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import beta

from evaluate_task10b_probe import _classifier, _fit, _split_arrays
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from task11a_confidence_router import (
    SEEDS,
    render_router_json,
    serialized_payload,
    temperature_scale,
    validate_router_json,
)


EXPECTED_BASE_FEATURE_SHA256 = "5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc"
EXPECTED_NULL_FEATURE_SHA256 = "412815de2d6addd61b2863b9ec5227879888ae04250aabd5d736cce70159907a"
EXPECTED_NULL_MANIFEST_SHA256 = "79b74c673c7796e8e24617824c59f2f48750b0b27db6b491c7ce401e950a5db8"
FROZEN_TEMPERATURE = 0.18887372662036642
FROZEN_THRESHOLD = 0.63
BOOTSTRAP_SEED = 20260723


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


def bootstrap_fpr(
    acceptance_by_seed: dict[int, list[bool]], repetitions: int = 1000
) -> dict[str, Any]:
    lengths = {len(values) for values in acceptance_by_seed.values()}
    if set(acceptance_by_seed) != set(SEEDS) or len(lengths) != 1 or not lengths:
        raise ValueError("bootstrap requires aligned acceptance for all frozen seeds")
    count = lengths.pop()
    if count <= 0 or repetitions <= 0:
        raise ValueError("bootstrap count and repetitions must be positive")

    def statistic(indices: list[int]) -> float:
        return mean(
            mean(float(acceptance_by_seed[seed][index]) for index in indices)
            for seed in SEEDS
        )

    observed = statistic(list(range(count)))
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [
        statistic([rng.randrange(count) for _ in range(count)])
        for _ in range(repetitions)
    ]
    return {
        "estimate": observed,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "repetitions": repetitions,
        "unit": "external_image",
    }


def exact_binomial_interval(successes: int, trials: int, alpha: float = 0.05) -> dict[str, Any]:
    if trials <= 0 or successes < 0 or successes > trials or not 0 < alpha < 1:
        raise ValueError("invalid exact binomial interval arguments")
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return {
        "estimate": successes / trials,
        "low": low,
        "high": high,
        "successes": successes,
        "trials": trials,
        "confidence": 1 - alpha,
        "method": "Clopper-Pearson exact",
        "unit": "unique_external_image_any_seed",
    }


def decide_real_null(fpr: float, bootstrap_high: float, exact_high: float) -> dict[str, Any]:
    gates = {
        "real_null_fpr_lt_0_10": float(fpr) < 0.10,
        "bootstrap_95ci_upper_lt_0_25": float(bootstrap_high) < 0.25,
        "exact_binomial_95ci_upper_lt_0_25": float(exact_high) < 0.25,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def run_evaluation(
    *,
    base_feature_root: Path,
    null_feature_root: Path,
    manual_audit_path: Path,
    output_root: Path,
    repetitions: int = 1000,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    status = destination / "status.json"
    status.write_text('{"state":"running","stage":"verify"}\n', encoding="utf-8")
    try:
        base_root = Path(base_feature_root)
        null_root = Path(null_feature_root)
        _verify_completion(base_root)
        _verify_completion(null_root)
        if sha256_file(base_root / "features.npy") != EXPECTED_BASE_FEATURE_SHA256:
            raise ValueError("unexpected Task 10B base feature SHA256")
        if sha256_file(null_root / "features.npy") != EXPECTED_NULL_FEATURE_SHA256:
            raise ValueError("unexpected Task 11A.2 null feature SHA256")
        null_summary = _read_json(null_root / "run_summary.json")
        if null_summary.get("manifest_sha256") != EXPECTED_NULL_MANIFEST_SHA256:
            raise ValueError("unexpected Task 11A.2 source manifest SHA256")
        audit = _read_json(Path(manual_audit_path))
        if (
            audit.get("decision") != "PASS_REAL_NULL_VISUAL_GATE"
            or audit.get("manifest_sha256") != EXPECTED_NULL_MANIFEST_SHA256
            or int(audit.get("insect_visible_count", -1)) != 0
            or int(audit.get("insect_free_count", -1)) != 40
        ):
            raise ValueError("Task 11A.2 manual visual gate is not valid")

        base_features = np.load(base_root / "features.npy", allow_pickle=False)
        null_features = np.load(null_root / "features.npy", allow_pickle=False)
        base_rows = _read_jsonl(base_root / "feature_rows.jsonl")
        null_rows = _read_jsonl(null_root / "feature_rows.jsonl")
        if base_features.shape != (320, 2048) or null_features.shape != (40, 2048):
            raise ValueError("unexpected Task 11A.2 feature shape")
        if len(null_rows) != 40 or [int(row["feature_index"]) for row in null_rows] != list(range(40)):
            raise ValueError("Task 11A.2 feature rows are not aligned")
        task10b_hashes = {str(row["source_image_sha256"]) for row in base_rows}
        null_hashes = {str(row["image_sha256"]) for row in null_rows}
        if len(null_hashes) != 40 or task10b_hashes & null_hashes:
            raise ValueError("Task 11A.2 contains duplicate or overlapping image content")

        x_train, y_train, _ = _split_arrays(base_features, base_rows, "train")
        acceptance_by_seed: dict[int, list[bool]] = {}
        seed_metrics: dict[str, Any] = {}
        signed: list[str] = []
        for seed in SEEDS:
            classifier = _classifier(seed)
            _fit(classifier, x_train, y_train)
            scaled = temperature_scale(
                classifier.predict_proba(null_features), FROZEN_TEMPERATURE
            )
            confidence = scaled.max(axis=1)
            prediction = classifier.classes_[scaled.argmax(axis=1)].astype(np.int64)
            accepted = confidence >= FROZEN_THRESHOLD
            acceptance_by_seed[seed] = accepted.tolist()
            rows = []
            valid = []
            for source, pest_id, score, keep in zip(
                null_rows, prediction, confidence, accepted, strict=True
            ):
                payload = render_router_json(int(pest_id) if keep else None)
                valid.append(validate_router_json(payload))
                rows.append(
                    {
                        "id": str(source["id"]),
                        "healthy_class": str(source["healthy_class"]),
                        "image_sha256": str(source["image_sha256"]),
                        "condition": "external_real_null",
                        "confidence": float(score),
                        "accepted": bool(keep),
                        "forced_prediction": int(pest_id),
                        "payload": serialized_payload(payload),
                    }
                )
            per_healthy_class = {}
            for class_name in sorted({str(row["healthy_class"]) for row in null_rows}):
                indices = np.asarray(
                    [
                        index
                        for index, row in enumerate(null_rows)
                        if str(row["healthy_class"]) == class_name
                    ],
                    dtype=np.int64,
                )
                per_healthy_class[class_name] = {
                    "image_count": int(len(indices)),
                    "accepted_count": int(accepted[indices].sum()),
                    "real_null_fpr": float(accepted[indices].mean()),
                    "maximum_confidence": float(confidence[indices].max()),
                    "mean_confidence": float(confidence[indices].mean()),
                }
            accepted_diagnosis_counts = {
                str(int(label)): int((prediction[accepted] == label).sum())
                for label in sorted(set(prediction[accepted].tolist()))
            }
            predictions_name = f"seed_{seed}_predictions.jsonl"
            with (destination / predictions_name).open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                for row in rows:
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            seed_metrics[str(seed)] = {
                "real_null_fpr": float(accepted.mean()),
                "refusal_accuracy": float((~accepted).mean()),
                "accepted_count": int(accepted.sum()),
                "image_count": len(accepted),
                "maximum_confidence": float(confidence.max()),
                "mean_confidence": float(confidence.mean()),
                "json_contract_validity": float(np.mean(valid)),
                "per_healthy_class": per_healthy_class,
                "accepted_diagnosis_counts": accepted_diagnosis_counts,
            }
            signed.append(predictions_name)

        bootstrap = bootstrap_fpr(acceptance_by_seed, repetitions)
        any_seed_acceptance = [
            any(acceptance_by_seed[seed][index] for seed in SEEDS)
            for index in range(len(null_rows))
        ]
        exact_interval = exact_binomial_interval(
            sum(any_seed_acceptance), len(any_seed_acceptance)
        )
        decision = decide_real_null(
            bootstrap["estimate"], bootstrap["high"], exact_interval["high"]
        )
        report = {
            "version": "task11a2-plantdoc-real-null-evaluation-2",
            "protocol": {
                "temperature": FROZEN_TEMPERATURE,
                "threshold": FROZEN_THRESHOLD,
                "seeds": list(SEEDS),
                "training_or_threshold_selection_on_plantdoc": False,
            },
            "seed_metrics": seed_metrics,
            "bootstrap": bootstrap,
            "exact_binomial": exact_interval,
            "decision": decision,
            "limitations": audit.get("limitations", []),
        }
        write_json_new(destination / "metrics.json", report)
        write_json_new(destination / "task11a2_decision_report.json", decision)
        write_json_new(
            destination / "run_summary.json",
            {
                "state": "completed",
                "base_features_sha256": EXPECTED_BASE_FEATURE_SHA256,
                "null_features_sha256": EXPECTED_NULL_FEATURE_SHA256,
                "null_manifest_sha256": EXPECTED_NULL_MANIFEST_SHA256,
                "manual_audit_sha256": sha256_file(Path(manual_audit_path)),
                "image_count": 40,
                "bootstrap_repetitions": repetitions,
                "decision": decision["decision"],
            },
        )
        signed.extend(["metrics.json", "task11a2_decision_report.json", "run_summary.json"])
        with (destination / "completion.sha256").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            for name in signed:
                handle.write(f"{sha256_file(destination / name)}  {name}\n")
        status.write_text('{"state":"completed","stage":"done"}\n', encoding="utf-8")
        return report
    except Exception as exc:
        write_json_new(
            destination / "failure.json",
            {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()},
        )
        status.write_text('{"state":"failed","stage":"evaluation"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 11A.2 PlantDoc nulls")
    parser.add_argument("--base-feature-root", type=Path, required=True)
    parser.add_argument("--null-feature-root", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    arguments = parser.parse_args()
    report = run_evaluation(
        base_feature_root=arguments.base_feature_root,
        null_feature_root=arguments.null_feature_root,
        manual_audit_path=arguments.manual_audit,
        output_root=arguments.output_root,
        repetitions=arguments.repetitions,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
