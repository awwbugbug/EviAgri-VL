"""Evaluate the frozen Task 11A router on the audited PlantSeg damage real-null set."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import beta, spearmanr

from evaluate_task10b_probe import _classifier, _fit, _split_arrays
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new
from task11a_confidence_router import (
    SEEDS, render_router_json, serialized_payload, temperature_scale, validate_router_json,
)


EXPECTED_BASE_FEATURE_SHA256 = "5c730bab8d37d125f430d6b2fae1721359c04818f1dd86682e2f33a1ebbcaccc"
EXPECTED_FINAL_AUDIT_SHA256 = "56d8256949315a420cc57fd71bfeb72105020eb5900de500f2c520cb6ae9efb2"
FROZEN_TEMPERATURE = 0.18887372662036642
FROZEN_THRESHOLD = 0.63
EXPECTED_IMAGES = 290
BOOTSTRAP_SEED = 20260727


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_completion(root: Path) -> None:
    for line in (root / "completion.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        target = root / relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"completion SHA256 mismatch: {target}")


def bootstrap_fpr(acceptance_by_seed: dict[int, list[bool]], repetitions: int) -> dict[str, Any]:
    lengths = {len(values) for values in acceptance_by_seed.values()}
    if set(acceptance_by_seed) != set(SEEDS) or len(lengths) != 1 or not lengths or repetitions <= 0:
        raise ValueError("invalid aligned bootstrap inputs")
    count = lengths.pop()
    if count <= 0:
        raise ValueError("bootstrap requires images")
    def statistic(indices: list[int]) -> float:
        return mean(mean(float(acceptance_by_seed[seed][i]) for i in indices) for seed in SEEDS)
    estimate = statistic(list(range(count)))
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [statistic([rng.randrange(count) for _ in range(count)]) for _ in range(repetitions)]
    return {"estimate": estimate, "low": float(np.quantile(samples, .025)),
            "high": float(np.quantile(samples, .975)), "repetitions": repetitions,
            "unit": "unique_external_image"}


def exact_interval(successes: int, trials: int) -> dict[str, Any]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid exact interval inputs")
    low = 0.0 if successes == 0 else float(beta.ppf(.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(.975, successes + 1, trials - successes))
    return {"estimate": successes / trials, "low": low, "high": high,
            "successes": successes, "trials": trials, "confidence": .95,
            "method": "Clopper-Pearson exact", "unit": "unique_external_image_any_seed"}


def decide(fpr: float, exact_high: float, json_validity: float) -> dict[str, Any]:
    gates = {
        "real_null_fpr_lt_0_10": float(fpr) < .10,
        "exact_binomial_95ci_upper_lt_0_25": float(exact_high) < .25,
        "json_contract_eq_1": float(json_validity) == 1.0,
    }
    return {"gates": gates, "passed": all(gates.values()),
            "decision": "PASS" if all(gates.values()) else "FAIL"}


def _group_metrics(rows: list[dict[str, Any]], accepted: np.ndarray, confidence: np.ndarray, field: str) -> dict[str, Any]:
    output = {}
    for value in sorted({str(row[field]) for row in rows}):
        idx = np.asarray([i for i, row in enumerate(rows) if str(row[field]) == value], dtype=np.int64)
        output[value] = {"image_count": int(len(idx)), "accepted_count": int(accepted[idx].sum()),
                         "real_null_fpr": float(accepted[idx].mean()),
                         "maximum_confidence": float(confidence[idx].max()),
                         "mean_confidence": float(confidence[idx].mean())}
    return output


def run_evaluation(
    *, base_feature_root: Path, null_feature_root: Path, dataset_root: Path,
    task11a2_feature_rows: Path, output_root: Path, repetitions: int = 1000,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    (destination / "status.json").write_text('{"state":"running","stage":"verify"}\n', encoding="utf-8")
    try:
        base_root, null_root, data_root = map(Path, (base_feature_root, null_feature_root, dataset_root))
        for root in (base_root, null_root, data_root):
            _verify_completion(root)
        if sha256_file(base_root / "features.npy") != EXPECTED_BASE_FEATURE_SHA256:
            raise ValueError("unexpected Task 10B base feature SHA256")
        data_report = _read_json(data_root / "dataset_report.json")
        feature_summary = _read_json(null_root / "run_summary.json")
        manifest_sha = sha256_file(data_root / "manifest.jsonl")
        if (
            data_report.get("final_audit_sha256") != EXPECTED_FINAL_AUDIT_SHA256
            or data_report.get("manifest_sha256") != manifest_sha
            or int(data_report.get("image_count", -1)) != EXPECTED_IMAGES
            or feature_summary.get("manifest_sha256") != manifest_sha
            or int(feature_summary.get("feature_count", -1)) != EXPECTED_IMAGES
            or feature_summary.get("version") != "task11a3-plantseg-feature-summary-1"
        ):
            raise ValueError("Task 11A.3 dataset/feature contract mismatch")
        base_features = np.load(base_root / "features.npy", allow_pickle=False)
        null_features = np.load(null_root / "features.npy", allow_pickle=False)
        base_rows = _read_jsonl(base_root / "feature_rows.jsonl")
        null_rows = _read_jsonl(null_root / "feature_rows.jsonl")
        if base_features.shape != (320, 2048) or null_features.shape != (EXPECTED_IMAGES, 2048):
            raise ValueError("unexpected Task 11A.3 feature shape")
        if [int(row["feature_index"]) for row in null_rows] != list(range(EXPECTED_IMAGES)):
            raise ValueError("Task 11A.3 feature rows are not aligned")
        base_hashes = {str(row["source_image_sha256"]) for row in base_rows}
        prior_null_hashes = {str(row["image_sha256"]) for row in _read_jsonl(Path(task11a2_feature_rows))}
        null_hashes = {str(row["image_sha256"]) for row in null_rows}
        if len(null_hashes) != EXPECTED_IMAGES or null_hashes & (base_hashes | prior_null_hashes):
            raise ValueError("Task 11A.3 duplicate or prior-content overlap")
        x_train, y_train, _ = _split_arrays(base_features, base_rows, "train")
        acceptance_by_seed: dict[int, list[bool]] = {}
        seed_metrics, signed = {}, []
        all_valid = []
        for seed in SEEDS:
            classifier = _classifier(seed)
            _fit(classifier, x_train, y_train)
            scaled = temperature_scale(classifier.predict_proba(null_features), FROZEN_TEMPERATURE)
            confidence = scaled.max(axis=1)
            prediction = classifier.classes_[scaled.argmax(axis=1)].astype(np.int64)
            accepted = confidence >= FROZEN_THRESHOLD
            acceptance_by_seed[seed] = accepted.tolist()
            predictions, valid = [], []
            for source, pest_id, score, keep in zip(null_rows, prediction, confidence, accepted, strict=True):
                payload = render_router_json(int(pest_id) if keep else None)
                valid.append(validate_router_json(payload))
                predictions.append({"id": str(source["id"]), "condition": "external_damage_real_null",
                    "plant": str(source["plant"]), "disease": str(source["disease"]),
                    "image_sha256": str(source["image_sha256"]), "confidence": float(score),
                    "accepted": bool(keep), "forced_prediction": int(pest_id),
                    "payload": serialized_payload(payload)})
            all_valid.extend(valid)
            correlation = spearmanr(
                np.asarray([float(row["mask_ratio"]) for row in null_rows]), confidence
            )
            coefficient = float(correlation.statistic) if np.isfinite(correlation.statistic) else None
            pvalue = float(correlation.pvalue) if np.isfinite(correlation.pvalue) else None
            name = f"seed_{seed}_predictions.jsonl"
            with (destination / name).open("x", encoding="utf-8", newline="\n") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            seed_metrics[str(seed)] = {
                "real_null_fpr": float(accepted.mean()), "refusal_accuracy": float((~accepted).mean()),
                "accepted_count": int(accepted.sum()), "image_count": EXPECTED_IMAGES,
                "maximum_confidence": float(confidence.max()), "mean_confidence": float(confidence.mean()),
                "json_contract_validity": float(np.mean(valid)),
                "per_plant": _group_metrics(null_rows, accepted, confidence, "plant"),
                "per_disease": _group_metrics(null_rows, accepted, confidence, "disease"),
                "accepted_diagnosis_counts": {str(int(label)): int((prediction[accepted] == label).sum()) for label in sorted(set(prediction[accepted].tolist()))},
                "mask_ratio_confidence_spearman": {"coefficient": coefficient, "pvalue": pvalue, "exploratory_only": True},
            }
            signed.append(name)
        bootstrap = bootstrap_fpr(acceptance_by_seed, repetitions)
        any_accept = [any(acceptance_by_seed[seed][i] for seed in SEEDS) for i in range(EXPECTED_IMAGES)]
        exact = exact_interval(sum(any_accept), EXPECTED_IMAGES)
        decision = decide(bootstrap["estimate"], exact["high"], float(np.mean(all_valid)))
        report = {"version": "task11a3-plantseg-real-null-evaluation-1",
            "protocol": {"temperature": FROZEN_TEMPERATURE, "threshold": FROZEN_THRESHOLD,
                "seeds": list(SEEDS), "training_or_threshold_selection_on_plantseg": False,
                "input_contract": "pixels_only"},
            "seed_metrics": seed_metrics, "bootstrap": bootstrap, "exact_binomial": exact,
            "decision": decision,
            "limitations": ["PlantSeg contains disease/damage imagery rather than healthy leaves.",
                "The audit measures pest false positives and does not measure pest diagnosis accuracy.",
                "Mask-ratio correlations are exploratory and cannot tune the frozen router."]}
        write_json_new(destination / "metrics.json", report)
        write_json_new(destination / "task11a3_decision_report.json", decision)
        write_json_new(destination / "run_summary.json", {"state": "completed",
            "base_features_sha256": EXPECTED_BASE_FEATURE_SHA256,
            "null_features_sha256": sha256_file(null_root / "features.npy"),
            "null_manifest_sha256": manifest_sha,
            "final_audit_sha256": EXPECTED_FINAL_AUDIT_SHA256,
            "image_count": EXPECTED_IMAGES, "bootstrap_repetitions": repetitions,
            "decision": decision["decision"]})
        signed += ["metrics.json", "task11a3_decision_report.json", "run_summary.json"]
        with (destination / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed:
                handle.write(f"{sha256_file(destination / name)}  {name}\n")
        (destination / "status.json").write_text('{"state":"completed","stage":"done"}\n', encoding="utf-8")
        return report
    except Exception as exc:
        write_json_new(destination / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        (destination / "status.json").write_text('{"state":"failed","stage":"evaluation"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-feature-root", type=Path, required=True)
    parser.add_argument("--null-feature-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task11a2-feature-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    report = run_evaluation(base_feature_root=args.base_feature_root,
        null_feature_root=args.null_feature_root, dataset_root=args.dataset_root,
        task11a2_feature_rows=args.task11a2_feature_rows,
        output_root=args.output_root, repetitions=args.repetitions)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
