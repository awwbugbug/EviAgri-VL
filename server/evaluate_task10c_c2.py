"""Evaluate the frozen Task 10C C2 learning curve and scientific gates."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from run_task10c_smoke_inference import CONDITIONS
from score_task10c_c2_candidates import validate_candidate_results
from task10c_c2_contract import C2_STEPS, verify_c2_protocol
from task10c_contract import (
    CLASS_IDS,
    EXPECTED_MANIFEST_SHA256,
    canonical_pest_id,
    strict_parse_pest_json,
)
from train_task10c_c2 import _sha256, _write_json, validate_c2_run_summary
from train_task10c_smoke import SEEDS, _verify_completion


SCIENTIFIC_GATES = (
    "d1_minus_d0_mean_macro_f1_ge_0_05",
    "pooled_paired_bootstrap_ci_low_gt_0",
    "at_least_two_seeds_above_d0",
    "d1_visual_gain_ge_0_10",
    "d1_mean_macro_f1_ge_0_5666020785",
    "every_seed_prompt_gap_lt_0_05",
    "every_seed_condition_syntax_schema_ge_0_99",
    "worst_seed_no_image_macro_f1_le_0_10",
    "source_and_component_overlap_eq_0",
)
EXPECTED_D2_MACRO_F1 = 0.8094315406815407
_CANONICAL_MENTION = re.compile(r"(?<![A-Za-z0-9])IP\d{3}(?![A-Za-z0-9])")
_FENCE = re.compile(r"^```(?:json)?\s*\n([\s\S]*?)\n```$", re.IGNORECASE)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def forensic_parse(raw: str) -> dict[str, Any]:
    raw = str(raw)
    match = _FENCE.fullmatch(raw)
    stripped = match.group(1) if match else None
    parsed = strict_parse_pest_json(stripped) if stripped is not None else {
        "syntax_valid": False, "schema_valid": False, "pest_id": None,
        "error": "no_single_outer_fence",
    }
    mentions = sorted(set(_CANONICAL_MENTION.findall(raw)))
    canonical_mentions = [value for value in mentions if value in {
        canonical_pest_id(class_id) for class_id in CLASS_IDS
    }]
    return {
        "has_single_outer_fence": match is not None,
        "fence_stripped_syntax_valid": bool(parsed["syntax_valid"]),
        "fence_stripped_schema_valid": bool(parsed["schema_valid"]),
        "canonical_id_mentioned": bool(canonical_mentions),
        "canonical_mentions": canonical_mentions,
    }


def forensic_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [forensic_parse(str(row.get("raw_text", ""))) for row in predictions]
    strict = [
        row.get("parsed") or strict_parse_pest_json(str(row.get("raw_text", "")))
        for row in predictions
    ]
    errors = Counter(
        str(item.get("error")) for item in strict if item.get("error") is not None
    )
    return {
        "count": len(predictions),
        "fence_rate": _rate(item["has_single_outer_fence"] for item in reports),
        "fence_stripped_schema_rate": _rate(
            item["fence_stripped_schema_valid"] for item in reports
        ),
        "canonical_mention_rate": _rate(item["canonical_id_mentioned"] for item in reports),
        "strict_error_counts": dict(sorted(errors.items())),
        "non_scoring": True,
    }


def _macro_f1(truth: list[str], predicted: list[str | None], labels: Iterable[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(left == label and right == label for left, right in zip(truth, predicted))
        fp = sum(left != label and right == label for left, right in zip(truth, predicted))
        fn = sum(left == label and right != label for left, right in zip(truth, predicted))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return statistics.mean(scores) if scores else 0.0


def condition_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [canonical_pest_id(int(row["class_id"])) for row in predictions]
    parsed = [
        row.get("parsed") or strict_parse_pest_json(str(row.get("raw_text", "")))
        for row in predictions
    ]
    predicted = [item.get("pest_id") if item.get("schema_valid") else None for item in parsed]
    labels = [canonical_pest_id(class_id) for class_id in CLASS_IDS]
    band_by_label: dict[str, str] = {}
    for row in predictions:
        band_by_label[canonical_pest_id(int(row["class_id"]))] = str(row["class_band"])
    band_macro = {}
    for band in ("head", "medium", "tail"):
        band_labels = [label for label in labels if band_by_label.get(label) == band]
        band_macro[band] = _macro_f1(truth, predicted, band_labels)
    confusion = Counter(
        f"{left}->{right if right is not None else 'INVALID'}"
        for left, right in zip(truth, predicted)
    )
    return {
        "count": len(predictions),
        "syntax_validity": _rate(bool(item.get("syntax_valid")) for item in parsed),
        "schema_validity": _rate(bool(item.get("schema_valid")) for item in parsed),
        "accuracy": _rate(left == right for left, right in zip(truth, predicted)),
        "macro_f1": _macro_f1(truth, predicted, labels),
        "band_macro_f1": band_macro,
        "confusion": dict(sorted(confusion.items())),
        "unique_predicted_ids": len({value for value in predicted if value is not None}),
        "parse_failures": dict(sorted(Counter(
            str(item.get("error")) for item in parsed if item.get("error") is not None
        ).items())),
    }


def seed_learning_signal(curve: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    early, final = curve["step_8"], curve["step_64"]
    deltas = {
        "image_schema_validity": (
            float(final["image_schema_validity"]) - float(early["image_schema_validity"])
        ),
        "image_macro_f1": float(final["image_macro_f1"]) - float(early["image_macro_f1"]),
        "visual_gain": float(final["visual_gain"]) - float(early["visual_gain"]),
    }
    checks = {
        "schema_delta_ge_0_25": deltas["image_schema_validity"] >= 0.25,
        "macro_f1_delta_ge_0_05": deltas["image_macro_f1"] >= 0.05,
        "visual_gain_delta_ge_0_05_and_final_gt_0": (
            deltas["visual_gain"] >= 0.05 and float(final["visual_gain"]) > 0
        ),
    }
    return {
        "passed": sum(checks.values()) >= 2,
        "passed_metric_count": sum(checks.values()),
        "deltas": deltas,
        "checks": checks,
    }


def aggregate_learning_signal(
    curves: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    if set(curves) != set(SEEDS):
        raise ValueError("learning curve seed set mismatch")
    per_seed = {seed: seed_learning_signal(curves[seed]) for seed in SEEDS}
    passed = sum(report["passed"] for report in per_seed.values())
    return {
        "passed": passed >= 2,
        "passed_seed_count": passed,
        "per_seed": {str(seed): report for seed, report in per_seed.items()},
    }


def _normalize_scoring_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        truth = str(row.get("truth") or canonical_pest_id(int(row["class_id"])))
        if "prediction" in row:
            prediction = row["prediction"]
        else:
            parsed = row.get("parsed") or strict_parse_pest_json(str(row.get("raw_text", "")))
            prediction = parsed.get("pest_id") if parsed.get("schema_valid") else None
        output.append({
            "source_image_sha256": str(row["source_image_sha256"]),
            "prompt_variant": str(row.get("prompt_variant") or row.get("condition")),
            "truth": truth,
            "prediction": prediction,
        })
    return sorted(output, key=lambda row: (row["source_image_sha256"], row["prompt_variant"]))


def _score_rows(rows: list[dict[str, Any]]) -> float:
    labels = [canonical_pest_id(class_id) for class_id in CLASS_IDS]
    return _macro_f1(
        [str(row["truth"]) for row in rows],
        [row.get("prediction") for row in rows],
        labels,
    )


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    location = (len(values) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return values[lower]
    weight = location - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def pooled_source_bootstrap(
    base_rows: list[dict[str, Any]],
    model_rows: Mapping[int, list[dict[str, Any]]],
    *,
    repetitions: int = 1000,
    seed: int = 20260717,
) -> dict[str, Any]:
    if repetitions <= 0 or set(model_rows) != set(SEEDS):
        raise ValueError("bootstrap requires positive repetitions and all frozen seeds")
    base = _normalize_scoring_rows(base_rows)
    models = {model_seed: _normalize_scoring_rows(rows) for model_seed, rows in model_rows.items()}
    base_keys = [(row["source_image_sha256"], row["prompt_variant"]) for row in base]
    if len(base_keys) != 160 or len(set(base_keys)) != 160:
        raise ValueError("bootstrap Base requires 160 unique paired image-prompt rows")
    if any(
        [(row["source_image_sha256"], row["prompt_variant"]) for row in rows] != base_keys
        for rows in models.values()
    ):
        raise ValueError("bootstrap model rows are not paired with Base")
    sources = sorted({row["source_image_sha256"] for row in base})
    if len(sources) != 80:
        raise ValueError("bootstrap requires exactly 80 source images")
    base_by_source = {source: [row for row in base if row["source_image_sha256"] == source] for source in sources}
    model_by_source = {
        model_seed: {
            source: [row for row in rows if row["source_image_sha256"] == source]
            for source in sources
        }
        for model_seed, rows in models.items()
    }

    def delta(sampled_sources: list[str]) -> float:
        sampled_base = [row for source in sampled_sources for row in base_by_source[source]]
        base_score = _score_rows(sampled_base)
        model_score = statistics.mean(
            _score_rows([
                row for source in sampled_sources for row in model_by_source[model_seed][source]
            ])
            for model_seed in SEEDS
        )
        return model_score - base_score

    estimate = delta(sources)
    generator = random.Random(seed)
    samples = [
        delta([generator.choice(sources) for _ in sources])
        for _ in range(repetitions)
    ]
    return {
        "unit": "source_image_sha256", "source_count": len(sources),
        "rows_per_source": 2, "seeds_clustered": True,
        "repetitions": repetitions, "seed": seed,
        "estimate": estimate,
        "low": _percentile(samples, 0.025), "high": _percentile(samples, 0.975),
        "delta_direction": "mean_D1_minus_D0",
    }


def paired_source_bootstrap(
    base_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    *,
    repetitions: int = 1000,
    seed: int = 20260717,
) -> dict[str, Any]:
    # Reuse the same source-cluster implementation by repeating the selected seed
    # into all three slots; this preserves the paired unit and percentile contract.
    return pooled_source_bootstrap(
        base_rows, {frozen_seed: model_rows for frozen_seed in SEEDS},
        repetitions=repetitions, seed=seed,
    )


def _gate_passed(value: Any) -> bool:
    return bool(value.get("passed")) if isinstance(value, Mapping) else bool(value)


def decide_c2(evidence: Mapping[str, Any]) -> dict[str, Any]:
    gates = evidence.get("gates", {})
    missing = [gate for gate in SCIENTIFIC_GATES if gate not in gates]
    engineering_complete = bool(evidence.get("engineering_complete")) and not missing
    if not engineering_complete:
        status = "ENGINEERING_FAILURE"
    elif all(_gate_passed(gates[gate]) for gate in SCIENTIFIC_GATES):
        status = "PASS"
    elif int(evidence.get("learning_signal", {}).get("passed_seed_count", 0)) >= 2:
        status = "LEARNING_SIGNAL_ONLY"
    else:
        status = "STRUCTURAL_FAILURE"
    return {
        "version": "task10c-c2-decision-v1",
        "status": status,
        "scientific_pass": status == "PASS",
        "failed_gates": [
            gate for gate in SCIENTIFIC_GATES
            if gate not in gates or not _gate_passed(gates[gate])
        ],
        "missing_gates": missing,
        "authorize_larger_training": False,
        "authorize_next_experiment": False,
        "requires_user_review": True,
    }


def _verify_predictions(
    root: Path,
    *,
    expected: int,
    condition_quota: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _verify_completion(root)
    summary = _read_json(root / "run_summary.json")
    rows = _read_jsonl(root / "predictions.jsonl")
    if summary.get("state") != "completed" or int(summary.get("prediction_count", -1)) != expected:
        raise ValueError(f"inference summary incomplete: {root}")
    if len(rows) != expected or len({str(row["id"]) for row in rows}) != expected:
        raise ValueError(f"inference row identity mismatch: {root}")
    counts = Counter(str(row["condition"]) for row in rows)
    if counts != Counter({condition: condition_quota for condition in CONDITIONS}):
        raise ValueError(f"inference condition count mismatch: {root}")
    return summary, rows


def _checkpoint_observation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    image = [row for row in rows if str(row["condition"]).startswith("image_")]
    no_image = [row for row in rows if str(row["condition"]).startswith("no_image_")]
    image_metrics = condition_metrics(image)
    no_image_metrics = condition_metrics(no_image)
    return {
        "image_schema_validity": image_metrics["schema_validity"],
        "image_macro_f1": image_metrics["macro_f1"],
        "no_image_macro_f1": no_image_metrics["macro_f1"],
        "visual_gain": image_metrics["macro_f1"] - no_image_metrics["macro_f1"],
        "conditions": {
            condition: condition_metrics([row for row in rows if row["condition"] == condition])
            for condition in CONDITIONS
        },
        "forensics": forensic_metrics(rows),
    }


def _verify_d2_reference(root: Path, protocol_root: Path) -> dict[str, Any]:
    _verify_completion(root)
    summary, metrics = _read_json(root / "run_summary.json"), _read_json(root / "metrics.json")
    if summary.get("state") != "completed" or summary.get("decision") != "PASS":
        raise ValueError("Task 10B D2 reference is not a completed PASS")
    decision = metrics.get("decision", {})
    observed = float(decision.get("mean_macro_f1", float("nan")))
    if not math.isclose(observed, EXPECTED_D2_MACRO_F1, rel_tol=0, abs_tol=1e-15):
        raise ValueError("Task 10B D2 signed mean Macro-F1 mismatch")
    sibling_manifest = root.parent / "protocol" / "manifest.jsonl"
    if not sibling_manifest.is_file() or _sha256(sibling_manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Task 10B D2 protocol manifest SHA mismatch")
    c2_config = _read_json(protocol_root / "config.snapshot.json")
    c2_manifest = Path(str(c2_config.get("manifest", "")))
    if (
        c2_config.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or not c2_manifest.is_file()
        or _sha256(c2_manifest) != EXPECTED_MANIFEST_SHA256
    ):
        raise ValueError("Task 10C source manifest SHA mismatch")
    return {
        "status": "PASS", "mean_macro_f1": observed,
        "completion_verified": True, "protocol_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "role": "read_only_reference_not_selected_or_retrained",
    }


def _candidate_report(root: Path) -> dict[str, Any]:
    _verify_completion(root)
    summary = _read_json(root / "run_summary.json")
    rows = _read_jsonl(root / "candidate_scores.jsonl")
    validate_candidate_results(rows)
    if summary.get("completed") is not True or summary.get("resource_preflight", {}).get("passed") is not True:
        raise ValueError(f"candidate scoring incomplete: {root}")
    return {
        "top1_accuracy": float(summary["top1_accuracy"]),
        "top3_accuracy": float(summary["top3_accuracy"]),
        "top5_accuracy": float(summary["top5_accuracy"]),
        "rows": len(rows), "completion_verified": True,
    }


def _write_completion(output: Path, names: list[str]) -> None:
    (output / "completion.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def run_c2_evaluation(
    *,
    protocol_root: str | Path,
    experiment_root: str | Path,
    task10b_evaluation_root: str | Path,
    output_root: str | Path,
    repetitions: int = 1000,
    bootstrap_seed: int = 20260717,
) -> dict[str, Any]:
    protocol_root, experiment_root, d2_root, output = map(
        Path, (protocol_root, experiment_root, task10b_evaluation_root, output_root)
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C2 evaluation: {output}")
    output.mkdir(parents=True)
    _write_json(output / "status.json", {"state": "running", "stage": "integrity"})
    try:
        protocol = verify_c2_protocol(protocol_root)
        d2 = _verify_d2_reference(d2_root, protocol_root)
        curves: dict[int, dict[str, Any]] = {}
        training_reports = {}
        for seed in SEEDS:
            training_root = experiment_root / "training" / f"seed_{seed}"
            _verify_completion(training_root)
            training = _read_json(training_root / "run_summary.json")
            validate_c2_run_summary(training)
            training_reports[str(seed)] = {
                "optimizer_steps": training["optimizer_steps"],
                "actual_exposures": training["actual_exposures"],
                "elapsed_seconds": training["elapsed_seconds"],
                "peak_vram_allocated_bytes": training["peak_vram_allocated_bytes"],
                "checkpoints": {
                    str(step): training["checkpoints"][str(step)]["adapter"]["sha256"]
                    for step in C2_STEPS
                },
            }
            curves[seed] = {}
            for step in C2_STEPS:
                inference_root = (
                    experiment_root / "inference" / "smoke" /
                    f"seed_{seed}" / f"step_{step:03d}"
                )
                summary, rows = _verify_predictions(
                    inference_root, expected=64, condition_quota=16,
                )
                expected_adapter = training["checkpoints"][str(step)]["adapter"]["sha256"]
                if summary.get("adapter_sha256") != expected_adapter:
                    raise ValueError(f"smoke adapter SHA mismatch seed={seed} step={step}")
                curves[seed][f"step_{step}"] = _checkpoint_observation(rows)

        learning_signal = aggregate_learning_signal(curves)
        base_summary, base_rows = _verify_predictions(
            experiment_root / "inference" / "dev" / "base",
            expected=320, condition_quota=80,
        )
        if base_summary.get("model_id") != "D0_base":
            raise ValueError("full-dev Base identity mismatch")
        full_rows: dict[int, list[dict[str, Any]]] = {}
        full_reports: dict[str, Any] = {}
        for seed in SEEDS:
            summary, rows = _verify_predictions(
                experiment_root / "inference" / "dev" / f"seed_{seed}",
                expected=320, condition_quota=80,
            )
            expected_adapter = training_reports[str(seed)]["checkpoints"]["64"]
            if summary.get("adapter_sha256") != expected_adapter:
                raise ValueError(f"full-dev adapter SHA mismatch seed={seed}")
            full_rows[seed] = rows
            observation = _checkpoint_observation(rows)
            train_macro = observation["conditions"]["image_train_prompt"]["macro_f1"]
            unseen_macro = observation["conditions"]["image_unseen_prompt"]["macro_f1"]
            full_reports[str(seed)] = {
                **observation,
                "prompt_gap": abs(train_macro - unseen_macro),
            }
        base_report = _checkpoint_observation(base_rows)

        base_image = [row for row in base_rows if str(row["condition"]).startswith("image_")]
        model_image = {
            seed: [row for row in rows if str(row["condition"]).startswith("image_")]
            for seed, rows in full_rows.items()
        }
        pooled = pooled_source_bootstrap(
            base_image, model_image, repetitions=repetitions, seed=bootstrap_seed,
        )
        per_seed_bootstrap = {
            str(seed): paired_source_bootstrap(
                base_image, model_image[seed], repetitions=repetitions,
                seed=bootstrap_seed + seed,
            )
            for seed in SEEDS
        }
        candidate_reports = {"base": _candidate_report(experiment_root / "candidates" / "base")}
        candidate_reports.update({
            f"seed_{seed}": _candidate_report(experiment_root / "candidates" / f"seed_{seed}")
            for seed in SEEDS
        })

        d0_macro = float(base_report["image_macro_f1"])
        d1_macros = [float(full_reports[str(seed)]["image_macro_f1"]) for seed in SEEDS]
        d1_mean = statistics.mean(d1_macros)
        d1_visual_gain = statistics.mean(
            float(full_reports[str(seed)]["visual_gain"]) for seed in SEEDS
        )
        every_condition_valid = all(
            metrics[validity] >= 0.99
            for seed in SEEDS
            for metrics in full_reports[str(seed)]["conditions"].values()
            for validity in ("syntax_validity", "schema_validity")
        )
        worst_no_image = max(
            float(full_reports[str(seed)]["no_image_macro_f1"]) for seed in SEEDS
        )
        gate_values = {
            SCIENTIFIC_GATES[0]: (d1_mean - d0_macro, 0.05, d1_mean - d0_macro >= 0.05),
            SCIENTIFIC_GATES[1]: (pooled["low"], 0.0, pooled["low"] > 0),
            SCIENTIFIC_GATES[2]: (sum(value > d0_macro for value in d1_macros), 2,
                                  sum(value > d0_macro for value in d1_macros) >= 2),
            SCIENTIFIC_GATES[3]: (d1_visual_gain, 0.10, d1_visual_gain >= 0.10),
            SCIENTIFIC_GATES[4]: (d1_mean, 0.5666020785, d1_mean >= 0.5666020785),
            SCIENTIFIC_GATES[5]: (max(full_reports[str(seed)]["prompt_gap"] for seed in SEEDS),
                                  0.05, all(full_reports[str(seed)]["prompt_gap"] < 0.05 for seed in SEEDS)),
            SCIENTIFIC_GATES[6]: (every_condition_valid, True, every_condition_valid),
            SCIENTIFIC_GATES[7]: (worst_no_image, 0.10, worst_no_image <= 0.10),
            SCIENTIFIC_GATES[8]: (
                {"source": protocol["source_overlap"], "component": protocol["component_overlap"]},
                {"source": 0, "component": 0},
                protocol["source_overlap"] == 0 and protocol["component_overlap"] == 0,
            ),
        }
        gates = {
            name: {"observed": observed, "threshold": threshold, "passed": passed}
            for name, (observed, threshold, passed) in gate_values.items()
        }
        evidence = {
            "engineering_complete": True,
            "learning_signal": learning_signal,
            "gates": gates,
        }
        decision = decide_c2(evidence)
        metrics = {
            "version": "task10c-c2-metrics-v1",
            "protocol_manifest_sha256": protocol["manifest_sha256"],
            "training": training_reports,
            "learning_curves": {str(seed): curve for seed, curve in curves.items()},
            "learning_signal": learning_signal,
            "D0_base": base_report,
            "D1_static_qlora": full_reports,
            "D2_reference": d2,
            "candidate_top_k_nonbinding": candidate_reports,
            "paired_bootstrap": per_seed_bootstrap,
            "pooled_paired_bootstrap": pooled,
            "gates": gates,
            "decision": decision,
        }
        _write_json(output / "metrics.json", metrics)
        _write_json(output / "task10c_c2_decision_report.json", {
            **decision,
            "gates": gates,
            "learning_signal": learning_signal,
            "D0_image_macro_f1": d0_macro,
            "D1_mean_image_macro_f1": d1_mean,
            "D2_reference_macro_f1": d2["mean_macro_f1"],
        })
        summary = {
            "version": "task10c-c2-evaluation-summary-v1",
            "state": "completed", "decision": decision["status"],
            "engineering_complete": True,
            "scientific_pass": decision["scientific_pass"],
            "bootstrap_repetitions": repetitions,
            "protocol_manifest_sha256": protocol["manifest_sha256"],
            "authorize_larger_training": False,
            "authorize_next_experiment": False,
        }
        _write_json(output / "run_summary.json", summary)
        _write_json(output / "status.json", {"state": "completed", "stage": "done", "decision": decision["status"]})
        _write_completion(output, [
            "metrics.json", "task10c_c2_decision_report.json",
            "run_summary.json", "status.json",
        ])
        return {"decision": decision, "metrics": metrics, "summary": summary}
    except Exception as exc:
        failure = {
            "state": "failed", "stage": "evaluation", "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output / "failure.json", failure)
        _write_json(output / "status.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--task10b-evaluation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    result = run_c2_evaluation(**vars(parser.parse_args()))
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
