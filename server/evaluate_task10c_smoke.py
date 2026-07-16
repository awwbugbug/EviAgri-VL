"""Engineering-only evaluator for the three-seed Task 10C C1 smoke."""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from run_task10c_smoke_inference import CONDITIONS
from task10c_contract import CLASS_IDS, canonical_pest_id, strict_parse_pest_json
from train_task10c_smoke import (
    SEEDS,
    _sha256,
    _verify_completion,
    _write_json,
    validate_smoke_summary,
    verify_protocol_gate,
)


EXPECTED_CONDITIONS = {condition: 16 for condition in CONDITIONS}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def condition_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [canonical_pest_id(int(row["class_id"])) for row in predictions]
    parsed = [row.get("parsed") or strict_parse_pest_json(str(row.get("raw_text", ""))) for row in predictions]
    predicted = [item.get("pest_id") if item.get("schema_valid") else None for item in parsed]
    class_f1 = []
    for class_id in CLASS_IDS:
        label = canonical_pest_id(class_id)
        tp = sum(truth == label and guess == label for truth, guess in zip(expected, predicted))
        fp = sum(truth != label and guess == label for truth, guess in zip(expected, predicted))
        fn = sum(truth == label and guess != label for truth, guess in zip(expected, predicted))
        denominator = 2 * tp + fp + fn
        class_f1.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return {
        "count": len(predictions),
        "syntax_validity": _rate(bool(item.get("syntax_valid")) for item in parsed),
        "schema_validity": _rate(bool(item.get("schema_valid")) for item in parsed),
        "accuracy": _rate(truth == guess for truth, guess in zip(expected, predicted)),
        "macro_f1": sum(class_f1) / len(CLASS_IDS),
        "unique_predicted_ids": len({value for value in predicted if value is not None}),
        "parse_failures": dict(sorted(Counter(
            str(item.get("error")) for item in parsed if item.get("error") is not None
        ).items())),
    }


def decide_c1_engineering(seed_reports: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if set(seed_reports) != set(SEEDS):
        reasons.append(f"seed set mismatch: {sorted(seed_reports)}")
    for seed, report in sorted(seed_reports.items()):
        if int(report.get("optimizer_steps", -1)) != 8:
            reasons.append(f"seed {seed} optimizer steps mismatch")
        if int(report.get("actual_exposures", -1)) != 64:
            reasons.append(f"seed {seed} exposure count mismatch")
        if int(report.get("prediction_count", -1)) != 64:
            reasons.append(f"seed {seed} prediction count mismatch")
        if report.get("condition_counts") != EXPECTED_CONDITIONS:
            reasons.append(f"seed {seed} condition count mismatch")
        if report.get("completion_verified") is not True or report.get("adapter_reload_verified") is not True:
            reasons.append(f"seed {seed} integrity gate failed")
    return {
        "version": "task10c-c1-engineering-decision-v1",
        "status": "FAIL_C1_ENGINEERING" if reasons else "PASS_C1_ENGINEERING",
        "reasons": reasons,
        "performance_is_nonbinding": True,
        "authorize_task10c_c2_execution": False,
        "requires_user_approval_for_c2": True,
    }


def evaluate_seed_smoke(
    *,
    protocol_root: Path,
    experiment_root: Path,
    seed: int,
) -> dict[str, Any]:
    training_root = experiment_root / "training" / f"seed_{seed}"
    inference_root = experiment_root / "inference" / f"seed_{seed}"
    _verify_completion(training_root)
    _verify_completion(inference_root)
    training = json.loads((training_root / "run_summary.json").read_text(encoding="utf-8"))
    validate_smoke_summary(training)
    inference = json.loads((inference_root / "run_summary.json").read_text(encoding="utf-8"))
    predictions = _read_jsonl(inference_root / "predictions.jsonl")
    if len(predictions) != 64 or len({str(row["id"]) for row in predictions}) != 64:
        raise ValueError(f"seed {seed} must have 64 unique predictions")
    condition_counts = dict(sorted(Counter(str(row["condition"]) for row in predictions).items()))
    if condition_counts != EXPECTED_CONDITIONS:
        raise ValueError(f"seed {seed} condition count mismatch: {condition_counts}")
    adapter_sha = training["adapter"]["sha256"]
    if inference.get("adapter_sha256") != adapter_sha:
        raise ValueError(f"seed {seed} adapter reload SHA mismatch")
    observations = {
        condition: condition_metrics([row for row in predictions if row["condition"] == condition])
        for condition in CONDITIONS
    }
    prompt_gap = abs(
        observations["image_train_prompt"]["macro_f1"]
        - observations["image_unseen_prompt"]["macro_f1"]
    )
    visual_gain = {
        "train_prompt": (
            observations["image_train_prompt"]["macro_f1"]
            - observations["no_image_train_prompt"]["macro_f1"]
        ),
        "unseen_prompt": (
            observations["image_unseen_prompt"]["macro_f1"]
            - observations["no_image_unseen_prompt"]["macro_f1"]
        ),
    }
    return {
        "seed": seed,
        "optimizer_steps": training["optimizer_steps"],
        "actual_exposures": training["actual_exposures"],
        "prediction_count": inference["prediction_count"],
        "condition_counts": condition_counts,
        "completion_verified": True,
        "adapter_reload_verified": bool(inference.get("adapter_reload_verified")),
        "adapter_sha256": adapter_sha,
        "elapsed_seconds": {
            "training": training["elapsed_seconds"],
            "inference": inference["elapsed_seconds"],
        },
        "peak_vram_allocated_bytes": training["peak_vram_allocated_bytes"],
        "peak_vram_reserved_bytes": training["peak_vram_reserved_bytes"],
        "observations": observations,
        "prompt_gap": prompt_gap,
        "visual_gain_macro_f1": visual_gain,
    }


def _write_completion(output: Path, names: list[str]) -> None:
    (output / "completion.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def run_c1_evaluation(
    *,
    protocol_root: str | Path,
    experiment_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    protocol_root, experiment_root, output = (
        Path(protocol_root), Path(experiment_root), Path(output_root)
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Task 10C C1 evaluation: {output}")
    output.mkdir(parents=True)
    _write_json(output / "status.json", {"state": "running", "stage": "evaluation"})
    try:
        protocol = verify_protocol_gate(protocol_root)
        seed_reports = {
            seed: evaluate_seed_smoke(
                protocol_root=protocol_root,
                experiment_root=experiment_root,
                seed=seed,
            )
            for seed in SEEDS
        }
        decision = decide_c1_engineering(seed_reports)
        metrics = {
            "version": "task10c-c1-smoke-metrics-v1",
            "seed_reports": {str(seed): report for seed, report in seed_reports.items()},
            "nonbinding_macro_f1_mean": {
                condition: statistics.mean(
                    seed_reports[seed]["observations"][condition]["macro_f1"] for seed in SEEDS
                )
                for condition in CONDITIONS
            },
            "decision": decision,
        }
        _write_json(output / "metrics.json", metrics)
        _write_json(output / "task10c_c1_decision_report.json", decision)
        summary = {
            "version": "task10c-c1-evaluation-summary-v1",
            "state": "completed" if decision["status"] == "PASS_C1_ENGINEERING" else "failed",
            "seed_count": len(seed_reports),
            "protocol_manifest_sha256": protocol["manifest_sha256"],
            "decision": decision["status"],
            "authorize_task10c_c2_execution": False,
        }
        _write_json(output / "run_summary.json", summary)
        _write_json(output / "status.json", {"state": summary["state"], "stage": "done"})
        if decision["status"] == "PASS_C1_ENGINEERING":
            _write_completion(output, [
                "metrics.json", "task10c_c1_decision_report.json",
                "run_summary.json", "status.json",
            ])
        else:
            _write_json(output / "failure.json", {
                "state": "failed", "stage": "evaluation", "reasons": decision["reasons"],
            })
        return {"decision": decision, "metrics": metrics, "summary": summary}
    except Exception as exc:
        _write_json(output / "failure.json", {
            "state": "failed", "stage": "evaluation", "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        _write_json(output / "status.json", {"state": "failed", "stage": "evaluation"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 10C C1 engineering smoke")
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_c1_evaluation(
        protocol_root=args.protocol_root,
        experiment_root=args.experiment_root,
        output_root=args.output_root,
    )
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    if result["decision"]["status"] != "PASS_C1_ENGINEERING":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
