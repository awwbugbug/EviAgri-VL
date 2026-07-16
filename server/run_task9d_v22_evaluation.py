"""Run paired evidence evaluation and frozen H1 decision for Task 9D v2.2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

from evaluate_task8 import exact_mcnemar, paired_bootstrap_delta
from evaluate_task9d_v22 import _evidence_metrics, decide_h1, evaluate_evidence_predictions


SEEDS = (17, 29, 43)


def _canonical_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in metrics["row_outcomes"]
        if row["primary_positive"] or not row["positive"]
    ]


def _tpr(rows: list[dict[str, Any]]) -> float:
    return _evidence_metrics(rows)["positive_tpr"]


def _fpr(rows: list[dict[str, Any]]) -> float:
    return _evidence_metrics(rows)["overall_null_fpr"]


def _ba(rows: list[dict[str, Any]]) -> float:
    return _evidence_metrics(rows)["balanced_accuracy"]


def paired_evidence_statistics(
    control: dict[str, Any],
    taxmask: dict[str, Any],
    *,
    repetitions: int = 1000,
    seed: int = 20260716,
) -> dict[str, Any]:
    control_rows = _canonical_rows(control)
    taxmask_rows = _canonical_rows(taxmask)
    left = {str(row["id"]): row for row in control_rows}
    right = {str(row["id"]): row for row in taxmask_rows}
    if set(left) != set(right):
        raise ValueError("v2.2 paired statistics require identical row IDs")
    identifiers = sorted(left)
    control_correct = [
        bool(left[i]["positive_correct"] if left[i]["primary_positive"] else not left[i]["null_fp"])
        for i in identifiers
    ]
    taxmask_correct = [
        bool(right[i]["positive_correct"] if right[i]["primary_positive"] else not right[i]["null_fp"])
        for i in identifiers
    ]
    return {
        "bootstrap": {"repetitions": repetitions, "seed": seed, "unit": "family_id"},
        "balanced_accuracy_delta_ci": paired_bootstrap_delta(
            control_rows, taxmask_rows, _ba, repetitions=repetitions, seed=seed,
        ),
        "positive_tpr_delta_ci": paired_bootstrap_delta(
            control_rows, taxmask_rows, _tpr, repetitions=repetitions, seed=seed + 1,
        ),
        "overall_null_fpr_delta_ci": paired_bootstrap_delta(
            control_rows, taxmask_rows, _fpr, repetitions=repetitions, seed=seed + 2,
        ),
        "mcnemar": exact_mcnemar(control_correct, taxmask_correct),
    }


def _summary(values: list[float], *, lower_is_better: bool) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "worst": max(values) if lower_is_better else min(values),
        "values": values,
    }


def aggregate_group_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seeds = tuple(str(seed) for seed in SEEDS)
    if set(metrics) != set(seeds):
        raise ValueError("v2.2 group aggregation requires seeds 17/29/43")
    specs: dict[str, tuple[Callable[[dict[str, Any]], float], bool]] = {
        "balanced_accuracy": (lambda m: m["evidence"]["balanced_accuracy"], False),
        "positive_tpr": (lambda m: m["evidence"]["positive_tpr"], False),
        "overall_null_fpr": (lambda m: m["evidence"]["overall_null_fpr"], True),
        "semantic_null_fpr": (lambda m: m["evidence"]["semantic_null_fpr"], True),
        "visual_null_fpr": (lambda m: m["evidence"]["visual_null_fpr"], True),
        "syntax_validity": (lambda m: m["json"]["syntax_validity"], False),
        "schema_validity": (lambda m: m["json"]["schema_validity"], False),
        "evidence_semantic_consistency": (lambda m: m["evidence_semantic_consistency"], False),
        "evidence_task_compliance": (lambda m: m["evidence_task_compliance"], False),
        "prompt_gap": (lambda m: m["prompt_gap"], True),
    }
    result: dict[str, Any] = {"seeds": len(seeds)}
    for name, (getter, lower) in specs.items():
        result[name] = _summary([float(getter(metrics[seed])) for seed in seeds], lower_is_better=lower)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _without_rows(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "row_outcomes"}


def run_evaluation(manifest_path: Path, inference_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite v2.2 evaluation: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_jsonl(manifest_path)
    if len(manifest) != 352:
        raise ValueError(f"v2.2 evaluation manifest must have 352 rows, found {len(manifest)}")
    groups: dict[str, dict[str, Any]] = {}
    input_hashes = {"manifest": _sha256(manifest_path)}
    for arm in ("Control", "TaxMask"):
        for seed in SEEDS:
            group = f"{arm}{seed}"
            path = inference_root / group / "predictions.jsonl"
            rows = _read_jsonl(path)
            if len(rows) != 352:
                raise ValueError(f"{group} must have 352 predictions, found {len(rows)}")
            input_hashes[group] = _sha256(path)
            groups[group] = evaluate_evidence_predictions(manifest, rows)
    paired = {
        str(seed): paired_evidence_statistics(
            groups[f"Control{seed}"], groups[f"TaxMask{seed}"],
            repetitions=1000, seed=20260716 + seed,
        ) for seed in SEEDS
    }
    pooled_control: list[dict[str, Any]] = []
    pooled_taxmask: list[dict[str, Any]] = []
    for seed in SEEDS:
        for source, target in (
            (groups[f"Control{seed}"]["row_outcomes"], pooled_control),
            (groups[f"TaxMask{seed}"]["row_outcomes"], pooled_taxmask),
        ):
            for row in source:
                copied = copy.deepcopy(row)
                copied["family_id"] = f"seed{seed}|{copied['family_id']}"
                copied["id"] = f"seed{seed}|{copied['id']}"
                target.append(copied)
    pooled_stats = paired_evidence_statistics(
        {"row_outcomes": pooled_control}, {"row_outcomes": pooled_taxmask},
        repetitions=1000, seed=20260716,
    )
    control_by_seed = {str(seed): groups[f"Control{seed}"] for seed in SEEDS}
    taxmask_by_seed = {str(seed): groups[f"TaxMask{seed}"] for seed in SEEDS}
    decision = decide_h1(
        control_by_seed, taxmask_by_seed,
        pooled_balanced_accuracy_delta_ci=pooled_stats["balanced_accuracy_delta_ci"],
    )
    report = {
        "version": "task9d-v22-decision-report-v1",
        "input_sha256": input_hashes,
        "groups": {name: _without_rows(value) for name, value in groups.items()},
        "aggregate": {
            "Control": aggregate_group_metrics(control_by_seed),
            "TaxMask": aggregate_group_metrics(taxmask_by_seed),
        },
        "paired_by_seed": paired,
        "pooled_paired": pooled_stats,
        "decision": decision,
        "task8_locked_set_read": False,
        "authorize_task9e": False,
    }
    _write_json(output_root / "v22_decision_report.json", report)
    _write_json(output_root / "group_metrics.json", report["groups"])
    names = ["v22_decision_report.json", "group_metrics.json"]
    (output_root / "completion.sha256").write_text(
        "".join(f"{_sha256(output_root / name)}  {name}\n" for name in names), encoding="utf-8"
    )
    _write_json(output_root / "status.json", {"state": "completed", "h1_passed": decision["passed"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_evaluation(args.manifest, args.inference_root, args.output_root)
    print(json.dumps({"decision": report["decision"], "aggregate": report["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
