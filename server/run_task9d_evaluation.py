"""Formal, same-protocol Task 9D evaluation orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from decide_task9d import decide_task9d
from evaluate_task9d import evaluate_predictions, paired_statistics


GROUPS = ("Base", "A17", "A29", "A43", "B17", "B29", "B43", "C17", "C29", "C43")
VARIANT_GROUPS = {
    "A": ("A17", "A29", "A43"),
    "B": ("B17", "B29", "B43"),
    "C": ("C17", "C29", "C43"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_completion(group_root: Path) -> None:
    completion = group_root / "completion.sha256"
    if not completion.is_file():
        raise ValueError(f"missing completion.sha256: {group_root}")
    for line in completion.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        target = group_root / relative
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(f"completion SHA256 mismatch: {target}")


def _preflight(manifest: Path, inference_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_sha256 = _sha256(manifest)
    manifest_rows = _read_jsonl(manifest)
    contracts: set[str] = set()
    predictions: dict[str, list[dict[str, Any]]] = {}
    for group in GROUPS:
        group_root = inference_root / group
        if (group_root / "failure.json").exists():
            raise ValueError(f"failure.json exists for {group}")
        _verify_completion(group_root)
        summary = json.loads((group_root / "run_summary.json").read_text(encoding="utf-8"))
        prediction_path = group_root / "predictions.jsonl"
        rows = _read_jsonl(prediction_path)
        if summary.get("state") != "completed" or summary.get("group") != group:
            raise ValueError(f"invalid run summary state/group for {group}")
        if int(summary.get("prediction_count", -1)) != len(rows) or len(rows) != len(manifest_rows):
            raise ValueError(f"prediction count mismatch for {group}")
        if summary.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"manifest SHA256 mismatch for {group}")
        if summary.get("predictions_sha256") != _sha256(prediction_path):
            raise ValueError(f"predictions SHA256 mismatch for {group}")
        contracts.add(json.dumps(summary.get("contract"), sort_keys=True, separators=(",", ":")))
        predictions[group] = rows
    if len(contracts) != 1:
        raise ValueError("inference groups do not share one decoding contract")
    return ({
        "groups_verified": len(GROUPS),
        "total_predictions": sum(len(rows) for rows in predictions.values()),
        "predictions_per_group": len(manifest_rows),
        "manifest_sha256": manifest_sha256,
        "same_manifest": True,
        "same_contract": True,
        "contract": json.loads(next(iter(contracts))),
        "completion_sha256_verified": True,
    }, predictions)


def _shortcut_gate_status(path: Path) -> dict[str, bool]:
    value = json.loads(path.read_text(encoding="utf-8"))
    gates = value.get("details", {}).get("shortcut_gates", {})
    result = {
        variant: bool(
            gates.get(variant, {}).get("decision") == "PASS"
            and gates.get(variant, {}).get("training_allowed") is True
        )
        for variant in VARIANT_GROUPS
    }
    if not value.get("passed", False) or not all(result.values()):
        raise ValueError("pretraining or shortcut gate is not PASS for all variants")
    return result


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "row_outcomes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Task 9D formal evaluation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--class-bands", type=Path, required=True)
    parser.add_argument("--pretraining-gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_root = args.output_root / "metrics"
    metrics_root.mkdir()

    preflight, predictions = _preflight(args.manifest, args.inference_root)
    shortcut_gates = _shortcut_gate_status(args.pretraining_gate)
    manifest = _read_jsonl(args.manifest)
    class_bands = {
        int(key): value
        for key, value in json.loads(args.class_bands.read_text(encoding="utf-8")).items()
    }
    metrics = {
        group: evaluate_predictions(manifest, predictions[group], class_bands)
        for group in GROUPS
    }
    base = metrics["Base"]
    paired: dict[str, dict[str, Any]] = {}
    for group in GROUPS[1:]:
        stats = paired_statistics(base, metrics[group], repetitions=args.repetitions, seed=args.seed)
        paired[group] = stats
        metrics[group].update({
            "shortcut_gate_passed": shortcut_gates[group[0]],
            "macro_f1_delta_ci": stats["macro_f1_delta_ci"],
            "null_fpr_delta_ci": stats["null_fpr_delta_ci"],
            "localization_delta_ci": stats["localization_delta_ci"],
        })

    variants = {
        variant: [metrics[group] for group in groups]
        for variant, groups in VARIANT_GROUPS.items()
    }
    decision = decide_task9d(base, variants)
    compact_groups: dict[str, Any] = {}
    metric_hashes: dict[str, str] = {}
    for group in GROUPS:
        payload = _compact(metrics[group])
        if group != "Base":
            payload["paired_vs_base"] = paired[group]
        metric_path = metrics_root / f"{group}.json"
        _write_json(metric_path, {**payload, "row_outcomes": metrics[group]["row_outcomes"]})
        metric_hashes[group] = _sha256(metric_path)
        compact_groups[group] = payload

    report = {
        "version": "task9d-formal-evaluation-v1",
        "bootstrap": {"repetitions": args.repetitions, "seed": args.seed, "unit": "family_id"},
        "preflight": preflight,
        "shortcut_gates": shortcut_gates,
        "groups": compact_groups,
        "decision": decision,
        "task8_locked_set_read": False,
        "task9e_started": False,
    }
    report_path = args.output_root / "task9d_decision_report.json"
    _write_json(report_path, report)
    script_root = Path(__file__).parent
    run_summary = {
        "version": "task9d-formal-evaluation-summary-v1",
        "state": "completed",
        "preflight": preflight,
        "bootstrap": report["bootstrap"],
        "decision_report_sha256": _sha256(report_path),
        "metrics_sha256": metric_hashes,
        "code_sha256": {
            "runner": _sha256(Path(__file__)),
            "evaluator": _sha256(script_root / "evaluate_task9d.py"),
            "decider": _sha256(script_root / "decide_task9d.py"),
        },
    }
    summary_path = args.output_root / "run_summary.json"
    _write_json(summary_path, run_summary)
    artifacts = [report_path, summary_path, *(metrics_root / f"{group}.json" for group in GROUPS)]
    completion = args.output_root / "completion.sha256"
    completion.write_text("".join(
        f"{_sha256(path)}  {path.relative_to(args.output_root).as_posix()}\n"
        for path in artifacts
    ), encoding="utf-8")
    print(json.dumps({
        "state": "completed",
        "selected_protocol": decision["selected_protocol"],
        "protocol_repair_passed": decision["protocol_repair_passed"],
        "scientific_passed": decision["scientific_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
