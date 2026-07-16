"""Fail-closed orchestrator for the no-training Task 10A forensic gate."""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path
from typing import Any, Callable

from audit_task10_bbox_coordinates import (
    BLOCKED_COORDINATE_PROTOCOL,
    PASSED_COORDINATE_PROTOCOL,
    audit_coordinate_records,
    collect_coordinate_records,
)
from audit_task10_pdm import run_pdm_audit
from evaluate_task10_pairs import evaluate_family_pairs
from evaluate_task9d import _parse
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


CONTROL_GROUPS = {
    "Control17": 17,
    "Control29": 29,
    "Control43": 43,
    "TaxMask17": 17,
    "TaxMask29": 29,
    "TaxMask43": 43,
}
PRIMARY_CONTROL_GROUPS = ("Control17", "Control29", "Control43")


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


def _write_json_replace(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_completion(directory: Path) -> None:
    completion = directory / "completion.sha256"
    if not completion.is_file():
        raise ValueError(f"missing completion SHA256: {directory}")
    entries = []
    for line in completion.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid completion SHA256 line: {directory}")
        expected, relative = parts
        relative = relative.strip().lstrip("*")
        target = directory / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"completion SHA256 mismatch: {target}")
        entries.append(relative)
    if set(entries) != {"predictions.jsonl", "run_summary.json"}:
        raise ValueError(f"unexpected completion SHA256 contract: {directory}")


def _contains_task8_reference(manifest: list[dict[str, Any]]) -> bool:
    for row in manifest:
        messages = row.get("messages")
        if isinstance(messages, list):
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    continue
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in {"image", "video"}
                        and "task8" in str(item.get(item.get("type"), "")).lower()
                    ):
                        return True
        for key, value in row.items():
            if any(marker in str(key).lower() for marker in ("path", "file", "source")):
                if "task8" in str(value).lower():
                    return True
    return False


def _adapter_weight(adapter_dir: Path) -> Path:
    candidates = [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"expected one adapter weight file: {adapter_dir}")
    return existing[0]


def preflight_historical_inputs(
    historical_root: Path,
    *,
    expected_predictions: int = 352,
    expected_families: int = 32,
) -> dict[str, Any]:
    root = Path(historical_root)
    manifest_path = root / "protocol" / "evaluation_manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"missing v2.2 manifest: {manifest_path}")
    manifest = _read_jsonl(manifest_path)
    if len(manifest) != expected_predictions:
        raise ValueError(f"expected {expected_predictions} manifest rows, got {len(manifest)}")
    families = {str(row.get("family_id")) for row in manifest}
    if len(families) != expected_families or "" in families:
        raise ValueError(f"expected {expected_families} manifest families, got {len(families)}")
    if _contains_task8_reference(manifest):
        raise ValueError("Task 8 reference found in Task 10A manifest")
    manifest_hash = sha256_file(manifest_path)

    inference_paths: dict[str, Path] = {}
    adapter_paths: dict[int, Path] = {}
    contracts = []
    model_paths = set()
    for group, seed in CONTROL_GROUPS.items():
        inference = root / "inference" / group
        _verify_completion(inference)
        predictions_path = inference / "predictions.jsonl"
        summary = _read_json(inference / "run_summary.json")
        predictions = _read_jsonl(predictions_path)
        if (
            summary.get("state") != "completed"
            or summary.get("group") != group
            or int(summary.get("prediction_count", -1)) != expected_predictions
            or int(summary.get("expected_count", -1)) != expected_predictions
            or len(predictions) != expected_predictions
        ):
            raise ValueError(f"incomplete inference summary: {group}")
        if summary.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"manifest SHA256 mismatch: {group}")
        if summary.get("predictions_sha256") != sha256_file(predictions_path):
            raise ValueError(f"predictions SHA256 mismatch: {group}")
        contract = summary.get("contract")
        if not isinstance(contract, dict):
            raise ValueError(f"missing decoding contract: {group}")
        contracts.append(json.dumps(contract, sort_keys=True, separators=(",", ":")))

        adapter_dir = Path(str(summary.get("adapter_path", "")))
        if not adapter_dir.is_dir():
            raise ValueError(f"missing adapter directory: {group}")
        weight = _adapter_weight(adapter_dir)
        training_summary = _read_json(adapter_dir.parent / "run_summary.json")
        declared_adapter = training_summary.get("adapter")
        if (
            training_summary.get("completed") is not True
            or not isinstance(declared_adapter, dict)
            or Path(str(declared_adapter.get("path"))) != weight
            or declared_adapter.get("sha256") != sha256_file(weight)
        ):
            raise ValueError(f"adapter lineage/hash mismatch: {group}")
        adapter_config = _read_json(adapter_dir / "adapter_config.json")
        base_model = Path(str(adapter_config.get("base_model_name_or_path", "")))
        if not base_model.is_dir():
            raise ValueError(f"missing base model: {group}")
        model_paths.add(base_model)
        if group.startswith("Control"):
            adapter_paths[seed] = adapter_dir
        inference_paths[group] = predictions_path

    if len(set(contracts)) != 1:
        raise ValueError("v2.2 inference groups do not share one decoding contract")
    if len(model_paths) != 1:
        raise ValueError("v2.2 adapters do not share one base model")
    if set(adapter_paths) != {17, 29, 43}:
        raise ValueError("missing Control adapter seed")
    return {
        "historical_root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "inference_paths": inference_paths,
        "adapter_paths": adapter_paths,
        "model_path": next(iter(model_paths)),
        "decoding_contract": json.loads(contracts[0]),
        "expected_predictions": expected_predictions,
        "expected_families": expected_families,
        "task8_locked_set_read": False,
    }


def _prediction_boxes(path: Path) -> dict[str, Any]:
    boxes = {}
    for row in _read_jsonl(path):
        parsed = _parse(str(row.get("raw_text", "")))
        value = parsed["value"] if parsed["schema_valid"] else None
        boxes[str(row["id"])] = value.get("evidence_region") if value else None
    return boxes


def _default_bbox_runner(context: dict[str, Any], destination: Path) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    contract = context["decoding_contract"]
    processor = AutoProcessor.from_pretrained(
        context["model_path"],
        min_pixels=int(contract["min_pixels"]),
        max_pixels=int(contract["max_pixels"]),
        use_fast=False,
        local_files_only=True,
    )
    base_records = collect_coordinate_records(
        context["manifest"],
        processor=processor,
        vision_info_fn=process_vision_info,
        expected_families=context["expected_families"],
    )
    group_reports = {}
    for group in PRIMARY_CONTROL_GROUPS:
        boxes = _prediction_boxes(context["inference_paths"][group])
        records = copy.deepcopy(base_records)
        for record in records:
            record["predicted_box"] = boxes.get(str(record["id"]))
        group_reports[group] = audit_coordinate_records(records)
    passed = all(report["passed"] for report in group_reports.values())
    return {
        "version": "task10a-bbox-coordinate-multiseed-v1",
        "passed": passed,
        "status": PASSED_COORDINATE_PROTOCOL if passed else BLOCKED_COORDINATE_PROTOCOL,
        "groups": group_reports,
        "primary_frame": "original_image_pixels",
    }


def _default_pair_runner(context: dict[str, Any], destination: Path) -> dict[str, Any]:
    return {
        group: evaluate_family_pairs(context["manifest"], _read_jsonl(predictions_path))
        for group, predictions_path in context["inference_paths"].items()
    }


def _default_pdm_runner(context: dict[str, Any], destination: Path) -> dict[str, Any]:
    return run_pdm_audit(
        model_path=context["model_path"],
        adapter_paths=context["adapter_paths"],
        manifest_path=context["manifest_path"],
        output_dir=destination / "pdm",
        repetitions=1000,
    )


def _compact_pairs(pairs: dict[str, Any]) -> dict[str, Any]:
    return {
        group: {key: value for key, value in report.items() if key != "families"}
        for group, report in pairs.items()
    }


def decide_task10a(
    *,
    bbox: dict[str, Any],
    pdm: dict[str, Any],
    pairs: dict[str, Any],
) -> dict[str, Any]:
    pdm_passed = bool(pdm.get("quality_passed") and pdm.get("visual_dependency_passed"))
    return {
        "version": "task10a-decision-report-v1",
        "bbox_coordinate_status": bbox.get("status", BLOCKED_COORDINATE_PROTOCOL),
        "existing_verifier_visual_dependency_status": (
            "PASSED_VISUAL_DEPENDENCY" if pdm_passed else "FAILED_VISUAL_DEPENDENCY"
        ),
        "pair_forensic_findings": _compact_pairs(pairs),
        "authorize_existing_verifier_for_task10d": pdm_passed,
        "authorize_task10b_planning": True,
        "authorize_task10b_execution": False,
        "authorize_training": False,
        "task8_locked_set_read": False,
    }


def run_task10a(
    *,
    historical_root: Path,
    output_root: Path,
    expected_predictions: int = 352,
    expected_families: int = 32,
    bbox_runner: Callable[[dict[str, Any], Path], dict[str, Any]] = _default_bbox_runner,
    pair_runner: Callable[[dict[str, Any], Path], dict[str, Any]] = _default_pair_runner,
    pdm_runner: Callable[[dict[str, Any], Path], dict[str, Any]] = _default_pdm_runner,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    _write_json_replace(destination / "status.json", {"state": "running", "stage": "preflight"})
    try:
        context = preflight_historical_inputs(
            Path(historical_root),
            expected_predictions=expected_predictions,
            expected_families=expected_families,
        )
        _write_json_replace(destination / "status.json", {"state": "running", "stage": "bbox"})
        bbox = bbox_runner(context, destination)
        write_json_new(destination / "bbox_coordinate_report.json", bbox)

        _write_json_replace(destination / "status.json", {"state": "running", "stage": "pairs"})
        pairs = pair_runner(context, destination)
        write_json_new(destination / "pair_metrics.json", pairs)

        _write_json_replace(destination / "status.json", {"state": "running", "stage": "pdm_h"})
        pdm = pdm_runner(context, destination)
        write_json_new(destination / "pdm_token_report.json", pdm)

        decision = decide_task10a(bbox=bbox, pdm=pdm, pairs=pairs)
        decision["input_sha256"] = {
            "manifest": context["manifest_sha256"],
            **{
                group: sha256_file(path)
                for group, path in context["inference_paths"].items()
            },
        }
        write_json_new(destination / "task10a_decision_report.json", decision)
        summary = {
            "version": "task10a-run-summary-v1",
            "state": "completed",
            "historical_root": str(context["historical_root"]),
            "model_path": str(context["model_path"]),
            "manifest_sha256": context["manifest_sha256"],
            "family_count": expected_families,
            "prediction_count_per_group": expected_predictions,
            "task8_locked_set_read": False,
            "authorize_training": False,
        }
        write_json_new(destination / "run_summary.json", summary)
        signed = [
            "bbox_coordinate_report.json",
            "pair_metrics.json",
            "pdm_token_report.json",
            "task10a_decision_report.json",
            "run_summary.json",
        ]
        (destination / "completion.sha256").write_text(
            "".join(f"{sha256_file(destination / name)}  {name}\n" for name in signed),
            encoding="utf-8",
        )
        _write_json_replace(destination / "status.json", {"state": "completed", "stage": "done"})
        return decision
    except Exception as exc:
        write_json_new(destination / "failure.json", {
            "state": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        _write_json_replace(destination / "status.json", {"state": "failed", "stage": "blocked"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-predictions", type=int, default=352)
    parser.add_argument("--expected-families", type=int, default=32)
    args = parser.parse_args()
    report = run_task10a(
        historical_root=args.historical_root,
        output_root=args.output_root,
        expected_predictions=args.expected_predictions,
        expected_families=args.expected_families,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
