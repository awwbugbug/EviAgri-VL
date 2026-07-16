import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from run_task10a_forensics import (
    CONTROL_GROUPS,
    decide_task10a,
    preflight_historical_inputs,
    run_task10a,
)
from task10_audit_common import sha256_file


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _completion(directory, names):
    (directory / "completion.sha256").write_text(
        "".join(f"{sha256_file(directory / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _manifest(image_path):
    conditions = (
        "original",
        "semantic_null",
        "source_visual_null",
        "blank",
        "blur",
        "shuffle",
    )
    rows = []
    for condition in conditions:
        rows.append({
            "id": f"f1-{condition}",
            "family_id": "f1",
            "role": "positive" if condition == "original" else "visual_counterfactual",
            "condition": condition,
            "prompt_view": "canonical",
            "query_class_id": 12,
            "gt_bbox": [1, 2, 30, 40] if condition == "original" else None,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "queried pest 'aphid'"},
                ]},
            ],
        })
    return rows


def _historical_fixture(tmp_path):
    root = tmp_path / "historical"
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"fake-image")
    manifest = root / "protocol" / "evaluation_manifest.jsonl"
    rows = _manifest(image)
    _write_jsonl(manifest, rows)
    manifest_hash = sha256_file(manifest)
    contract = {
        "do_sample": False,
        "max_new_tokens": 512,
        "min_pixels": 200704,
        "max_pixels": 401408,
        "parser_version": "task9d-json-parser-v1",
    }
    for group, seed in CONTROL_GROUPS.items():
        arm = "Control" if group.startswith("Control") else "TaxMask"
        adapter_dir = root / "runs" / arm / f"seed_{seed}" / "adapter"
        adapter_dir.mkdir(parents=True)
        weight = adapter_dir / "adapter_model.safetensors"
        weight.write_bytes(f"{group}-weights".encode())
        _write_json(adapter_dir / "adapter_config.json", {
            "base_model_name_or_path": str(model),
            "r": 16,
            "target_modules": ["q_proj", "v_proj"],
        })
        _write_json(adapter_dir.parent / "run_summary.json", {
            "completed": True,
            "arm": arm,
            "seed": seed,
            "adapter": {"path": str(weight), "sha256": sha256_file(weight)},
        })
        inference = root / "inference" / group
        predictions = inference / "predictions.jsonl"
        prediction_rows = [{"id": row["id"], "raw_text": "{}"} for row in rows]
        _write_jsonl(predictions, prediction_rows)
        _write_json(inference / "run_summary.json", {
            "state": "completed",
            "group": group,
            "adapter_path": str(adapter_dir),
            "prediction_count": len(rows),
            "expected_count": len(rows),
            "contract": contract,
            "manifest_sha256": manifest_hash,
            "predictions_sha256": sha256_file(predictions),
        })
        _completion(inference, ["predictions.jsonl", "run_summary.json"])
    return root, model, rows


def _passing_bbox():
    return {"passed": True, "status": "PASSED_COORDINATE_PROTOCOL"}


def _passing_pdm():
    return {"quality_passed": True, "visual_dependency_passed": True}


def _failing_pdm():
    return {"quality_passed": True, "visual_dependency_passed": False}


def _pair_report():
    return {
        group: {
            "original_positive_tpr": 0.5,
            "strict_family_success": 0.25,
            "invalid_prediction_count": 0,
        }
        for group in ("Control17", "Control29", "Control43")
    }


def _failing_pair_report():
    report = _pair_report()
    report["Control29"] = {
        "original_positive_tpr": 0.0,
        "strict_family_success": 0.0,
        "invalid_prediction_count": 7,
    }
    return report


def test_preflight_verifies_six_groups_hashes_contract_and_adapter_lineage(tmp_path):
    root, model, rows = _historical_fixture(tmp_path)

    context = preflight_historical_inputs(
        root,
        expected_predictions=len(rows),
        expected_families=1,
    )

    assert context["model_path"] == model
    assert set(context["inference_paths"]) == set(CONTROL_GROUPS)
    assert set(context["adapter_paths"]) == {17, 29, 43}
    assert context["manifest_sha256"] == sha256_file(context["manifest_path"])
    assert context["task8_locked_set_read"] is False


def test_preflight_blocks_tampered_completion_hash(tmp_path):
    root, _, rows = _historical_fixture(tmp_path)
    predictions = root / "inference" / "Control17" / "predictions.jsonl"
    predictions.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completion SHA256"):
        preflight_historical_inputs(root, expected_predictions=len(rows), expected_families=1)


def test_preflight_blocks_any_task8_reference_without_reading_it(tmp_path):
    root, _, rows = _historical_fixture(tmp_path)
    manifest = root / "protocol" / "evaluation_manifest.jsonl"
    rows[0]["messages"][1]["content"][0]["image"] = "/forbidden/task8_locked/image.jpg"
    _write_jsonl(manifest, rows)

    with pytest.raises(ValueError, match="Task 8 reference"):
        preflight_historical_inputs(root, expected_predictions=len(rows), expected_families=1)


def test_decision_blocks_verifier_when_pdm_fails_but_preserves_diagnosis_path():
    report = decide_task10a(
        bbox=_passing_bbox(),
        pdm=_failing_pdm(),
        pairs=_pair_report(),
    )

    assert report["authorize_existing_verifier_for_task10d"] is False
    assert report["authorize_task10b_planning"] is True
    assert report["authorize_training"] is False


def test_decision_never_authorizes_training_or_task10b_execution():
    report = decide_task10a(
        bbox=_passing_bbox(),
        pdm=_passing_pdm(),
        pairs=_pair_report(),
    )

    assert report["authorize_existing_verifier_for_task10d"] is True
    assert report["authorize_training"] is False
    assert report["authorize_task10b_execution"] is False
    assert report["task8_locked_set_read"] is False


def test_visual_dependency_alone_cannot_authorize_contract_failing_verifier():
    report = decide_task10a(
        bbox=_passing_bbox(),
        pdm=_passing_pdm(),
        pairs=_failing_pair_report(),
    )

    assert report["existing_verifier_visual_dependency_status"] == "PASSED_VISUAL_DEPENDENCY"
    assert report["existing_verifier_pair_contract_status"] == "FAILED_PAIR_CONTRACT"
    assert report["authorize_existing_verifier_for_task10d"] is False
    assert "pair_contract_failed" in report["verifier_reuse_blockers"]


def test_orchestrator_writes_signed_outputs_and_refuses_existing_root(tmp_path):
    historical, _, rows = _historical_fixture(tmp_path)
    output = tmp_path / "task10a"

    report = run_task10a(
        historical_root=historical,
        output_root=output,
        expected_predictions=len(rows),
        expected_families=1,
        bbox_runner=lambda context, destination: _passing_bbox(),
        pair_runner=lambda context, destination: _pair_report(),
        pdm_runner=lambda context, destination: _passing_pdm(),
    )

    assert report["authorize_training"] is False
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["state"] == "completed"
    assert (output / "completion.sha256").is_file()
    with pytest.raises(FileExistsError):
        run_task10a(
            historical_root=historical,
            output_root=output,
            expected_predictions=len(rows),
            expected_families=1,
            bbox_runner=lambda context, destination: _passing_bbox(),
            pair_runner=lambda context, destination: _pair_report(),
            pdm_runner=lambda context, destination: _passing_pdm(),
        )


def test_orchestrator_records_failure_and_does_not_continue_to_pdm(tmp_path):
    historical, _, rows = _historical_fixture(tmp_path)
    output = tmp_path / "failed-task10a"
    called = {"pdm": False}

    def broken_pair(context, destination):
        raise RuntimeError("pair integrity failed")

    def forbidden_pdm(context, destination):
        called["pdm"] = True
        return _passing_pdm()

    with pytest.raises(RuntimeError, match="pair integrity failed"):
        run_task10a(
            historical_root=historical,
            output_root=output,
            expected_predictions=len(rows),
            expected_families=1,
            bbox_runner=lambda context, destination: _passing_bbox(),
            pair_runner=broken_pair,
            pdm_runner=forbidden_pdm,
        )

    assert called["pdm"] is False
    assert json.loads((output / "failure.json").read_text(encoding="utf-8"))["state"] == "failed"
