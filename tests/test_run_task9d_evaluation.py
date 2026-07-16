import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
GROUPS = ("Base", "A17", "A29", "A43", "B17", "B29", "B43", "C17", "C29", "C43")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _output(present: bool, pest_id: int | None = None) -> str:
    return json.dumps({
        "evidence_present": present,
        "evidence_region": [0, 0, 10, 10] if present else None,
        "visible_attributes": ["visible"] if present else [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": pest_id if present else None,
            "pest_name": f"p{pest_id}" if present else None,
            "species": None,
            "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }, separators=(",", ":"))


def test_formal_runner_verifies_protocol_and_writes_decision_artifacts(tmp_path):
    manifest_rows = [
        {"id": "p", "family_id": "f1", "role": "positive", "condition": "original",
         "prompt_view": "canonical", "query_class_id": 1, "gt_bbox": [0, 0, 10, 10]},
        {"id": "n0", "family_id": "f1", "role": "positive", "condition": "original",
         "prompt_view": "native_0", "query_class_id": 1, "gt_bbox": [0, 0, 10, 10]},
        {"id": "n1", "family_id": "f1", "role": "positive", "condition": "original",
         "prompt_view": "native_1", "query_class_id": 1, "gt_bbox": [0, 0, 10, 10]},
        {"id": "n2", "family_id": "f1", "role": "positive", "condition": "original",
         "prompt_view": "native_2", "query_class_id": 1, "gt_bbox": [0, 0, 10, 10]},
        {"id": "sem", "family_id": "f2", "role": "semantic_negative", "condition": "semantic_null",
         "prompt_view": "canonical", "query_class_id": 1, "gt_bbox": None},
        {"id": "blank", "family_id": "f1", "role": "visual_counterfactual", "condition": "blank",
         "prompt_view": "canonical", "query_class_id": 1, "gt_bbox": None},
        {"id": "blur", "family_id": "f1", "role": "visual_counterfactual", "condition": "blur",
         "prompt_view": "canonical", "query_class_id": 1, "gt_bbox": None},
        {"id": "shuffle", "family_id": "f1", "role": "visual_counterfactual", "condition": "shuffle",
         "prompt_view": "canonical", "query_class_id": 1, "gt_bbox": None},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
    bands = tmp_path / "class_bands.json"
    bands.write_text(json.dumps({"1": "head"}), encoding="utf-8")
    gate = tmp_path / "pretraining_gate.json"
    gate.write_text(json.dumps({
        "passed": True,
        "details": {"shortcut_gates": {
            variant: {"decision": "PASS", "training_allowed": True}
            for variant in ("A", "B", "C")
        }},
    }), encoding="utf-8")
    inference_root = tmp_path / "inference"
    contract = {"do_sample": False, "max_new_tokens": 512, "max_pixels": 401408,
                "min_pixels": 200704, "parser_version": "task9d-json-parser-v1"}
    predictions = [
        {"id": row["id"], "raw_text": _output(row["role"] == "positive", 1)}
        for row in manifest_rows
    ]
    for group in GROUPS:
        group_root = inference_root / group
        group_root.mkdir(parents=True)
        prediction_path = group_root / "predictions.jsonl"
        prediction_path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
        summary_path = group_root / "run_summary.json"
        summary_path.write_text(json.dumps({
            "state": "completed", "group": group, "prediction_count": len(predictions),
            "expected_count": len(predictions), "manifest_sha256": _sha256(manifest),
            "predictions_sha256": _sha256(prediction_path), "contract": contract,
        }), encoding="utf-8")
        (group_root / "completion.sha256").write_text(
            f"{_sha256(prediction_path)}  predictions.jsonl\n{_sha256(summary_path)}  run_summary.json\n",
            encoding="utf-8",
        )
    output = tmp_path / "formal_evaluation"
    command = [
        sys.executable, str(ROOT / "server" / "run_task9d_evaluation.py"),
        "--manifest", str(manifest), "--inference-root", str(inference_root),
        "--class-bands", str(bands), "--pretraining-gate", str(gate),
        "--output-root", str(output), "--repetitions", "20",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "task9d_decision_report.json").read_text(encoding="utf-8"))
    assert report["preflight"]["groups_verified"] == 10
    assert report["preflight"]["total_predictions"] == 80
    assert report["preflight"]["same_manifest"] is True
    assert report["preflight"]["same_contract"] is True
    assert report["groups"]["Base"]["positive"]["samples"] == 1
    assert report["groups"]["A17"]["paired_vs_base"]["bootstrap"]["repetitions"] == 20
    assert report["decision"]["protocol_repair_passed"] is True
    assert report["decision"]["authorize_9e_recommendation"] is False
    assert (output / "metrics" / "C43.json").is_file()
    assert (output / "run_summary.json").is_file()
    completion_lines = (output / "completion.sha256").read_text(encoding="utf-8")
    assert "task9d_decision_report.json" in completion_lines
    assert "run_summary.json" in completion_lines
