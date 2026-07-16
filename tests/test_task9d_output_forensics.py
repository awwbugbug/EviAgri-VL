import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "server" / "audit_task9d_outputs.py"


def _module():
    assert MODULE_PATH.is_file(), "audit_task9d_outputs.py has not been implemented"
    spec = importlib.util.spec_from_file_location("audit_task9d_outputs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tolerant_parser_recovers_fenced_json_but_preserves_schema_failure():
    audit = _module()
    raw = """```json
{"evidence_present":true,"evidence_region":"center","visible_attributes":[],
 "diagnosis":{"status":"confirmed","pest":"Miridae"},"reliability":"high"}
```"""
    result = audit.analyze_raw_output(raw)
    assert result["strict_syntax_valid"] is False
    assert result["tolerant_json_recovered"] is True
    assert result["strict_schema_valid"] is False
    assert "invalid_evidence_region" in result["issues"]
    assert "invalid_diagnosis_keys" in result["issues"]
    assert "invalid_reliability" in result["issues"]


def _manifest_row(identifier, role, condition, query_id=9, query_name="Miridae"):
    return {
        "id": identifier,
        "family_id": f"family-{identifier}",
        "role": role,
        "condition": condition,
        "prompt_view": "canonical",
        "query_class_id": query_id,
        "messages": [{"role": "user", "content": [
            {"type": "image", "image": "opaque.jpg"},
            {"type": "text", "text": f"Using only pixels, decide whether the image supports the queried pest '{query_name}'."},
        ]}],
    }


def _prediction(identifier, present, pest_id=None, pest_name=None):
    value = {
        "evidence_present": present,
        "evidence_region": [0, 0, 10, 10] if present else None,
        "visible_attributes": [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": pest_id if present else None,
            "pest_name": pest_name if present else None,
            "species": None,
            "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }
    return {"id": identifier, "raw_text": json.dumps(value, separators=(",", ":"))}


def test_positive_forensics_separate_query_name_echo_from_numeric_id_accuracy():
    audit = _module()
    manifest = [_manifest_row(f"p{index}", "positive", "original") for index in range(4)]
    predictions = [
        _prediction(f"p{index}", True, pest_id=index + 1, pest_name="Miridae")
        for index in range(4)
    ]
    result = audit.analyze_group_outputs(manifest, predictions)
    positive = result["canonical_positive"]
    assert positive["samples"] == 4
    assert positive["evidence_present_rate"] == 1.0
    assert positive["query_name_echo_rate"] == 1.0
    assert positive["exact_id_accuracy"] == 0.0
    assert positive["conditional_id_accuracy_given_evidence"] == 0.0
    assert positive["unique_predicted_ids"] == 4
    assert positive["top1_predicted_id_share"] == 0.25
    assert positive["predicted_name_to_id_consistency"] == 0.25
    assert positive["normalized_predicted_id_entropy"] == 1.0


def test_null_forensics_separate_semantic_echo_from_visual_refusal():
    audit = _module()
    manifest = [
        _manifest_row("sem", "semantic_negative", "semantic_null"),
        _manifest_row("blank", "visual_counterfactual", "blank"),
    ]
    predictions = [
        _prediction("sem", True, pest_id=9, pest_name="Miridae"),
        _prediction("blank", False),
    ]
    result = audit.analyze_group_outputs(manifest, predictions)
    assert result["null_by_condition"]["semantic_null"]["concrete_diagnosis_rate"] == 1.0
    assert result["null_by_condition"]["semantic_null"]["query_name_echo_rate"] == 1.0
    assert result["null_by_condition"]["semantic_null"]["refusal_rate"] == 0.0
    assert result["null_by_condition"]["blank"]["refusal_rate"] == 1.0
    assert result["null_by_condition"]["blank"]["concrete_diagnosis_rate"] == 0.0


def _schedule_positive(identifier, query_name, pest_id, pest_name=None):
    target = _prediction(identifier, True, pest_id=pest_id, pest_name=pest_name or query_name)
    return {
        "id": identifier,
        "role": "positive",
        "model": {"messages": [
            _manifest_row(identifier, "positive", "original", query_name=query_name)["messages"][0],
            {"role": "assistant", "content": [{"type": "text", "text": target["raw_text"]}]},
        ]},
    }


def test_label_map_forensics_detect_one_to_many_name_id_corruption():
    audit = _module()
    clean = audit.assess_label_map([
        _schedule_positive("a", "Miridae", 9),
        _schedule_positive("b", "Miridae", 9),
    ])
    assert clean["consistent"] is True
    assert clean["query_target_name_match_rate"] == 1.0
    corrupted = audit.assess_label_map([
        _schedule_positive("a", "Miridae", 9),
        _schedule_positive("b", "Miridae", 10),
    ])
    assert corrupted["consistent"] is False
    assert corrupted["names_with_multiple_ids"] == {"miridae": [9, 10]}


def test_dominant_cause_gate_prioritizes_clean_map_numeric_id_bottleneck():
    audit = _module()
    group_reports = {
        group: {
            "canonical_positive": {
                "query_name_echo_rate": 0.95,
                "exact_id_accuracy": 0.02,
                "predicted_name_to_id_consistency": 0.20,
                "evidence_present_rate": 0.90,
            },
            "format": {"tolerant_recovery_gain": 0.0},
        }
        for group in ("A17", "A29", "A43", "B17", "B29", "B43", "C17", "C29", "C43")
    }
    clean_map = {"consistent": True}
    result = audit.choose_dominant_cause(group_reports, clean_map)
    assert result["value"] == "numeric_id_generation_bottleneck"
    assert result["evidence"]["mean_query_name_echo_rate"] == pytest.approx(0.95)
    assert result["evidence"]["mean_exact_id_accuracy"] == pytest.approx(0.02)


def test_dominant_cause_gate_blocks_on_label_map_corruption_first():
    audit = _module()
    result = audit.choose_dominant_cause({}, {"consistent": False})
    assert result["value"] == "label_map_corruption"


def test_atomic_writer_emits_report_cases_summary_and_verifiable_hashes(tmp_path):
    audit = _module()
    report = {"version": "test", "dominant_cause": {"value": "numeric_id_generation_bottleneck"}}
    cases = [{"id": "opaque", "group": "A17", "case_type": "name_echo_id_mismatch"}]
    audit.write_forensics_artifacts(tmp_path, report, cases)
    for name in ("forensics_report.json", "forensics_cases.jsonl", "run_summary.json", "completion.sha256"):
        assert (tmp_path / name).is_file()
    for line in (tmp_path / "completion.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "completed"
    assert summary["case_count"] == 1
