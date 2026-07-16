import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_task10c_c2 import (
    SCIENTIFIC_GATES,
    aggregate_learning_signal,
    condition_metrics,
    decide_c2,
    forensic_metrics,
    forensic_parse,
    pooled_source_bootstrap,
    seed_learning_signal,
)
from task10c_contract import CLASS_IDS, canonical_pest_id, strict_parse_pest_json


def _parsed(pest_id):
    return {
        "syntax_valid": pest_id is not None,
        "schema_valid": pest_id is not None,
        "pest_id": pest_id,
        "error": None if pest_id is not None else "invalid_json",
    }


def test_forensics_never_changes_strict_prediction():
    raw = '```json\n{"pest_id":"IP009"}\n```'
    strict = strict_parse_pest_json(raw)
    forensic = forensic_parse(raw)
    assert strict["schema_valid"] is False
    assert forensic["fence_stripped_schema_valid"] is True
    assert forensic["canonical_id_mentioned"] is True
    assert "prediction" not in forensic

    summary = forensic_metrics([{"raw_text": raw}, {"raw_text": "IP010"}])
    assert summary["fence_rate"] == 0.5
    assert summary["canonical_mention_rate"] == 1.0


def test_condition_metrics_are_strict_and_keep_sixteen_class_denominator():
    bands = ("head", "medium", "tail")
    rows = []
    for index, class_id in enumerate(CLASS_IDS):
        truth = canonical_pest_id(class_id)
        rows.append({
            "class_id": class_id,
            "class_band": bands[index % 3],
            "parsed": _parsed(truth if index < 8 else None),
        })
    metrics = condition_metrics(rows)
    assert metrics["count"] == 16
    assert metrics["schema_validity"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert set(metrics["band_macro_f1"]) == {"head", "medium", "tail"}
    assert sum(metrics["confusion"].values()) == 16


def _curve(delta_schema, delta_macro, delta_gain, final_gain=0.0):
    return {
        "step_8": {
            "image_schema_validity": 0.1,
            "image_macro_f1": 0.1,
            "visual_gain": final_gain - delta_gain,
        },
        "step_64": {
            "image_schema_validity": 0.1 + delta_schema,
            "image_macro_f1": 0.1 + delta_macro,
            "visual_gain": final_gain,
        },
    }


def test_learning_signal_requires_two_metrics_in_two_seeds():
    curves = {
        17: _curve(delta_schema=.30, delta_macro=.06, delta_gain=.00),
        29: _curve(delta_schema=.30, delta_macro=.00, delta_gain=.06, final_gain=.06),
        43: _curve(delta_schema=.10, delta_macro=.01, delta_gain=.01),
    }
    assert seed_learning_signal(curves[17])["passed"] is True
    assert seed_learning_signal(curves[29])["passed"] is True
    result = aggregate_learning_signal(curves)
    assert result["passed_seed_count"] == 2
    assert result["passed"] is True


def _bootstrap_rows(source_count=80, prompts=2):
    base = []
    models = {17: [], 29: [], 43: []}
    for source in range(source_count):
        truth = canonical_pest_id(CLASS_IDS[source % 16])
        for prompt in range(prompts):
            row = {
                "source_image_sha256": f"{source:064x}",
                "prompt_variant": str(prompt),
                "truth": truth,
                "prediction": truth if source % 2 == 0 else canonical_pest_id(CLASS_IDS[0]),
            }
            base.append(row)
            for seed in models:
                models[seed].append({**row, "prediction": truth})
    return base, models


def test_bootstrap_resamples_80_sources_not_prompt_or_seed_rows():
    base, models = _bootstrap_rows()
    result = pooled_source_bootstrap(base, models, repetitions=1000, seed=20260717)
    assert result["unit"] == "source_image_sha256"
    assert result["source_count"] == 80
    assert result["repetitions"] == 1000
    assert result["estimate"] > 0
    assert result == pooled_source_bootstrap(
        list(reversed(base)),
        {key: list(reversed(value)) for key, value in models.items()},
        repetitions=1000,
        seed=20260717,
    )


def _passing_evidence():
    return {
        "engineering_complete": True,
        "learning_signal": {"passed_seed_count": 3},
        "gates": {gate: True for gate in SCIENTIFIC_GATES},
    }


@pytest.mark.parametrize("failed_gate", SCIENTIFIC_GATES)
def test_decision_fails_when_any_preregistered_gate_fails(failed_gate):
    evidence = _passing_evidence()
    evidence["gates"][failed_gate] = False
    assert decide_c2(evidence)["status"] != "PASS"


def test_decision_priority_and_never_authorizes_continuation():
    evidence = _passing_evidence()
    passed = decide_c2(evidence)
    assert passed["status"] == "PASS"
    assert passed["authorize_larger_training"] is False
    assert passed["authorize_next_experiment"] is False

    evidence["engineering_complete"] = False
    assert decide_c2(evidence)["status"] == "ENGINEERING_FAILURE"

    evidence = _passing_evidence()
    evidence["gates"][SCIENTIFIC_GATES[0]] = False
    evidence["learning_signal"]["passed_seed_count"] = 2
    assert decide_c2(evidence)["status"] == "LEARNING_SIGNAL_ONLY"

    evidence["learning_signal"]["passed_seed_count"] = 1
    assert decide_c2(evidence)["status"] == "STRUCTURAL_FAILURE"
