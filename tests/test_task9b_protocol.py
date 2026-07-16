import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9b_protocol import (
    DEV_TEMPLATE_IDS,
    OUTPUT_KEYS,
    SYSTEM_PROMPT,
    TRAIN_TEMPLATE_IDS,
    build_prompt,
    build_target,
    length_bucket_for_family,
    opaque_id,
    serialize_target,
    validate_target_semantics,
)


def test_system_prompt_is_fixed_neutral_and_contains_no_role_marker():
    assert isinstance(SYSTEM_PROMPT, str) and SYSTEM_PROMPT
    assert all(token not in SYSTEM_PROMPT.lower() for token in ("positive", "null sample", "task_type"))


def test_train_and_dev_template_ids_are_disjoint_and_not_task8_prompt():
    assert len(TRAIN_TEMPLATE_IDS) == 3
    assert len(DEV_TEMPLATE_IDS) == 2
    assert set(TRAIN_TEMPLATE_IDS).isdisjoint(DEV_TEMPLATE_IDS)
    task8 = "Is rice leaf roller visibly present in this image?"
    for template_id in (*TRAIN_TEMPLATE_IDS, *DEV_TEMPLATE_IDS):
        prompt = build_prompt(template_id, "rice leaf roller")
        assert task8 not in prompt
        assert "positive" not in prompt.lower()
        assert "null sample" not in prompt.lower()
        assert "task_type" not in prompt.lower()
        assert "/root/" not in prompt


def test_prompt_builder_has_no_label_or_role_input_and_is_byte_stable():
    prompt_a = build_prompt(TRAIN_TEMPLATE_IDS[0], "rice leaf roller")
    prompt_b = build_prompt(TRAIN_TEMPLATE_IDS[0], "rice leaf roller")

    assert prompt_a.encode("utf-8") == prompt_b.encode("utf-8")
    assert "evidence_region" in prompt_a
    assert "uncertain" in prompt_a and "abstain" in prompt_a


def test_positive_and_null_targets_have_same_keys_and_fixed_diagnosis_shape():
    positive = build_target(
        evidence_present=True,
        evidence_region=[1, 2, 30, 40],
        pest_id=7,
        pest_name="rice leaf roller",
    )
    null = build_target(
        evidence_present=False,
        evidence_region=None,
        pest_id=None,
        pest_name=None,
        abstention_status="uncertain",
    )

    assert tuple(positive) == OUTPUT_KEYS
    assert tuple(null) == OUTPUT_KEYS
    assert tuple(positive["diagnosis"]) == tuple(null["diagnosis"])
    assert positive["diagnosis"]["species"] is None
    assert positive["diagnosis"]["stage"] is None
    assert null["diagnosis"] == {
        "status": "uncertain",
        "pest_id": None,
        "pest_name": None,
        "species": None,
        "stage": None,
    }
    validate_target_semantics(positive)
    validate_target_semantics(null)


@pytest.mark.parametrize("bad_status", ["supported", "rice leaf roller", "unknown-species"])
def test_null_target_rejects_specific_or_non_abstaining_diagnosis(bad_status):
    target = build_target(
        evidence_present=False,
        evidence_region=None,
        pest_id=None,
        pest_name=None,
        abstention_status="abstain",
    )
    target["diagnosis"]["status"] = bad_status

    with pytest.raises(ValueError, match="null diagnosis"):
        validate_target_semantics(target)


def test_serialized_target_length_bucket_is_label_independent_and_valid_json():
    family_id = "family-a"
    bucket = length_bucket_for_family(family_id)
    positive = build_target(True, [1, 2, 30, 40], 7, "rice leaf roller")
    null = build_target(False, None, None, None, abstention_status="abstain")

    positive_text = serialize_target(positive, bucket)
    null_text = serialize_target(null, bucket)

    assert bucket in {320, 352, 384}
    assert len(positive_text) == len(null_text) == bucket
    assert json.loads(positive_text) == positive
    assert json.loads(null_text) == null


def test_opaque_ids_do_not_expose_role_class_or_path_tokens():
    values = {
        opaque_id("seed", "family-a", role)
        for role in ("positive", "semantic_negative", "synthetic_null")
    }

    assert len(values) == 3
    for value in values:
        assert re.fullmatch(r"[0-9a-f]{32}", value)
        lowered = value.lower()
        assert all(token not in lowered for token in ("positive", "null", "semantic", "rice", "/"))
