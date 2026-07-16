import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task8_protocol import (
    EXPECTED_SCHEMA_KEYS,
    GROUPS,
    build_prompt,
    generation_kwargs,
    protocol_hash,
)


def test_b1_b2_are_byte_identical_under_the_neutral_protocol():
    b1 = build_prompt("B1", "Miridae", "original_correct")
    b2 = build_prompt("B2", "Miridae", "original_correct")

    assert b1 == b2
    assert protocol_hash("B1") == protocol_hash("B2")


def test_registered_prompt_treatments_are_the_only_prompt_differences():
    prompts = {
        group: build_prompt(group, "Miridae", "original_correct") for group in GROUPS
    }

    assert prompts["B0"] != prompts["B1"]
    assert prompts["B2"] != prompts["B3"]
    assert protocol_hash("B0") != protocol_hash("B1")
    assert protocol_hash("B2") != protocol_hash("B3")


def test_neutral_prompt_contains_no_hidden_split_or_target_state_tokens():
    prompt = build_prompt("B1", "Miridae", "original_wrong_query").lower()

    for forbidden in (
        "positive sample",
        "null sample",
        "task_type",
        "split=",
        ".jpg",
        "/root/",
    ):
        assert forbidden not in prompt


def test_b3_reproduces_the_registered_positive_and_query_templates():
    positive = build_prompt("B3", "Miridae", "original_correct")
    negative = build_prompt("B3", "Miridae", "original_wrong_query")

    assert positive.startswith("Identify the pest supported by visible evidence.")
    assert negative.startswith("Is Miridae visibly present in this image?")


def test_schema_and_generation_settings_are_frozen():
    assert EXPECTED_SCHEMA_KEYS == (
        "evidence_present",
        "evidence_bbox",
        "visible_attributes",
        "diagnosis",
        "reliability",
    )
    assert generation_kwargs() == {
        "max_new_tokens": 128,
        "do_sample": False,
        "temperature": None,
    }


def test_neutral_prompt_defines_nested_diagnosis_schema_and_query_id():
    prompt = build_prompt("B1", "Miridae", "original_correct", queried_pest_id=37)

    assert '"pest_id":37' in prompt
    assert '"pest_name":"Miridae"' in prompt
    assert 'diagnosis must be either' in prompt
    assert prompt.index('"pest_id":37') < prompt.index("If visible evidence is insufficient")
    assert prompt.endswith(
        'reliability="insufficient_visual_evidence".'
    )
