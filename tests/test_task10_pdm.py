import copy
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from audit_task10_pdm import (
    assistant_token_spans,
    build_condition_messages,
    canonical_positive_target,
    find_token_subsequence,
    hellinger_from_logits,
    summarize_pdm_records,
    teacher_forced_group_pdm,
)


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _manifest_row(condition="original", image="original.png"):
    role = "positive" if condition == "original" else "visual_counterfactual"
    return {
        "id": f"f01-{condition}",
        "family_id": "f01",
        "role": role,
        "condition": condition,
        "prompt_view": "canonical",
        "query_class_id": 12,
        "gt_bbox": [1, 2, 30, 40] if condition == "original" else None,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Using only the pixels, decide whether the image supports "
                    "the queried pest 'aphid'. Return JSON."
                )},
            ]},
        ],
    }


def test_hellinger_identical_distributions_is_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    assert hellinger_from_logits(logits, logits).item() == pytest.approx(0.0, abs=1e-7)


def test_hellinger_is_symmetric_and_bounded():
    left = torch.tensor([[8.0, 0.0, -2.0]])
    right = torch.tensor([[-2.0, 0.0, 8.0]])

    forward = hellinger_from_logits(left, right).item()

    assert forward == pytest.approx(hellinger_from_logits(right, left).item())
    assert 0.0 <= forward <= 1.0


def test_hellinger_rejects_non_finite_logits():
    with pytest.raises(FloatingPointError, match="non-finite"):
        hellinger_from_logits(
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        )


def test_json_spans_identify_value_tokens_without_field_names():
    target = (
        '{"evidence_present":true,"evidence_region":[1,2,3,4],'
        '"visible_attributes":[],"diagnosis":{"status":"supported",'
        '"pest_id":12,"pest_name":"aphid","species":null,"stage":null},'
        '"reliability":"supported"}'
    )

    spans = assistant_token_spans(CharacterTokenizer(), target)

    assert spans.coverage == pytest.approx(1.0)
    assert "true" == "".join(target[index] for index in spans.groups["evidence_present"])
    assert "[1,2,3,4]" == "".join(target[index] for index in spans.groups["bbox_value"])
    taxonomy = "".join(target[index] for index in spans.groups["taxonomy_value"])
    assert "12" in taxonomy and '"aphid"' in taxonomy
    evidence_key_start = target.index('"evidence_present"')
    assert evidence_key_start not in spans.groups["evidence_present"]
    assert set().union(*map(set, spans.groups.values())) == set(range(len(target)))


def test_refusal_value_tokens_are_grouped_for_null_target():
    target = (
        '{"evidence_present":false,"evidence_region":null,"visible_attributes":[],'
        '"diagnosis":{"status":"abstain","pest_id":null,"pest_name":null,'
        '"species":null,"stage":null},"reliability":"insufficient_visual_evidence"}'
    )

    spans = assistant_token_spans(CharacterTokenizer(), target)
    refusal = "".join(target[index] for index in spans.groups["refusal_uncertain"])

    assert '"abstain"' in refusal
    assert '"insufficient_visual_evidence"' in refusal


def test_find_token_subsequence_is_exact_and_rejects_ambiguity():
    assert find_token_subsequence([9, 1, 2, 3, 8], [1, 2, 3]) == 1
    with pytest.raises(ValueError, match="exactly once"):
        find_token_subsequence([1, 2, 1, 2], [1, 2])


def test_canonical_positive_target_reconstructs_frozen_json_contract():
    target = json.loads(canonical_positive_target(_manifest_row()))

    assert target["evidence_present"] is True
    assert target["evidence_region"] == [1, 2, 30, 40]
    assert target["diagnosis"]["pest_id"] == 12
    assert target["diagnosis"]["pest_name"] == "aphid"


def test_condition_messages_reuse_positive_text_and_target_but_swap_pixels():
    original = _manifest_row()
    blank = _manifest_row(condition="blank", image="blank.png")

    conditioned, unconditioned, target = build_condition_messages(original, blank)

    assert conditioned[1]["content"][0]["image"] == "blank.png"
    assert conditioned[1]["content"][1] == original["messages"][1]["content"][1]
    assert conditioned[2]["content"][0]["text"] == target
    assert [item["type"] for item in unconditioned[1]["content"]] == ["text"]
    assert unconditioned[1]["content"][0] == original["messages"][1]["content"][1]
    assert original == _manifest_row()


def test_teacher_forced_group_pdm_compares_same_target_without_gradients():
    target = canonical_positive_target(_manifest_row())
    spans = assistant_token_spans(CharacterTokenizer(), target)
    conditioned, unconditioned, _ = build_condition_messages(
        _manifest_row(), _manifest_row()
    )

    class FakeProcessor:
        tokenizer = CharacterTokenizer()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[2]["content"][0]["text"]

        def __call__(self, *, text, images, videos, padding, return_tensors):
            ids = [2, 3] + [ord(character) for character in text[0]] + [4]
            return {
                "input_ids": torch.tensor([ids]),
                "visual_marker": torch.tensor([1.0 if images is not None else 0.0]),
            }

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.grad_states = []

        def __call__(self, input_ids, visual_marker):
            self.grad_states.append(torch.is_grad_enabled())
            logits = torch.zeros((1, input_ids.shape[1], 128))
            if visual_marker.item() == 1.0:
                logits[..., 7] = 4.0
            return type("Output", (), {"logits": logits})()

    model = FakeModel()
    measurement = teacher_forced_group_pdm(
        model,
        FakeProcessor(),
        conditioned,
        unconditioned,
        spans,
        vision_info_fn=lambda messages: ([object()], []),
    )

    assert model.grad_states == [False, False]
    assert measurement["finite"] is True
    assert measurement["normalization_error"] <= 1e-5
    assert measurement["group_mean_pdm_h"]["evidence_present"] > 0
    assert "logits" not in measurement


def _observation(seed, family, condition, taxonomy, evidence):
    return {
        "seed": seed,
        "family_id": family,
        "condition": condition,
        "coverage": 1.0,
        "finite": True,
        "normalization_error": 1e-7,
        "group_mean_pdm_h": {
            "taxonomy_value": taxonomy,
            "evidence_present": evidence,
            "bbox_value": evidence,
            "refusal_uncertain": 0.0,
            "other_assistant": 0.01,
        },
    }


def test_pdm_summary_passes_positive_family_bootstrap_visual_gain():
    rows = []
    for seed in (17, 29, 43):
        for index in range(32):
            family = f"f{index:02d}"
            rows.extend([
                _observation(seed, family, "original", 0.50, 0.40),
                _observation(seed, family, "blank", 0.20, 0.15),
                _observation(seed, family, "shuffle", 0.25, 0.10),
            ])

    report = summarize_pdm_records(rows, repetitions=1000, bootstrap_seed=20260717)

    assert report["quality_passed"] is True
    assert report["visual_dependency_passed"] is True
    assert report["pooled"]["taxonomy_value"]["original_minus_blank"]["low"] > 0
    assert report["family_count"] == 32


def test_pdm_summary_blocks_missing_family_and_bad_normalization():
    rows = []
    for seed in (17, 29, 43):
        for index in range(31):
            family = f"f{index:02d}"
            rows.extend([
                _observation(seed, family, "original", 0.50, 0.40),
                _observation(seed, family, "blank", 0.20, 0.15),
                _observation(seed, family, "shuffle", 0.25, 0.10),
            ])
    rows[0] = copy.deepcopy(rows[0])
    rows[0]["normalization_error"] = 1e-3

    report = summarize_pdm_records(rows, repetitions=100, bootstrap_seed=1)

    assert report["quality_passed"] is False
    assert report["visual_dependency_passed"] is False
    assert "expected_32_families_got_31" in report["quality_failures"]
    assert "normalization_error_above_1e-5" in report["quality_failures"]
