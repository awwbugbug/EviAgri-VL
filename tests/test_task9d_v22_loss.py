import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9d_v22_loss import (
    apply_taxonomy_value_mask,
    audit_loss_mass_equivalence,
    per_example_active_token_mean,
    taxonomy_value_character_spans,
)


def _target(present: bool = True) -> str:
    value = {
        "evidence_present": present,
        "evidence_region": [1, 2, 3, 4] if present else None,
        "visible_attributes": [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": 49 if present else None,
            "pest_name": "lytta polita" if present else None,
            "species": None,
            "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def test_taxonomy_spans_cover_only_positive_value_literals():
    text = _target(True)
    spans = taxonomy_value_character_spans(text)
    assert [(field, text[start:end]) for field, start, end in spans] == [
        ("pest_id", "49"),
        ("pest_name", '"lytta polita"'),
    ]
    assert taxonomy_value_character_spans(_target(False)) == []


def test_per_example_active_token_mean_does_not_weight_longer_example_more():
    # Example 0 has two active tokens with losses near [0, 4]; example 1 has
    # one active token with loss near 2. The required reduction is
    # mean(mean(example_0), mean(example_1)), not a global token mean.
    logits = torch.tensor([
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    labels = torch.tensor([
        [-100, 0, 0, -100],
        [-100, 0, -100, -100],
    ])
    loss, audit = per_example_active_token_mean(logits, labels)
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    raw = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, 2), shifted_labels.view(-1),
        ignore_index=-100, reduction="none",
    ).view(2, -1)
    expected = torch.stack([
        raw[0][shifted_labels[0] != -100].mean(),
        raw[1][shifted_labels[1] != -100].mean(),
    ]).mean()
    assert torch.allclose(loss, expected)
    assert audit["active_tokens"].tolist() == [2, 1]
    assert audit["per_example_loss_weight"].tolist() == [0.5, 0.5]


def test_loss_mass_audit_requires_identical_role_gradient_shares():
    control = {
        "positive": {"samples": 4, "active_tokens": [20, 21, 22, 23]},
        "semantic_negative": {"samples": 4, "active_tokens": [18, 18, 19, 19]},
        "visual_counterfactual": {"samples": 4, "active_tokens": [17, 17, 18, 18]},
    }
    taxmask = {
        "positive": {"samples": 4, "active_tokens": [12, 13, 14, 15]},
        "semantic_negative": {"samples": 4, "active_tokens": [18, 18, 19, 19]},
        "visual_counterfactual": {"samples": 4, "active_tokens": [17, 17, 18, 18]},
    }
    report = audit_loss_mass_equivalence(control, taxmask, gradient_accumulation_steps=8)
    assert report["passed"] is True
    assert report["arms"]["Control"]["positive"]["mean_example_loss_weight"] == 1.0
    assert report["arms"]["TaxMask"]["positive"]["mean_example_loss_weight"] == 1.0
    assert report["arms"]["Control"]["positive"]["normalized_total_gradient_weight"] == pytest.approx(1 / 3)
    assert report["arms"]["TaxMask"]["positive"]["normalized_total_gradient_weight"] == pytest.approx(1 / 3)

    taxmask["positive"]["samples"] = 3
    taxmask["positive"]["active_tokens"] = [12, 13, 14]
    blocked = audit_loss_mass_equivalence(control, taxmask, gradient_accumulation_steps=8)
    assert blocked["passed"] is False
    assert "role sample counts differ" in blocked["block_reasons"][0]


class _CharacterTokenizer:
    def __init__(self, *, id_offset: int = 0):
        self.id_offset = id_offset

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        result = {"input_ids": [ord(char) + self.id_offset for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, values, skip_special_tokens=False):
        return "".join(chr(value - self.id_offset) for value in values)


def test_taxmask_masks_only_positive_taxonomy_values_in_full_sequence():
    text = _target(True)
    tokenizer = _CharacterTokenizer()
    target_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([[999, *target_ids, 998]])
    labels = input_ids.clone()
    labels[:, 0] = -100
    masked, audit = apply_taxonomy_value_mask(
        labels, input_ids, text,
        slow_tokenizer=tokenizer, fast_tokenizer=tokenizer,
    )
    masked_positions = {index - 1 for index, value in enumerate(masked[0].tolist()) if index > 0 and value == -100}
    expected = set()
    for _, start, end in taxonomy_value_character_spans(text):
        expected.update(range(start, end))
    assert masked_positions == expected
    assert audit["before_active_tokens"] - audit["after_active_tokens"] == len(expected)
    assert audit["masked_fields"] == ["pest_id", "pest_name"]
    assert masked[0, -1].item() == 998


def test_taxmask_leaves_null_unchanged_and_blocks_tokenizer_mismatch():
    text = _target(False)
    tokenizer = _CharacterTokenizer()
    target_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([[999, *target_ids, 998]])
    labels = input_ids.clone()
    labels[:, 0] = -100
    masked, audit = apply_taxonomy_value_mask(
        labels, input_ids, text,
        slow_tokenizer=tokenizer, fast_tokenizer=tokenizer,
    )
    assert torch.equal(masked, labels)
    assert audit["masked_token_count"] == 0

    with pytest.raises(ValueError, match="fast/slow tokenizer IDs differ"):
        apply_taxonomy_value_mask(
            labels, input_ids, _target(True),
            slow_tokenizer=_CharacterTokenizer(),
            fast_tokenizer=_CharacterTokenizer(id_offset=1),
        )
