import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9d_v22_training import (
    V22LossCollator,
    compute_per_example_causal_loss,
    v22_training_arguments,
)


def _target(present=True):
    return {
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


def _record(present=True):
    target = _target(present)
    return {"id": "r1", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [
            {"type": "image", "image": "opaque.png"},
            {"type": "text", "text": "neutral query"},
        ]},
        {"role": "assistant", "content": [{
            "type": "text", "text": json.dumps(target, separators=(",", ":")),
        }]},
    ]}


class _CharTokenizer:
    pad_token_id = 0
    is_fast = True

    def convert_tokens_to_ids(self, token):
        return -1

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        values = [ord(char) for char in text]
        result = {"input_ids": values}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class _CharProcessor:
    tokenizer = _CharTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if add_generation_prompt:
            return "P"
        return "P" + messages[-1]["content"][0]["text"] + "S"

    def __call__(self, text, **kwargs):
        values = [ord(char) for char in text[0]]
        return {
            "input_ids": torch.tensor([values]),
            "attention_mask": torch.ones((1, len(values)), dtype=torch.long),
        }


def test_v22_collator_preserves_control_and_masks_only_positive_taxonomy_values():
    processor = _CharProcessor()
    control = V22LossCollator(
        processor, 1024, arm="Control", fast_tokenizer=_CharTokenizer(),
        vision_info_fn=lambda _: ([], []),
    )
    taxmask = V22LossCollator(
        processor, 1024, arm="TaxMask", fast_tokenizer=_CharTokenizer(),
        vision_info_fn=lambda _: ([], []),
    )
    positive = _record(True)
    control_positive = control([positive])
    taxmask_positive = taxmask([positive])
    assert taxmask.last_audit["active_tokens"] < control.last_audit["active_tokens"]
    assert taxmask.last_audit["masked_fields"] == ["pest_id", "pest_name"]
    assert torch.equal(control_positive["input_ids"], taxmask_positive["input_ids"])

    null = _record(False)
    control_null = control([null])
    taxmask_null = taxmask([null])
    assert torch.equal(control_null["labels"], taxmask_null["labels"])
    assert taxmask.last_audit["masked_token_count"] == 0


class _Model:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, **inputs):
        assert "labels" not in inputs
        return SimpleNamespace(logits=self.logits, loss=torch.tensor(999.0))


def test_training_loss_uses_explicit_per_example_reduction_not_model_default():
    logits = torch.tensor([
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    labels = torch.tensor([[-100, 0, 0, -100], [-100, 0, -100, -100]])
    loss, outputs, audit = compute_per_example_causal_loss(
        _Model(logits), {"input_ids": torch.ones_like(labels), "labels": labels},
    )
    assert loss.item() != 999.0
    assert outputs.logits is logits
    assert audit["active_tokens"].tolist() == [2, 1]


def test_v22_training_arguments_freeze_micro_steps_and_reduction_contract():
    args = v22_training_arguments(seed=29)
    assert args["per_device_train_batch_size"] == 1
    assert args["gradient_accumulation_steps"] == 8
    assert args["max_steps"] == 64
    assert args["save_steps"] == 64
    assert args["load_best_model_at_end"] is False
    assert args["reduction_contract"] == "per_example_active_token_mean_then_batch_mean"
