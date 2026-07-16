"""Frozen training primitives for the Task 9D v2.2 loss-factorization micro test."""

from __future__ import annotations

import json
from typing import Any, Callable

from transformers import Trainer

from task9d_data import AssistantOnlyV2Collator, _vision_info
from task9d_v22_loss import apply_taxonomy_value_mask, per_example_active_token_mean
from train_task9d import task9d_training_arguments


REDUCTION_CONTRACT = "per_example_active_token_mean_then_batch_mean"
V22_ARMS = ("Control", "TaxMask")


class V22LossCollator(AssistantOnlyV2Collator):
    """Task9D collator with an optional positive taxonomy-value token mask."""

    def __init__(
        self,
        processor,
        max_length: int,
        *,
        arm: str,
        fast_tokenizer,
        vision_info_fn: Callable = _vision_info,
    ):
        if arm not in V22_ARMS:
            raise ValueError(f"invalid v2.2 arm: {arm}")
        if not getattr(fast_tokenizer, "is_fast", False):
            raise ValueError("v2.2 mask locator requires a fast tokenizer")
        super().__init__(processor, max_length, vision_info_fn=vision_info_fn)
        self.arm = arm
        self.fast_tokenizer = fast_tokenizer
        self.last_audit: dict[str, Any] | None = None

    def __call__(self, records):
        batch = super().__call__(records)
        record = records[0]
        target_text = record["messages"][2]["content"][0]["text"]
        masked, audit = apply_taxonomy_value_mask(
            batch["labels"], batch["input_ids"], target_text,
            slow_tokenizer=self.processor.tokenizer,
            fast_tokenizer=self.fast_tokenizer,
        )
        if self.arm == "TaxMask":
            batch["labels"] = masked
            active_tokens = audit["after_active_tokens"]
        else:
            active_tokens = audit["before_active_tokens"]
        self.last_audit = {
            **audit,
            "arm": self.arm,
            "active_tokens": active_tokens,
            "reduction_contract": REDUCTION_CONTRACT,
        }
        return batch


def compute_per_example_causal_loss(model, inputs: dict[str, Any]):
    values = dict(inputs)
    labels = values.pop("labels")
    outputs = model(**values)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    loss, audit = per_example_active_token_mean(logits, labels)
    return loss, outputs, audit


class V22PerExampleMeanTrainer(Trainer):
    """Trainer that never delegates reduction to the model's token-global loss."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        loss, outputs, _ = compute_per_example_causal_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss


def v22_training_arguments(*, seed: int) -> dict[str, Any]:
    values = task9d_training_arguments(seed=seed, mode="formal")
    values.update({
        "max_steps": 64,
        "eval_steps": 64,
        "save_steps": 64,
        "logging_steps": 8,
        "reduction_contract": REDUCTION_CONTRACT,
    })
    return values


def assistant_target_text(record: dict[str, Any]) -> str:
    text = record["messages"][2]["content"][0]["text"]
    json.loads(text)
    return text
