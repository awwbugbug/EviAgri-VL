"""Diagnosis-only dataset, collation, and explicit loss reduction for Task 10C."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import Trainer

from task10c_contract import SYSTEM_PROMPT, TRAIN_PROMPT, strict_parse_pest_json
from task9d_data import VISUAL_TOKENS, _vision_info
from task9d_v22_loss import per_example_active_token_mean
from train_task9d import task9d_training_arguments


REDUCTION_CONTRACT = "per_example_active_token_mean_then_batch_mean"
SEEDS = (17, 29, 43)


def validate_diagnosis_record(record: dict[str, Any]) -> None:
    identifier = str(record.get("id", "<missing-id>"))
    messages = record.get("messages")
    if not isinstance(messages, list) or [row.get("role") for row in messages] != [
        "system", "user", "assistant"
    ]:
        raise ValueError(f"{identifier} requires exact system/user/assistant messages")
    system = messages[0].get("content")
    if not isinstance(system, list) or system != [{"type": "text", "text": SYSTEM_PROMPT}]:
        raise ValueError(f"{identifier} system prompt mismatch")
    user = messages[1].get("content")
    if not isinstance(user, list) or [item.get("type") for item in user] != ["image", "text"]:
        raise ValueError(f"{identifier} user content must be image then text")
    if not isinstance(user[0].get("image"), str) or not user[0]["image"]:
        raise ValueError(f"{identifier} image reference is invalid")
    if user[1].get("text") != TRAIN_PROMPT:
        raise ValueError(f"{identifier} training prompt mismatch")
    assistant = messages[2].get("content")
    if not isinstance(assistant, list) or len(assistant) != 1 or assistant[0].get("type") != "text":
        raise ValueError(f"{identifier} assistant content is invalid")
    parsed = strict_parse_pest_json(assistant[0].get("text"))
    if not parsed["schema_valid"]:
        raise ValueError(f"{identifier} assistant target is invalid")


class Task10CDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.envelopes = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.envelopes:
            raise ValueError(f"empty Task 10C dataset: {self.path}")

    def __len__(self) -> int:
        return len(self.envelopes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        envelope = self.envelopes[index]
        model = copy.deepcopy(envelope.get("model"))
        if not isinstance(model, dict):
            raise ValueError(f"Task 10C envelope lacks model row: {envelope.get('id')}")
        validate_diagnosis_record(model)
        image = Path(model["messages"][1]["content"][0]["image"])
        if not image.is_absolute():
            raise ValueError(f"Task 10C image must use frozen absolute reference: {image}")
        if not image.is_file():
            raise ValueError(f"missing Task 10C image: {image}")
        return model


class DiagnosisOnlyCollator:
    def __init__(
        self,
        processor,
        max_length: int,
        vision_info_fn: Callable = _vision_info,
    ):
        self.processor = processor
        self.max_length = int(max_length)
        self.vision_info_fn = vision_info_fn
        self.visual_token_ids = {
            processor.tokenizer.convert_tokens_to_ids(token) for token in VISUAL_TOKENS
        }
        self.visual_token_ids = {
            value for value in self.visual_token_ids if isinstance(value, int) and value >= 0
        }

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(records) != 1:
            raise ValueError("Task 10C requires per-device batch size 1")
        record = records[0]
        validate_diagnosis_record(record)
        messages = record["messages"]
        prefix = self.processor.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        full = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        images, videos = self.vision_info_fn(messages)
        common = {
            "images": images,
            "videos": videos,
            "padding": True,
            "return_tensors": "pt",
        }
        inputs = self.processor(text=[full], **common)
        prefix_inputs = self.processor(text=[prefix], **common)
        length = int(inputs["input_ids"].shape[-1])
        if length > self.max_length:
            raise ValueError(
                f"sample {record['id']} length {length} exceeds max_length {self.max_length}"
            )
        prefix_length = int(prefix_inputs["input_ids"].shape[-1])
        labels = inputs["input_ids"].clone()
        labels[:, :prefix_length] = -100
        labels[inputs["attention_mask"] == 0] = -100
        for token_id in self.visual_token_ids:
            labels[labels == token_id] = -100
        if not torch.any(labels != -100):
            raise ValueError(f"empty Task 10C assistant loss mask: {record['id']}")
        inputs["labels"] = labels
        return inputs


def compute_task10c_loss(model, inputs: dict[str, Any]):
    values = dict(inputs)
    labels = values.pop("labels")
    outputs = model(**values)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    loss, audit = per_example_active_token_mean(logits, labels)
    return loss, outputs, audit


class Task10CTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        loss, outputs, _ = compute_task10c_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss


def smoke_training_arguments(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"Task 10C smoke requires frozen seed: {seed}")
    values = task9d_training_arguments(seed=seed, mode="formal")
    values.update({
        "max_steps": 8,
        "gradient_accumulation_steps": 8,
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 1,
        "dataloader_num_workers": 0,
        "reduction_contract": REDUCTION_CONTRACT,
    })
    return values
