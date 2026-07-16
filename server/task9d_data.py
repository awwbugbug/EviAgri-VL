"""Dataset, semantic validation, and assistant-only collation for Task 9D."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import torch

from task9b_protocol import validate_target_semantics


VISUAL_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")


def _vision_info(messages):
    from qwen_vl_utils import process_vision_info
    return process_vision_info(messages)


def validate_v2_record(record: dict[str, Any]) -> dict[str, Any]:
    identifier = str(record.get("id", "<missing-id>"))
    messages = record.get("messages")
    if not isinstance(messages, list) or [row.get("role") for row in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"{identifier} must use exact system/user/assistant layout")
    user = messages[1].get("content")
    if not isinstance(user, list) or len(user) != 2 or [item.get("type") for item in user] != ["image", "text"]:
        raise ValueError(f"{identifier} must use opaque image then text content")
    image = user[0].get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"{identifier} has invalid image reference")
    assistant = messages[2].get("content")
    if not isinstance(assistant, list) or len(assistant) != 1 or assistant[0].get("type") != "text":
        raise ValueError(f"{identifier} has invalid assistant content")
    try:
        target = json.loads(assistant[0]["text"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{identifier} assistant target is not JSON") from exc
    validate_target_semantics(target)
    return target


class Task9dDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, image_root: str | Path):
        self.path, self.image_root = Path(path), Path(image_root)
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"empty Task 9D dataset: {self.path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        envelope = self.records[index]
        record = copy.deepcopy(envelope.get("model", envelope))
        record.setdefault("id", envelope.get("id"))
        validate_v2_record(record)
        reference = record["messages"][1]["content"][0]["image"]
        if Path(reference).is_absolute():
            raise ValueError("Task 9D model rows must use opaque relative image references")
        resolved = (self.image_root / reference).resolve()
        if not resolved.is_file():
            raise ValueError(f"missing Task 9D image: {resolved}")
        record["messages"][1]["content"][0]["image"] = str(resolved)
        return record


class AssistantOnlyV2Collator:
    def __init__(self, processor, max_length: int, vision_info_fn: Callable = _vision_info):
        self.processor, self.max_length, self.vision_info_fn = processor, max_length, vision_info_fn
        self.visual_token_ids = {
            processor.tokenizer.convert_tokens_to_ids(token) for token in VISUAL_TOKENS
        }
        self.visual_token_ids = {value for value in self.visual_token_ids if isinstance(value, int) and value >= 0}

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(records) != 1:
            raise ValueError("Task 9D requires per-device batch size 1")
        record = records[0]
        validate_v2_record(record)
        messages = record["messages"]
        prefix = self.processor.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        full = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images, videos = self.vision_info_fn(messages)
        common = {"images": images, "videos": videos, "padding": True, "return_tensors": "pt"}
        inputs = self.processor(text=[full], **common)
        prefix_inputs = self.processor(text=[prefix], **common)
        length = int(inputs["input_ids"].shape[-1])
        if length > self.max_length:
            raise ValueError(f"sample {record['id']} length {length} exceeds max_length {self.max_length}")
        prefix_length = int(prefix_inputs["input_ids"].shape[-1])
        labels = inputs["input_ids"].clone()
        labels[:, :prefix_length] = -100
        labels[inputs["attention_mask"] == 0] = -100
        for token_id in self.visual_token_ids:
            labels[labels == token_id] = -100
        if not torch.any(labels != -100):
            raise ValueError(f"empty assistant loss mask: {record['id']}")
        inputs["labels"] = labels
        return inputs
