import json
from pathlib import Path
from typing import Any, Callable

import torch


EXPECTED_TARGET_KEYS = (
    "evidence_present",
    "evidence_bbox",
    "visible_attributes",
    "diagnosis",
    "reliability",
)
VISUAL_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")


def _default_process_vision_info(messages):
    from qwen_vl_utils import process_vision_info

    return process_vision_info(messages)


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {self.path}:{line_number}: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"record at {self.path}:{line_number} must be an object")
                self.records.append(row)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def validate_evidence_record(record: dict[str, Any]) -> None:
    record_id = record.get("id", "<missing-id>")
    image = record.get("image")
    if not isinstance(image, str) or not Path(image).is_file():
        raise ValueError(f"missing image for {record_id}: {image}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{record_id} must contain one user and one assistant message")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError(f"invalid message roles for {record_id}")
    assistant_content = messages[1].get("content")
    if not isinstance(assistant_content, list) or not assistant_content:
        raise ValueError(f"missing assistant content for {record_id}")
    assistant_text = assistant_content[0].get("text")
    if not isinstance(assistant_text, str):
        raise ValueError(f"missing assistant JSON text for {record_id}")
    try:
        assistant_target = json.loads(assistant_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid assistant JSON for {record_id}: {error}") from error
    if not isinstance(assistant_target, dict) or tuple(assistant_target) != EXPECTED_TARGET_KEYS:
        raise ValueError(f"invalid Evidence-First key order for {record_id}")
    if record.get("target") != assistant_target:
        raise ValueError(f"assistant JSON does not match target for {record_id}")


class AssistantOnlyVisionCollator:
    def __init__(
        self,
        processor,
        max_length: int,
        vision_info_fn: Callable = _default_process_vision_info,
    ):
        self.processor = processor
        self.max_length = max_length
        self.vision_info_fn = vision_info_fn
        self.visual_token_ids = {
            processor.tokenizer.convert_tokens_to_ids(token) for token in VISUAL_TOKENS
        }
        self.visual_token_ids = {
            token_id for token_id in self.visual_token_ids if isinstance(token_id, int) and token_id >= 0
        }
        self.last_prefix_length = 0

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(records) != 1:
            raise ValueError("static_qlora_v1 requires per-device batch size 1")
        record = records[0]
        validate_evidence_record(record)
        messages = record["messages"]
        prefix = self.processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        full = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = self.vision_info_fn(messages)
        common = {
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        model_inputs = self.processor(text=[full], **common)
        prefix_inputs = self.processor(text=[prefix], **common)
        sequence_length = int(model_inputs["input_ids"].shape[-1])
        if sequence_length > self.max_length:
            raise ValueError(
                f"sample {record['id']} length {sequence_length} exceeds max_length {self.max_length}"
            )
        prefix_length = int(prefix_inputs["input_ids"].shape[-1])
        labels = model_inputs["input_ids"].clone()
        labels[:, :prefix_length] = -100
        labels[model_inputs["attention_mask"] == 0] = -100
        for token_id in self.visual_token_ids:
            labels[labels == token_id] = -100
        if not torch.any(labels != -100):
            raise ValueError(f"empty assistant loss mask: {record['id']}")
        model_inputs["labels"] = labels
        self.last_prefix_length = prefix_length
        return model_inputs


def preflight_dataset(
    dataset: torch.utils.data.Dataset,
    collator: AssistantOnlyVisionCollator,
    max_length: int,
    progress_every: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    lengths: list[int] = []
    assistant_lengths: list[int] = []
    for index in range(len(dataset)):
        record = dataset[index]
        try:
            batch = collator([record])
        except Exception as error:
            raise ValueError(f"preflight failed for {record.get('id', index)}: {error}") from error
        length = int(batch["input_ids"].shape[-1])
        if length > max_length:
            raise ValueError(
                f"preflight max_length violation for {record.get('id', index)}: {length} > {max_length}"
            )
        lengths.append(length)
        assistant_lengths.append(int((batch["labels"] != -100).sum().item()))
        done = index + 1
        if progress_callback is not None and (
            progress_every is None or done % progress_every == 0 or done == len(dataset)
        ):
            progress_callback(done, len(dataset), str(record.get("id", index)))
    if not lengths:
        raise ValueError("preflight dataset is empty")
    return {
        "samples": len(lengths),
        "token_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "assistant_tokens": {
            "min": min(assistant_lengths),
            "max": max(assistant_lengths),
            "mean": sum(assistant_lengths) / len(assistant_lengths),
        },
    }
