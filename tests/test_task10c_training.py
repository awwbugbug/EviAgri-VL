import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10c_contract import SYSTEM_PROMPT, TRAIN_PROMPT
from task10c_training import (
    REDUCTION_CONTRACT,
    DiagnosisOnlyCollator,
    Task10CDataset,
    compute_task10c_loss,
    smoke_training_arguments,
    validate_diagnosis_record,
)


def _record(class_id: int = 9, image: str = "opaque.png") -> dict:
    return {
        "id": "row-1",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TRAIN_PROMPT},
            ]},
            {"role": "assistant", "content": [{
                "type": "text", "text": json.dumps(
                    {"pest_id": f"IP{class_id:03d}"}, separators=(",", ":")
                ),
            }]},
        ],
    }


class _CharTokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return -1


class _CharProcessor:
    tokenizer = _CharTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if add_generation_prompt:
            return "PREFIX"
        return "PREFIX" + messages[-1]["content"][0]["text"] + "SUFFIX"

    def __call__(self, text, **kwargs):
        values = [ord(char) for char in text[0]]
        return {
            "input_ids": torch.tensor([values]),
            "attention_mask": torch.ones((1, len(values)), dtype=torch.long),
        }


def test_validate_record_requires_exact_prompts_and_compact_target():
    validate_diagnosis_record(_record())

    wrong = _record()
    wrong["messages"][1]["content"][1]["text"] = "candidate IP009"
    with pytest.raises(ValueError, match="training prompt mismatch"):
        validate_diagnosis_record(wrong)

    spaced = _record()
    spaced["messages"][2]["content"][0]["text"] = '{ "pest_id": "IP009" }'
    with pytest.raises(ValueError, match="assistant target"):
        validate_diagnosis_record(spaced)


def test_dataset_returns_only_model_record_and_requires_existing_image(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    envelope = {
        "id": "row-1",
        "class_id": 9,
        "source_image_id": "must-not-enter-model",
        "model": _record(image=str(image)),
    }
    path = tmp_path / "smoke.jsonl"
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    dataset = Task10CDataset(path)

    assert len(dataset) == 1
    assert dataset[0] == envelope["model"]
    assert "source_image_id" not in dataset[0]

    image.unlink()
    with pytest.raises(ValueError, match="missing Task 10C image"):
        Task10CDataset(path)[0]


def test_collator_masks_prompt_and_keeps_only_assistant_target():
    collator = DiagnosisOnlyCollator(
        _CharProcessor(), 1024, vision_info_fn=lambda _: ([], [])
    )
    batch = collator([_record()])
    active = batch["labels"][0][batch["labels"][0] != -100]
    active_text = "".join(chr(value) for value in active.tolist())

    assert "IP009" in active_text
    assert TRAIN_PROMPT not in active_text
    assert torch.all(batch["labels"][0, : len("PREFIX")] == -100)

    with pytest.raises(ValueError, match="batch size 1"):
        collator([_record(), _record()])


class _Model:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, **inputs):
        assert "labels" not in inputs
        return SimpleNamespace(logits=self.logits, loss=torch.tensor(999.0))


def test_loss_ignores_model_default_and_uses_per_example_active_token_mean():
    logits = torch.tensor([
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    labels = torch.tensor([[-100, 0, 0, -100], [-100, 0, -100, -100]])

    loss, outputs, audit = compute_task10c_loss(
        _Model(logits), {"input_ids": torch.ones_like(labels), "labels": labels}
    )

    assert loss.item() != 999.0
    assert outputs.logits is logits
    assert audit["active_tokens"].tolist() == [2, 1]


def test_smoke_arguments_are_exactly_eight_steps_and_sixty_four_exposures():
    args = smoke_training_arguments(29)
    assert args["per_device_train_batch_size"] == 1
    assert args["gradient_accumulation_steps"] == 8
    assert args["max_steps"] == 8
    assert args["learning_rate"] == 1e-4
    assert args["eval_strategy"] == "no"
    assert args["save_strategy"] == "no"
    assert args["logging_steps"] == 1
    assert args["dataloader_num_workers"] == 0
    assert args["reduction_contract"] == REDUCTION_CONTRACT
    with pytest.raises(ValueError, match="frozen seed"):
        smoke_training_arguments(31)
