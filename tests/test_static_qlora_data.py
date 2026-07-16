import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from static_qlora_data import AssistantOnlyVisionCollator, JsonlDataset, preflight_dataset


class FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|vision_start|>": 90,
            "<|vision_end|>": 91,
            "<|image_pad|>": 99,
            "<|video_pad|>": 98,
        }[token]


class FakeProcessor:
    def __init__(self, full_ids: list[int] | None = None):
        self.tokenizer = FakeTokenizer()
        self.full_ids = full_ids or [10, 11, 99, 20, 21, 0]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if len(messages) == 1 and add_generation_prompt:
            return "PREFIX"
        return "FULL"

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        ids = [10, 11, 99] if text == ["PREFIX"] else self.full_ids
        attention = [0 if token == 0 else 1 for token in ids]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attention]),
        }


def fixture_record(image: Path, record_id: str = "fixture-positive") -> dict:
    target = {
        "evidence_present": True,
        "evidence_bbox": [1, 2, 10, 12],
        "visible_attributes": [],
        "diagnosis": {"pest_id": 0, "pest_name": "fixture pest"},
        "reliability": "supported",
    }
    return {
        "id": record_id,
        "image": str(image),
        "target": target,
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": str(image)}, {"type": "text", "text": "diagnose"}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]},
        ],
    }


def fake_vision_info(_messages):
    return ["image-input"], None


class StaticQloraDataTest(unittest.TestCase):
    def test_jsonl_dataset_loads_records(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            image = tmp / "image.jpg"
            image.write_bytes(b"fixture")
            record = fixture_record(image)
            path = tmp / "data.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = JsonlDataset(path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["id"], "fixture-positive")

    def test_collator_masks_user_padding_and_visual_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            image = Path(tmp_string) / "image.jpg"
            image.write_bytes(b"fixture")
            collator = AssistantOnlyVisionCollator(
                FakeProcessor(), max_length=16, vision_info_fn=fake_vision_info
            )

            batch = collator([fixture_record(image)])

            labels = batch["labels"][0]
            self.assertTrue(torch.all(labels[: collator.last_prefix_length] == -100))
            self.assertTrue(torch.any(labels[collator.last_prefix_length :] != -100))
            self.assertTrue(torch.all(labels[batch["attention_mask"][0] == 0] == -100))
            self.assertTrue(torch.all(labels[batch["input_ids"][0] == 99] == -100))

    def test_collator_rejects_empty_assistant_loss_mask(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            image = Path(tmp_string) / "image.jpg"
            image.write_bytes(b"fixture")
            collator = AssistantOnlyVisionCollator(
                FakeProcessor(full_ids=[10, 11, 99, 0]),
                max_length=16,
                vision_info_fn=fake_vision_info,
            )

            with self.assertRaisesRegex(ValueError, "empty assistant loss mask"):
                collator([fixture_record(image)])

    def test_preflight_rejects_overlength_sample(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            image = tmp / "image.jpg"
            image.write_bytes(b"fixture")
            record = fixture_record(image)
            path = tmp / "data.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = JsonlDataset(path)
            collator = AssistantOnlyVisionCollator(
                FakeProcessor(full_ids=[10, 11, 99, 20, 21, 22]),
                max_length=5,
                vision_info_fn=fake_vision_info,
            )

            with self.assertRaisesRegex(ValueError, "max_length"):
                preflight_dataset(dataset, collator, max_length=5)

    def test_preflight_reports_deterministic_progress(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            image = tmp / "image.jpg"
            image.write_bytes(b"fixture")
            path = tmp / "data.jsonl"
            path.write_text(json.dumps(fixture_record(image)) + "\n", encoding="utf-8")
            dataset = JsonlDataset(path)
            collator = AssistantOnlyVisionCollator(
                FakeProcessor(), max_length=16, vision_info_fn=fake_vision_info
            )
            progress = []

            summary = preflight_dataset(
                dataset,
                collator,
                max_length=16,
                progress_every=1,
                progress_callback=lambda done, total, record_id: progress.append(
                    (done, total, record_id)
                ),
            )

            self.assertEqual(summary["samples"], 1)
            self.assertEqual(progress, [(1, 1, "fixture-positive")])


if __name__ == "__main__":
    unittest.main()
