import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9d_config import load_task9d_config
from task9d_data import AssistantOnlyV2Collator, validate_v2_record
from task9d_model import language_qv_targets, reject_unsafe_trainables


def _target(present=True):
    return {
        "evidence_present": present,
        "evidence_region": [1, 2, 3, 4] if present else None,
        "visible_attributes": [],
        "diagnosis": {
            "status": "supported" if present else "abstain",
            "pest_id": 1 if present else None,
            "pest_name": "pest" if present else None,
            "species": None,
            "stage": None,
        },
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }


def _record(target=None):
    target = target or _target()
    return {"id": "r1", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [
            {"type": "image", "image": "images/opaque.png"},
            {"type": "text", "text": "neutral query"},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": json.dumps(target, separators=(",", ":"))}
        ]},
    ]}


def test_v2_schema_and_null_semantics_are_strict():
    validate_v2_record(_record())
    validate_v2_record(_record(_target(False)))
    invalid = _target(False)
    invalid["diagnosis"]["species"] = "leaked species"
    with pytest.raises(ValueError, match="null diagnosis"):
        validate_v2_record(_record(invalid))
    reordered = _target()
    reordered = {"diagnosis": reordered["diagnosis"], **{k: v for k, v in reordered.items() if k != "diagnosis"}}
    with pytest.raises(ValueError, match="key order"):
        validate_v2_record(_record(reordered))


class _Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 99}.get(token, -1)


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "prefix" if add_generation_prompt else "full"

    def __call__(self, text, **kwargs):
        values = [1, 99, 2, 3, 4] if text == ["prefix"] else [1, 99, 2, 3, 4, 7, 8, 9]
        return {"input_ids": torch.tensor([values]), "attention_mask": torch.ones((1, len(values)), dtype=torch.long)}


def test_collator_masks_system_user_image_and_padding_but_keeps_assistant():
    collator = AssistantOnlyV2Collator(_Processor(), max_length=8, vision_info_fn=lambda _: ([], []))
    result = collator([_record()])
    assert result["labels"].tolist() == [[-100, -100, -100, -100, -100, 7, 8, 9]]
    with pytest.raises(ValueError, match="exceeds max_length"):
        AssistantOnlyV2Collator(_Processor(), max_length=7, vision_info_fn=lambda _: ([], []))([_record()])


class _FakeModel:
    def named_modules(self):
        return [(name, object()) for name in (
            "visual.blocks.0.attn.q_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.self_attn.v_proj",
        )]


def test_only_language_qv_targets_and_lora_parameters_are_allowed():
    assert language_qv_targets(_FakeModel()) == [
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj",
        "model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.v_proj",
    ]
    reject_unsafe_trainables(["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"])
    with pytest.raises(ValueError, match="unsafe trainable"):
        reject_unsafe_trainables(["base_model.model.visual.blocks.0.attn.q_proj.lora_A.default.weight"])


def _config():
    return {
        "version": "task9d-static-qlora-v1", "seeds": [17, 29, 43],
        "variants": ["A", "B", "C"], "model_path": "/models/Qwen2___5-VL-3B-Instruct",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
        "quantization": {"type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
        "vision": {"min_pixels": 200704, "max_pixels": 401408},
        "training": {"max_length": 1024, "batch_size": 1, "gradient_accumulation_steps": 8,
                     "max_steps": 192, "learning_rate": 0.0001, "early_stopping": False,
                     "eval_steps": [64, 128, 192]},
        "decoding": {"do_sample": False, "max_new_tokens": 512},
    }


def test_config_freezes_all_control_variables(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert load_task9d_config(path)["seeds"] == [17, 29, 43]
    for mutation in (
        lambda c: c["training"].update(max_steps=191),
        lambda c: c["lora"].update(targets=["q_proj", "k_proj", "v_proj"]),
        lambda c: c.update(seeds=[17, 29]),
        lambda c: c["training"].update(early_stopping=True),
    ):
        config = _config()
        mutation(config)
        path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match="frozen"):
            load_task9d_config(path)
