"""Qwen2.5-VL q/v-only Static QLoRA construction for Task 9D."""

from __future__ import annotations

import re
from typing import Any, Iterable


LANGUAGE_QV = re.compile(r"^model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$")


def language_qv_targets(model) -> list[str]:
    targets = sorted(name for name, _ in model.named_modules() if LANGUAGE_QV.fullmatch(name))
    if not targets:
        raise RuntimeError("empty q/v-only language LoRA target set")
    if any(token in name for name in targets for token in ("visual", "merger", "projector")):
        raise RuntimeError("unsafe multimodal module in q/v target set")
    return targets


def reject_unsafe_trainables(names: Iterable[str]) -> list[str]:
    names = sorted(map(str, names))
    if not names:
        raise ValueError("model has no trainable LoRA parameters")
    unsafe = [name for name in names if "lora_" not in name or any(
        token in name for token in ("visual", "merger", "projector", "k_proj", "o_proj")
    )]
    if unsafe:
        raise ValueError(f"unsafe trainable parameters: {unsafe[:10]}")
    return names


def build_task9d_model(config: dict[str, Any]):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_path"], quantization_config=quantization, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    targets = language_qv_targets(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=targets, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = reject_unsafe_trainables(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    return model, trainable, targets
