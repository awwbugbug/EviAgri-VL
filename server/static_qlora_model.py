import re
from typing import Any


LANGUAGE_ATTN = re.compile(
    r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)


def language_attention_targets(model) -> list[str]:
    targets = sorted(
        name for name, _module in model.named_modules() if LANGUAGE_ATTN.fullmatch(name)
    )
    if not targets:
        raise RuntimeError("empty LoRA target set: Qwen language self-attention modules not found")
    if any("visual" in name or "merger" in name or "projector" in name for name in targets):
        raise RuntimeError("unsafe LoRA target set contains a multimodal module")
    return targets


def safe_trainable_parameter_names(model) -> list[str]:
    names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not names:
        raise RuntimeError("model has no trainable LoRA parameters")
    unsafe = [
        name
        for name in names
        if "lora_" not in name
        or "visual" in name
        or "merger" in name
        or "projector" in name
    ]
    if unsafe:
        raise RuntimeError(f"unsafe trainable parameters: {unsafe[:10]}")
    return names


def build_qlora_model(config: dict[str, Any]):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["type"],
        bnb_4bit_use_double_quant=config["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_path"],
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    targets = language_attention_targets(model)
    lora = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = safe_trainable_parameter_names(model)
    return model, trainable, targets
