import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

from static_qlora_config import load_training_config
from static_qlora_data import EXPECTED_TARGET_KEYS, JsonlDataset
from static_qlora_model import LANGUAGE_ATTN
from train_static_qlora import deterministic_smoke_subset


EXPECTED_VERSIONS = {
    "torch": "2.5.1+cu121",
    "transformers": "4.51.3",
    "peft": "0.15.2",
    "bitsandbytes": "0.45.5",
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing smoke artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_structured_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict) or tuple(value) != EXPECTED_TARGET_KEYS:
        raise ValueError("invalid Evidence-First schema or key order")
    return value


def _language_only_lora(trainable: Any, targets: Any) -> bool:
    if not isinstance(trainable, list) or not trainable or not isinstance(targets, list) or not targets:
        return False
    trainables_safe = all(
        isinstance(name, str)
        and "lora_" in name
        and "visual" not in name
        and "merger" not in name
        and "projector" not in name
        for name in trainable
    )
    targets_safe = all(isinstance(name, str) and LANGUAGE_ATTN.fullmatch(name) for name in targets)
    return trainables_safe and targets_safe


def validate_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    environment = _read_json(output_dir / "environment.json")
    summary = _read_json(output_dir / "run_summary.json")
    trainable = _read_json(output_dir / "trainable_parameters.json")
    targets = _read_json(output_dir / "lora_targets.json")
    reload_report = _read_json(output_dir / "reload_and_generation.json")

    versions_ok = all(environment.get(name) == version for name, version in EXPECTED_VERSIONS.items())
    losses = summary.get("losses", [])
    finite_loss = (
        isinstance(losses, list)
        and len(losses) == 2
        and all(isinstance(loss, (int, float)) and math.isfinite(loss) for loss in losses)
    )
    peak_bytes = summary.get("peak_vram_reserved_bytes")
    peak_ok = isinstance(peak_bytes, int) and peak_bytes < 30 * 1024**3
    generations = reload_report.get("generations", {})
    generation_ok = False
    try:
        parse_structured_json(generations["positive"])
        parse_structured_json(generations["null"])
        generation_ok = True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        generation_ok = False

    gates = {
        "dependencies_and_4bit_load": bool(
            versions_ok
            and environment.get("cuda_available") is True
            and reload_report.get("base_loaded_in_4bit") is True
        ),
        "language_only_lora": _language_only_lora(trainable, targets),
        "finite_loss": finite_loss,
        "peak_vram_below_30gb": peak_ok,
        "adapter_reload": reload_report.get("adapter_reload") is True,
        "positive_and_null_json_generation": generation_ok,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "observed": {
            "losses": losses,
            "peak_vram_reserved_bytes": peak_bytes,
            "peak_vram_reserved_gb": peak_bytes / 1024**3 if isinstance(peak_bytes, int) else None,
            "versions": {name: environment.get(name) for name in EXPECTED_VERSIONS},
        },
    }


def _generate_one(model, processor, record: dict[str, Any]) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = record["messages"][:1]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    continuation = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def run_reload_and_generation(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    output_dir = Path(output_dir)
    report_path = output_dir / "reload_and_generation.json"
    if report_path.exists():
        raise ValueError(f"refusing to overwrite smoke reload report: {report_path}")
    report: dict[str, Any] = {
        "base_loaded_in_4bit": False,
        "adapter_reload": False,
        "generations": {},
    }
    try:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["quantization"]["type"],
            bnb_4bit_use_double_quant=config["quantization"]["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model_path"],
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        report["base_loaded_in_4bit"] = bool(getattr(base, "is_loaded_in_4bit", False))
        model = PeftModel.from_pretrained(base, output_dir / "adapter", is_trainable=False)
        model.eval()
        report["adapter_reload"] = True
        processor = AutoProcessor.from_pretrained(
            config["model_path"],
            min_pixels=config["vision"]["min_pixels"],
            max_pixels=config["vision"]["max_pixels"],
            use_fast=False,
            local_files_only=True,
        )
        dataset = JsonlDataset(Path(config["mixed_data_root"]) / "train.jsonl")
        subset = deterministic_smoke_subset(
            dataset.records,
            positive=config["smoke"]["positive"],
            null=config["smoke"]["null"],
            seed=config["seed"],
        )
        positive = next(row for row in subset if row["task_type"] == "pest_evidence_grounding")
        null = next(row for row in subset if row["task_type"] == "prompt_conflict_null_evidence")
        report["sample_ids"] = {"positive": positive["id"], "null": null["id"]}
        report["generations"] = {
            "positive": _generate_one(model, processor, positive),
            "null": _generate_one(model, processor, null),
        }
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    _write_json_atomic(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all six Static QLoRA v1 smoke gates")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    if not (args.output_dir / "reload_and_generation.json").exists():
        run_reload_and_generation(config, args.output_dir)
    report = validate_smoke(args.output_dir)
    _write_json_atomic(args.output_dir / "smoke_gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
