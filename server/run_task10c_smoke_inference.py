"""Reload one Task 10C smoke adapter and run the frozen 16x4 inference matrix."""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from task10c_contract import (
    CLASS_IDS,
    PARSER_VERSION,
    SYSTEM_PROMPT,
    TRAIN_PROMPT,
    UNSEEN_PROMPT,
    strict_parse_pest_json,
)
from train_task10c_smoke import SEEDS, _sha256, _verify_completion, _write_json


CONDITIONS = (
    "image_train_prompt",
    "image_unseen_prompt",
    "no_image_train_prompt",
    "no_image_unseen_prompt",
)


def generation_contract() -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 32,
        "min_pixels": 200704,
        "max_pixels": 401408,
        "parser_version": PARSER_VERSION,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def build_smoke_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 16 or {int(row["class_id"]) for row in rows} != set(CLASS_IDS):
        raise ValueError("Task 10C smoke dev must contain one row for each frozen class")
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (int(item["class_id"]), str(item["source_image_sha256"]))):
        model_messages = row["model"]["messages"]
        if model_messages[0] != {
            "role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        }:
            raise ValueError("Task 10C smoke system prompt mismatch")
        image = model_messages[1]["content"][0]["image"]
        for condition in CONDITIONS:
            image_present = condition.startswith("image_")
            prompt = TRAIN_PROMPT if condition.endswith("train_prompt") else UNSEEN_PROMPT
            content: list[dict[str, Any]] = []
            if image_present:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            output.append({
                "id": f"{row['id']}::{condition}",
                "condition": condition,
                "prompt_variant": "train" if prompt == TRAIN_PROMPT else "unseen",
                "image_present": image_present,
                "class_id": int(row["class_id"]),
                "class_band": str(row["class_band"]),
                "source_image_id": str(row["source_image_id"]),
                "source_image_sha256": str(row["source_image_sha256"]),
                "near_duplicate_component_id": str(row["near_duplicate_component_id"]),
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content},
                ],
            })
    return output


def inference_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != ["system", "user"]:
        raise ValueError("Task 10C inference requires exact system/user messages")
    if messages[0] != {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}:
        raise ValueError("Task 10C inference system prompt mismatch")
    content = messages[1].get("content")
    types = [item.get("type") for item in content] if isinstance(content, list) else []
    if types not in (["image", "text"], ["text"]):
        raise ValueError("Task 10C inference condition is invalid")
    text = content[-1].get("text")
    if text not in {TRAIN_PROMPT, UNSEEN_PROMPT}:
        raise ValueError("Task 10C inference prompt mismatch")
    return messages


def verify_prediction_ids(
    manifest: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
) -> None:
    expected = [str(row["id"]) for row in manifest]
    actual = [str(row["id"]) for row in predictions]
    if len(expected) != len(set(expected)) or len(actual) != len(set(actual)):
        raise ValueError("duplicate IDs in Task 10C inference")
    if set(expected) != set(actual):
        raise ValueError(f"prediction ID mismatch: expected={len(expected)} actual={len(actual)}")


def _write_completion(output: Path, names: list[str]) -> None:
    (output / "completion.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def run_smoke_inference(
    *,
    protocol_root: str | Path,
    model_path: str | Path,
    adapter_root: str | Path,
    output_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if seed not in SEEDS:
        raise ValueError(f"Task 10C inference requires frozen seed: {seed}")
    protocol_root, model_path = Path(protocol_root), Path(model_path)
    adapter_root, output = Path(adapter_root), Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Task 10C inference: {output}")
    _verify_completion(protocol_root)
    _verify_completion(adapter_root)
    training = json.loads((adapter_root / "run_summary.json").read_text(encoding="utf-8"))
    if training.get("seed") != seed or training.get("optimizer_steps") != 8:
        raise ValueError("Task 10C inference training gate mismatch")
    adapter_report = json.loads((adapter_root / "adapter.sha256.json").read_text(encoding="utf-8"))
    adapter_file = adapter_root / "adapter" / "adapter_model.safetensors"
    if _sha256(adapter_file) != adapter_report.get("sha256"):
        raise ValueError("Task 10C adapter SHA mismatch before reload")
    rows = build_smoke_conditions(_read_jsonl(protocol_root / "smoke_dev.jsonl"))
    output.mkdir(parents=True)
    _write_json(output / "status.json", {
        "state": "running", "seed": seed, "stage": "inference", "completed": 0, "expected": 64,
    })
    started = time.time()
    try:
        contract = generation_contract()
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=contract["min_pixels"],
            max_pixels=contract["max_pixels"],
            use_fast=False,
            local_files_only=True,
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, str(adapter_root / "adapter"), is_trainable=False)
        model.eval()
        device = next(model.parameters()).device
        predictions = []
        for index, row in enumerate(rows):
            row_started = time.time()
            messages = inference_messages(row)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            processor_values: dict[str, Any] = {
                "text": [prompt], "padding": True, "return_tensors": "pt",
            }
            if images:
                processor_values["images"] = images
            if videos:
                processor_values["videos"] = videos
            inputs = processor(**processor_values).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=contract["do_sample"],
                    num_beams=contract["num_beams"],
                    max_new_tokens=contract["max_new_tokens"],
                )
            completion = generated[:, inputs.input_ids.shape[1] :]
            raw = processor.batch_decode(
                completion,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed = strict_parse_pest_json(raw)
            predictions.append({
                "id": row["id"],
                "seed": seed,
                "condition": row["condition"],
                "class_id": row["class_id"],
                "class_band": row["class_band"],
                "source_image_id": row["source_image_id"],
                "source_image_sha256": row["source_image_sha256"],
                "near_duplicate_component_id": row["near_duplicate_component_id"],
                "raw_text": raw,
                "parsed": parsed,
                "latency_seconds": time.time() - row_started,
            })
            if (index + 1) % 16 == 0:
                _write_json(output / "status.json", {
                    "state": "running", "seed": seed, "stage": "inference",
                    "completed": index + 1, "expected": len(rows),
                })
        verify_prediction_ids(rows, predictions)
        prediction_path = output / "predictions.jsonl"
        _write_jsonl(prediction_path, predictions)
        summary = {
            "version": "task10c-smoke-inference-summary-v1",
            "state": "completed",
            "seed": seed,
            "prediction_count": len(predictions),
            "condition_counts": dict(sorted(Counter(row["condition"] for row in rows).items())),
            "adapter_sha256": adapter_report["sha256"],
            "adapter_reload_verified": True,
            "protocol_manifest_sha256": json.loads(
                (protocol_root / "preflight_report.json").read_text(encoding="utf-8")
            )["manifest_sha256"],
            "predictions_sha256": _sha256(prediction_path),
            "elapsed_seconds": time.time() - started,
            "contract": contract,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            },
        }
        _write_json(output / "run_summary.json", summary)
        _write_json(output / "status.json", {
            "state": "completed", "seed": seed, "stage": "inference",
            "completed": len(predictions), "expected": len(rows),
        })
        _write_completion(output, ["predictions.jsonl", "run_summary.json", "status.json"])
        return summary
    except Exception as exc:
        _write_json(output / "failure.json", {
            "state": "failed", "seed": seed, "stage": "inference",
            "error": str(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        })
        _write_json(output / "status.json", {"state": "failed", "seed": seed, "stage": "inference"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 10C smoke inference")
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke_inference(
        protocol_root=args.protocol_root,
        model_path=args.model_path,
        adapter_root=args.adapter_root,
        output_root=args.output_root,
        seed=args.seed,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
