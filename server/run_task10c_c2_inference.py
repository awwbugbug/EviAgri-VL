"""Run frozen Task 10C C2 generation for Base or one adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from run_task10c_smoke_inference import (
    CONDITIONS,
    generation_contract,
    inference_messages,
)
from task10c_c2_contract import C2_STEPS, validate_checkpoint_summary, verify_c2_protocol
from task10c_contract import (
    CLASS_IDS,
    SYSTEM_PROMPT,
    TRAIN_PROMPT,
    UNSEEN_PROMPT,
    strict_parse_pest_json,
)
from train_task10c_c2 import _sha256, _write_json
from train_task10c_smoke import SEEDS, _verify_completion


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


def _write_completion(output: Path, names: list[str]) -> None:
    (output / "completion.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def build_four_condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (int(item["class_id"]), str(item["source_image_sha256"]))):
        messages = row.get("model", {}).get("messages", [])
        if len(messages) < 2 or messages[0] != {
            "role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        }:
            raise ValueError("C2 inference system prompt mismatch")
        user_content = messages[1].get("content", [])
        if [item.get("type") for item in user_content] != ["image", "text"]:
            raise ValueError("C2 inference source message mismatch")
        image = user_content[0].get("image")
        if not isinstance(image, str) or not image:
            raise ValueError("C2 inference source image is missing")
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


def build_c2_conditions(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    try:
        expected_per_class = {"smoke_dev": 1, "dev": 5}[split]
    except KeyError as exc:
        raise ValueError(f"invalid C2 inference split: {split}") from exc
    counts = Counter(int(row["class_id"]) for row in rows)
    if counts != Counter({class_id: expected_per_class for class_id in CLASS_IDS}):
        raise ValueError(f"C2 {split} class quota mismatch")
    expected_rows = len(CLASS_IDS) * expected_per_class
    if len(rows) != expected_rows or len({str(row["id"]) for row in rows}) != expected_rows:
        raise ValueError(f"C2 {split} row identity mismatch")
    return build_four_condition_rows(rows)


def model_identity(
    *,
    model_kind: str,
    seed: int | None = None,
    checkpoint_step: int | None = None,
) -> dict[str, Any]:
    if model_kind == "base":
        if seed is not None or checkpoint_step is not None:
            raise ValueError("C2 Base identity cannot include seed or checkpoint")
        return {
            "model_id": "D0_base", "model_kind": "base",
            "seed": None, "checkpoint_step": 0,
        }
    if model_kind != "adapter" or seed not in SEEDS or checkpoint_step not in C2_STEPS:
        raise ValueError("C2 adapter identity requires frozen seed and checkpoint step")
    return {
        "model_id": f"D1_seed_{seed}_step_{checkpoint_step:03d}",
        "model_kind": "adapter",
        "seed": seed,
        "checkpoint_step": checkpoint_step,
    }


def verify_c2_adapter(
    adapter_root: str | Path,
    *,
    seed: int,
    step: int,
) -> dict[str, Any]:
    root = Path(adapter_root)
    _verify_completion(root)
    summary = json.loads((root / "checkpoint_summary.json").read_text(encoding="utf-8"))
    validate_checkpoint_summary(summary, seed=seed, step=step)
    adapter = root / "adapter" / "adapter_model.safetensors"
    if _sha256(adapter) != summary["adapter"]["sha256"]:
        raise ValueError("C2 adapter SHA mismatch before reload")
    return summary


def verify_prediction_ids(
    manifest: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
) -> None:
    expected = [str(row["id"]) for row in manifest]
    actual = [str(row["id"]) for row in predictions]
    if len(expected) != len(set(expected)) or len(actual) != len(set(actual)):
        raise ValueError("duplicate IDs in C2 inference")
    if set(expected) != set(actual):
        raise ValueError(f"C2 prediction ID mismatch: expected={len(expected)} actual={len(actual)}")


def run_c2_inference(
    *,
    protocol_root: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    split: str,
    model_kind: str,
    seed: int | None = None,
    checkpoint_step: int | None = None,
    adapter_root: str | Path | None = None,
) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    protocol_root, model_path, output = Path(protocol_root), Path(model_path), Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C2 inference: {output}")
    gate = verify_c2_protocol(protocol_root)
    config = json.loads((protocol_root / "config.snapshot.json").read_text(encoding="utf-8"))
    if Path(config["model_path"]).resolve() != model_path.resolve():
        raise ValueError("C2 inference model path differs from signed protocol")
    identity = model_identity(
        model_kind=model_kind, seed=seed, checkpoint_step=checkpoint_step,
    )
    checkpoint = None
    adapter_path = None
    if model_kind == "adapter":
        if adapter_root is None:
            raise ValueError("C2 adapter inference requires adapter root")
        adapter_path = Path(adapter_root)
        checkpoint = verify_c2_adapter(
            adapter_path, seed=int(seed), step=int(checkpoint_step),
        )
        if checkpoint.get("protocol_manifest_sha256") not in (None, gate["manifest_sha256"]):
            raise ValueError("C2 adapter protocol SHA mismatch")
    elif adapter_root is not None:
        raise ValueError("C2 Base inference cannot include adapter root")
    source_name = "smoke_dev.jsonl" if split == "smoke_dev" else "dev.jsonl"
    rows = build_c2_conditions(_read_jsonl(protocol_root / source_name), split=split)
    expected = 64 if split == "smoke_dev" else 320
    if len(rows) != expected:
        raise ValueError("C2 inference matrix size mismatch")
    output.mkdir(parents=True)
    _write_json(output / "status.json", {
        "state": "running", "stage": "inference", "completed": 0,
        "expected": expected, **identity,
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
        model = base if model_kind == "base" else PeftModel.from_pretrained(
            base, str(adapter_path / "adapter"), is_trainable=False,
        )
        model.eval()
        device = next(model.parameters()).device
        predictions: list[dict[str, Any]] = []
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
            completion = generated[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(
                completion,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            predictions.append({
                "id": row["id"],
                **identity,
                "split": split,
                "condition": row["condition"],
                "class_id": row["class_id"],
                "class_band": row["class_band"],
                "source_image_id": row["source_image_id"],
                "source_image_sha256": row["source_image_sha256"],
                "near_duplicate_component_id": row["near_duplicate_component_id"],
                "raw_text": raw,
                "parsed": strict_parse_pest_json(raw),
                "latency_seconds": time.time() - row_started,
            })
            if (index + 1) % 16 == 0:
                _write_json(output / "status.json", {
                    "state": "running", "stage": "inference",
                    "completed": index + 1, "expected": expected, **identity,
                })
        verify_prediction_ids(rows, predictions)
        prediction_path = output / "predictions.jsonl"
        _write_jsonl(prediction_path, predictions)
        summary = {
            "version": "task10c-c2-inference-summary-v1",
            "state": "completed",
            **identity,
            "split": split,
            "prediction_count": len(predictions),
            "condition_counts": dict(sorted(Counter(row["condition"] for row in rows).items())),
            "adapter_sha256": None if checkpoint is None else checkpoint["adapter"]["sha256"],
            "adapter_reload_verified": model_kind == "adapter",
            "protocol_manifest_sha256": gate["manifest_sha256"],
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
            "state": "completed", "stage": "inference",
            "completed": len(predictions), "expected": expected, **identity,
        })
        _write_completion(output, ["predictions.jsonl", "run_summary.json", "status.json"])
        return summary
    except Exception as exc:
        _write_json(output / "failure.json", {
            "state": "failed", "stage": "inference", **identity,
            "error": str(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        })
        _write_json(output / "status.json", {"state": "failed", "stage": "inference", **identity})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 10C C2 unified inference")
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke_dev", "dev"), required=True)
    parser.add_argument("--model-kind", choices=("base", "adapter"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--checkpoint-step", type=int, choices=C2_STEPS)
    parser.add_argument("--adapter-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_c2_inference(
        protocol_root=args.protocol_root,
        model_path=args.model_path,
        output_root=args.output_root,
        split=args.split,
        model_kind=args.model_kind,
        seed=args.seed,
        checkpoint_step=args.checkpoint_step,
        adapter_root=args.adapter_root,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
