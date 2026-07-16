"""Shared Base/adapter inference runner for frozen Task 9D evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


def generation_contract() -> dict[str, Any]:
    return {
        "do_sample": False,
        "max_new_tokens": 512,
        "min_pixels": 200704,
        "max_pixels": 401408,
        "parser_version": "task9d-json-parser-v1",
    }


def inference_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"invalid inference messages for {row.get('id')}")
    prompt = messages[:2]
    if [item.get("role") for item in prompt] != ["system", "user"]:
        raise ValueError("Task 9D inference requires system/user prompt")
    content = prompt[1].get("content")
    if not isinstance(content, list) or [item.get("type") for item in content] != ["image", "text"]:
        raise ValueError("Task 9D inference must contain pixels then neutral text")
    return prompt


def verify_expected_ids(manifest: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]) -> None:
    expected = [str(row["id"]) for row in manifest]
    actual = [str(row["id"]) for row in predictions]
    if len(expected) != len(set(expected)) or len(actual) != len(set(actual)):
        raise ValueError("duplicate IDs in Task 9D manifest or predictions")
    if set(expected) != set(actual):
        raise ValueError(f"prediction ID mismatch: expected={len(expected)} actual={len(actual)}")


def ensure_prediction_output_new(path: str | Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite Task 9D inference output: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_inference(
    model_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    group: str,
    adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    ensure_prediction_output_new(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        rows = _read_jsonl(manifest_path)
        if not rows:
            raise ValueError("Task 9D inference manifest is empty")
        contract = generation_contract()
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=contract["min_pixels"], max_pixels=contract["max_pixels"],
            use_fast=False, local_files_only=True,
        )
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()
        predictions = []
        for index, row in enumerate(rows):
            row_started = time.time()
            prompt_messages = inference_messages(row)
            prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(prompt_messages)
            inputs = processor(
                text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=contract["max_new_tokens"]
                )
            completion = generated[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            predictions.append({
                "id": str(row["id"]), "group": group, "raw_text": raw,
                "latency_seconds": time.time() - row_started, "error": None,
            })
            if (index + 1) % 32 == 0:
                _write_json(output_dir / "status.json", {
                    "state": "running", "group": group, "completed": index + 1, "expected": len(rows)
                })
        verify_expected_ids(rows, predictions)
        predictions_path = output_dir / "predictions.jsonl"
        _write_jsonl(predictions_path, predictions)
        summary = {
            "version": "task9d-inference-summary-v1", "state": "completed", "group": group,
            "adapter_path": None if adapter_path is None else str(adapter_path),
            "prediction_count": len(predictions), "expected_count": len(rows),
            "elapsed_seconds": time.time() - started, "contract": contract,
            "manifest_sha256": _sha256(manifest_path),
            "predictions_sha256": _sha256(predictions_path),
        }
        _write_json(output_dir / "run_summary.json", summary)
        completion_path = output_dir / "completion.sha256"
        completion_path.write_text(
            f"{_sha256(predictions_path)}  predictions.jsonl\n{_sha256(output_dir / 'run_summary.json')}  run_summary.json\n",
            encoding="utf-8",
        )
        _write_json(output_dir / "status.json", {"state": "completed", "group": group, "completed": len(rows)})
        return summary
    except Exception as exc:
        _write_json(output_dir / "failure.json", {"state": "failed", "group": group, "error": str(exc),
                                                    "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_inference(args.model_path, args.manifest, args.output, group=args.group,
                                   adapter_path=args.adapter), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
