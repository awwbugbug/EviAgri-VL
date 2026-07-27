"""Extract exact frozen Qwen query-token representations for Task11B.1."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from extract_task10b_features import MAX_PIXELS, MIN_PIXELS, _model_identity, _write_feature_payload, _write_json_replace, assert_frozen
from task10_audit_common import ensure_new_directory, write_json_new
from task11a_confidence_router import build_stress_rows, transform_image
from task11b0_query_rep_smoke import layer_output_index, mean_pool_l2, unique_subsequence


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_query_features(*, manifest: Path, model_path: Path, output_root: Path,
                           query: str, expected_layer: int, expected_query_tokens: int,
                           mode: str) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    status = destination / "status.json"
    _write_json_replace(status, {"state": "running", "stage": "verify"})
    started = time.monotonic()
    try:
        from PIL import Image
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        source_rows = read_rows(Path(manifest))
        rows = build_stress_rows(source_rows) if mode == "stress" else source_rows
        if mode not in {"plain", "stress"} or not rows:
            raise ValueError("invalid or empty Task11B.1 extraction mode")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True).get_text_config()
        layers, hidden_size = int(config.num_hidden_layers), int(config.hidden_size)
        target = layer_output_index(layers)
        if target != expected_layer or hidden_size != 2048:
            raise ValueError("Task11B.1 architecture/layer drift")
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS, use_fast=False, local_files_only=True)
        query_ids = processor.tokenizer(query, add_special_tokens=False).input_ids
        if len(query_ids) != expected_query_tokens:
            raise ValueError("Task11B.1 query token drift")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
            local_files_only=True)
        model.eval(); model.requires_grad_(False); assert_frozen(model)
        device = next(model.parameters()).device
        torch.cuda.reset_peak_memory_stats(device)
        vectors, feature_rows = [], []
        _write_json_replace(status, {"state": "running", "stage": "extract", "expected": len(rows), "completed": 0})
        for index, row in enumerate(rows):
            image_path = Path(str(row.get("image", "")))
            if not image_path.is_file():
                raise ValueError(f"missing image: {image_path}")
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
            if mode == "stress":
                image = transform_image(image, condition=str(row["condition"]), split=str(row["split"]),
                    row_id=str(row["id"]), seed=int(row["stress_seed"]))
            inputs = processor(text=[rendered], images=[image], return_tensors="pt")
            start, end = unique_subsequence(inputs["input_ids"][0].tolist(), query_ids)
            moved = {key: (value.to(device=device, dtype=torch.bfloat16) if key == "pixel_values" else value.to(device))
                     if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**moved, output_hidden_states=True, use_cache=False, return_dict=True)
            states = outputs.hidden_states
            if states is None or len(states) != layers + 1:
                raise ValueError("unexpected hidden-state contract")
            vector = mean_pool_l2(states[target][0, start:end, :]).cpu().numpy().astype(np.float32, copy=False)
            vectors.append(vector)
            feature_rows.append({**row, "feature_index": index, "query_start": start, "query_end": end,
                "representation": "qwen_text_hidden_states_27_query_mean_l2"})
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                _write_json_replace(status, {"state": "running", "stage": "extract", "expected": len(rows), "completed": index + 1})
        matrix = np.stack(vectors).astype(np.float32, copy=False)
        summary = _write_feature_payload(matrix=matrix, feature_rows=feature_rows,
            output_root=destination, manifest_path=Path(manifest),
            config={"version": "task11b1-query-feature-config-v1", "mode": mode,
                "query": query, "query_tokens": len(query_ids), "layer_output_index": target,
                "pooling": "query_token_mean_then_l2", "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            model_identity=_model_identity(Path(model_path)),
            run_metadata={"elapsed_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "all_parameters_frozen": True, "optimizer_created": False,
                "task8_locked_set_read": False},
            summary_version="task11b1-query-feature-summary-v1")
        _write_json_replace(status, {"state": "completed", "stage": "done"})
        return summary
    except Exception as exc:
        write_json_new(destination / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        _write_json_replace(status, {"state": "failed", "stage": "extract"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--expected-layer", type=int, required=True)
    parser.add_argument("--expected-query-tokens", type=int, required=True)
    parser.add_argument("--mode", choices=("plain", "stress"), required=True)
    args = parser.parse_args()
    print(json.dumps(extract_query_features(manifest=args.manifest, model_path=args.model_path,
        output_root=args.output_root, query=args.query, expected_layer=args.expected_layer,
        expected_query_tokens=args.expected_query_tokens, mode=args.mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
