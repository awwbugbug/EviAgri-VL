"""Extract and validate a frozen Qwen query-token representation smoke set."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from extract_task10b_features import MAX_PIXELS, MIN_PIXELS, _model_identity, assert_frozen
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


BASE_ROWS_SHA = "2ad5192520a2fdbf1b1f058cfd987d6ad121985f62239e41234ae0d2d2a25ffd"
PLANTSEG_MANIFEST_SHA = "7196eb45259b851908c362dfbbb08e1b5b81f65f36dfe1a25d36692e63efc025"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    if not subsequence or len(subsequence) > len(sequence):
        raise ValueError("invalid query token subsequence")
    hits = [i for i in range(len(sequence) - len(subsequence) + 1)
            if sequence[i:i + len(subsequence)] == subsequence]
    if len(hits) != 1:
        raise ValueError(f"query token subsequence must occur exactly once, got {hits}")
    return hits[0], hits[0] + len(subsequence)


def layer_output_index(num_hidden_layers: int) -> int:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    return math.ceil(3 * num_hidden_layers / 4)


def mean_pool_l2(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("query hidden states must be a non-empty matrix")
    vector = values.detach().to(torch.float32).mean(dim=0)
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(vector).all() or not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("invalid query representation")
    return vector / norm


def decide_smoke(*, layers: int, hidden_size: int, layer_index: int, query_tokens: int,
                 duplicate_max_abs: float, cosine_distances: list[float], l2_distances: list[float],
                 gates: dict[str, Any]) -> dict[str, Any]:
    conditions = {
        "layers_match": layers == int(gates["expected_layers"]),
        "hidden_size_match": hidden_size == int(gates["expected_hidden_size"]),
        "layer_output_index_match": layer_index == int(gates["expected_layer_output_index"]),
        "query_tokens_match": query_tokens == int(gates["expected_query_tokens"]),
        "duplicate_reproducible": duplicate_max_abs <= float(gates["duplicate_max_abs_le"]),
        "median_image_dependence": float(np.median(cosine_distances)) >= float(gates["median_original_blank_cosine_distance_ge"]),
        "all_pairs_image_dependent": min(l2_distances) >= float(gates["minimum_original_blank_l2_ge"]),
    }
    return {"conditions": conditions, "passed": all(conditions.values()),
            "decision": "PASS" if all(conditions.values()) else "FAIL"}


def run_smoke(*, config_path: Path, base_rows_path: Path, plantseg_manifest_path: Path,
              model_path: Path, output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    ensure_new_directory(output_root)
    status_path = output_root / "status.json"
    status_path.write_text('{"state":"running","stage":"verify"}\n', encoding="utf-8")
    started = time.monotonic()
    try:
        from PIL import Image
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if sha256_file(Path(base_rows_path)) != BASE_ROWS_SHA or sha256_file(Path(plantseg_manifest_path)) != PLANTSEG_MANIFEST_SHA:
            raise ValueError("frozen manifest SHA mismatch")
        base_rows, null_rows = read_jsonl(Path(base_rows_path)), read_jsonl(Path(plantseg_manifest_path))
        base_by_id, null_by_id = {str(r["id"]): r for r in base_rows}, {str(r["id"]): r for r in null_rows}
        selected = []
        for sample in config["samples"]:
            source = base_by_id if sample["kind"] == "ip102_positive" else null_by_id
            row = source.get(str(sample["id"]))
            if row is None or str(row.get("image_sha256", row.get("source_image_sha256"))) != str(sample["image_sha256"]):
                raise ValueError(f"selected sample mismatch: {sample['id']}")
            image_path = Path(str(row["image"]))
            if not image_path.is_file() or sha256_file(image_path) != str(sample["image_sha256"]):
                raise ValueError(f"selected image SHA mismatch: {sample['id']}")
            selected.append({**sample, "image": str(image_path)})

        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        text_config = model_config.get_text_config()
        layers, hidden_size = int(text_config.num_hidden_layers), int(text_config.hidden_size)
        target_layer = layer_output_index(layers)
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                                                  use_fast=False, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
            low_cpu_mem_usage=True, local_files_only=True)
        model.eval(); model.requires_grad_(False); assert_frozen(model)
        device = next(model.parameters()).device
        torch.cuda.reset_peak_memory_stats(device)
        query = str(config["query"])
        query_ids = processor.tokenizer(query, add_special_tokens=False).input_ids
        if len(query_ids) != int(config["gates"]["expected_query_tokens"]):
            raise ValueError("query token count drift")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        status_path.write_text('{"state":"running","stage":"extract","expected_pairs":8}\n', encoding="utf-8")

        def extract(image: Image.Image) -> tuple[np.ndarray, dict[str, int]]:
            inputs = processor(text=[rendered], images=[image], return_tensors="pt")
            ids = inputs["input_ids"][0].tolist()
            start, end = unique_subsequence(ids, query_ids)
            moved = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    moved[key] = value.to(device=device, dtype=torch.bfloat16) if key == "pixel_values" else value.to(device)
                else:
                    moved[key] = value
            with torch.inference_mode():
                outputs = model(**moved, output_hidden_states=True, use_cache=False, return_dict=True)
            states = outputs.hidden_states
            if states is None or len(states) != layers + 1:
                raise ValueError("unexpected hidden-state count")
            hidden = states[target_layer][0, start:end, :]
            if hidden.shape != (len(query_ids), hidden_size):
                raise ValueError(f"unexpected query hidden shape: {tuple(hidden.shape)}")
            vector = mean_pool_l2(hidden).cpu().numpy().astype(np.float32, copy=False)
            return vector, {"sequence_tokens": len(ids), "query_start": start, "query_end": end}

        vectors, rows, original_vectors = [], [], []
        first_image = None
        for sample in selected:
            with Image.open(sample["image"]) as loaded:
                original = loaded.convert("RGB")
            if first_image is None: first_image = original.copy()
            blank = Image.new("RGB", original.size, tuple(config["blank_rgb"]))
            original_vector, original_span = extract(original)
            blank_vector, blank_span = extract(blank)
            original_vectors.append(original_vector)
            for condition, vector, span in (("original", original_vector, original_span), ("blank", blank_vector, blank_span)):
                rows.append({**sample, "condition": condition, **span, "feature_index": len(vectors),
                             "input_contract": "pixels_plus_constant_neutral_prompt"})
                vectors.append(vector)
        assert first_image is not None
        duplicate, _ = extract(first_image)
        matrix = np.stack(vectors).astype(np.float32, copy=False)
        if matrix.shape != (16, hidden_size) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5):
            raise ValueError("invalid smoke feature matrix")
        original_matrix, blank_matrix = matrix[0::2], matrix[1::2]
        cosine_distances = (1.0 - np.sum(original_matrix * blank_matrix, axis=1)).tolist()
        l2_distances = np.linalg.norm(original_matrix - blank_matrix, axis=1).tolist()
        duplicate_max_abs = float(np.max(np.abs(original_vectors[0] - duplicate)))
        decision = decide_smoke(layers=layers, hidden_size=hidden_size, layer_index=target_layer,
            query_tokens=len(query_ids), duplicate_max_abs=duplicate_max_abs,
            cosine_distances=cosine_distances, l2_distances=l2_distances, gates=config["gates"])
        with (output_root / "features.npy").open("xb") as handle: np.save(handle, matrix, allow_pickle=False)
        with (output_root / "feature_rows.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        report = {"version": "task11b0-query-rep-smoke-result-v1", "state": "completed",
            "decision": decision, "architecture": {"layers": layers, "hidden_size": hidden_size,
            "layer_output_index": target_layer, "hidden_states_index_zero_is_embeddings": True},
            "query": query, "query_tokens": len(query_ids), "sample_count": len(selected), "feature_count": len(rows),
            "duplicate_max_abs": duplicate_max_abs,
            "original_blank_cosine_distance": {"values": cosine_distances, "minimum": min(cosine_distances),
                "median": float(np.median(cosine_distances)), "maximum": max(cosine_distances)},
            "original_blank_l2": {"values": l2_distances, "minimum": min(l2_distances),
                "median": float(np.median(l2_distances)), "maximum": max(l2_distances)},
            "model_all_parameters_frozen": True, "optimizer_created": False, "task8_locked_set_read": False,
            "task11b1_started": False, "elapsed_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "config_sha256": sha256_file(Path(config_path)), "model_identity": _model_identity(Path(model_path))}
        write_json_new(output_root / "smoke_report.json", report)
        write_json_new(output_root / "run_summary.json", {"state": "completed", "decision": decision["decision"],
            "features_sha256": sha256_file(output_root / "features.npy"), "feature_count": len(rows),
            "layer_output_index": target_layer, "all_parameters_frozen": True})
        signed = ["features.npy", "feature_rows.jsonl", "smoke_report.json", "run_summary.json"]
        with (output_root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed: handle.write(f"{sha256_file(output_root / name)}  {name}\n")
        status_path.write_text('{"state":"completed","stage":"done"}\n', encoding="utf-8")
        return report
    except Exception as exc:
        write_json_new(output_root / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        status_path.write_text('{"state":"failed","stage":"smoke"}\n', encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-rows", type=Path, required=True)
    parser.add_argument("--plantseg-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(config_path=args.config, base_rows_path=args.base_rows,
        plantseg_manifest_path=args.plantseg_manifest, model_path=args.model_path, output_root=args.output_root)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
