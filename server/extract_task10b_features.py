"""Extract frozen post-merge Qwen visual features for Task 10B."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


MIN_PIXELS = 200704
MAX_PIXELS = 401408


def mean_pool_l2(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError("visual tokens must be two-dimensional")
    if tokens.shape[0] == 0 or tokens.shape[1] == 0:
        raise ValueError("visual tokens must be non-empty")
    values = tokens.detach().to(dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("visual tokens contain non-finite values")
    pooled = values.mean(dim=0)
    norm = torch.linalg.vector_norm(pooled)
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        raise ValueError("visual token mean has zero-norm")
    return pooled / norm


def assert_frozen(model: torch.nn.Module) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"model has trainable parameters: {trainable[:10]}")


def build_feature_matrix(
    rows: Iterable[dict[str, Any]],
    token_fn: Callable[[dict[str, Any]], torch.Tensor],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors = []
    feature_rows = []
    dimension = None
    for index, row in enumerate(rows):
        vector = mean_pool_l2(token_fn(row)).cpu().numpy().astype(np.float32, copy=False)
        if dimension is None:
            dimension = int(vector.shape[0])
        elif vector.shape != (dimension,):
            raise ValueError("inconsistent feature dimensions")
        vectors.append(vector)
        feature_rows.append({**row, "feature_index": index})
    if not vectors:
        raise ValueError("feature manifest is empty")
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("feature matrix is not L2-normalized")
    return matrix, feature_rows


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _write_json_replace(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_feature_payload(
    *,
    matrix: np.ndarray,
    feature_rows: list[dict[str, Any]],
    output_root: Path,
    manifest_path: Path,
    config: dict[str, Any],
    model_identity: dict[str, str],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(output_root)
    if matrix.ndim != 2 or matrix.shape[0] != len(feature_rows) or not len(feature_rows):
        raise ValueError("feature matrix/row cardinality mismatch")
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("features must be finite float32")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("serialized features must be unit-normalized")
    with (destination / "features.npy").open("xb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    _write_jsonl_new(destination / "feature_rows.jsonl", feature_rows)
    snapshot = {
        **config,
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "model_identity": dict(sorted(model_identity.items())),
    }
    write_json_new(destination / "config.snapshot.json", snapshot)
    summary = {
        "version": "task10b-v2-feature-summary-1",
        "state": "completed",
        "feature_count": int(matrix.shape[0]),
        "feature_dimension": int(matrix.shape[1]),
        "feature_dtype": str(matrix.dtype),
        "minimum_l2_norm": float(norms.min()),
        "maximum_l2_norm": float(norms.max()),
        "manifest_sha256": snapshot["manifest_sha256"],
        "features_sha256": sha256_file(destination / "features.npy"),
        "feature_rows_sha256": sha256_file(destination / "feature_rows.jsonl"),
        "all_parameters_frozen": True,
        **(run_metadata or {}),
    }
    write_json_new(destination / "run_summary.json", summary)
    signed = ["features.npy", "feature_rows.jsonl", "config.snapshot.json", "run_summary.json"]
    with (destination / "completion.sha256").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for name in signed:
            handle.write(f"{sha256_file(destination / name)}  {name}\n")
    return summary


def write_feature_outputs(
    *,
    matrix: np.ndarray,
    feature_rows: list[dict[str, Any]],
    output_root: Path,
    manifest_path: Path,
    config: dict[str, Any],
    model_identity: dict[str, str],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    summary = _write_feature_payload(
        matrix=matrix,
        feature_rows=feature_rows,
        output_root=destination,
        manifest_path=manifest_path,
        config=config,
        model_identity=model_identity,
        run_metadata=run_metadata,
    )
    write_json_new(destination / "status.json", {"state": "completed", "stage": "features"})
    return summary


def _model_identity(model_path: Path) -> dict[str, str]:
    root = Path(model_path)
    names = {
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    files = [path for path in root.iterdir() if path.name in names or path.suffix == ".safetensors"]
    if not any(path.name == "config.json" for path in files):
        raise ValueError("model identity lacks config.json")
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("model identity lacks safetensors weights")
    return {path.name: sha256_file(path) for path in sorted(files)}


def _read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("feature manifest is empty")
    return rows


def extract_features(
    *,
    manifest_path: Path,
    model_path: Path,
    output_root: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    destination = Path(output_root)
    ensure_new_directory(destination)
    _write_json_replace(destination / "status.json", {"state": "running", "stage": "load"})
    started = time.monotonic()
    try:
        rows = _read_manifest(Path(manifest_path), limit)
        identity = _model_identity(Path(model_path))
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            use_fast=False,
            local_files_only=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model.eval()
        model.requires_grad_(False)
        assert_frozen(model)
        visual_parameter = next(model.visual.parameters())
        torch.cuda.reset_peak_memory_stats(visual_parameter.device)
        _write_json_replace(
            destination / "status.json",
            {"state": "running", "stage": "extract", "expected": len(rows)},
        )

        def token_fn(row: dict[str, Any]) -> torch.Tensor:
            image_path = Path(str(row.get("image", "")))
            if not image_path.is_file():
                raise ValueError(f"missing manifest image: {image_path}")
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
            inputs = processor(images=[image], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(
                device=visual_parameter.device,
                dtype=visual_parameter.dtype,
            )
            grid = inputs["image_grid_thw"].to(device=visual_parameter.device)
            with torch.inference_mode():
                tokens = model.visual(pixel_values, grid_thw=grid)
            if not isinstance(tokens, torch.Tensor):
                raise ValueError("Qwen visual tower did not return a tensor")
            return tokens.detach().cpu()

        matrix, feature_rows = build_feature_matrix(rows, token_fn)
        summary = _write_feature_payload(
            matrix=matrix,
            feature_rows=feature_rows,
            output_root=destination,
            manifest_path=Path(manifest_path),
            config={
                "version": "task10b-v2-feature-config-1",
                "model_path": str(Path(model_path)),
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "pooling": "post_merge_token_mean_then_l2",
                "limit": limit,
            },
            model_identity=identity,
            run_metadata={
                "elapsed_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(visual_parameter.device)),
            },
        )
        _write_json_replace(destination / "status.json", {"state": "completed", "stage": "done"})
        return summary
    except Exception as exc:
        write_json_new(
            destination / "failure.json",
            {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()},
        )
        _write_json_replace(destination / "status.json", {"state": "failed", "stage": "features"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Task 10B Qwen visual features")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    summary = extract_features(
        manifest_path=arguments.manifest,
        model_path=arguments.model_path,
        output_root=arguments.output_root,
        limit=arguments.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
