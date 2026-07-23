"""Extract frozen Qwen visual features for Task 11A counterfactual stress inputs."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from extract_task10b_features import (
    MAX_PIXELS,
    MIN_PIXELS,
    _model_identity,
    _read_manifest,
    _write_feature_payload,
    _write_json_replace,
    assert_frozen,
    build_feature_matrix,
    prepare_visual_inputs,
)
from task10_audit_common import ensure_new_directory, write_json_new
from task11a_confidence_router import build_stress_rows, transform_image


def _load_runtime():
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    return Image, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def extract_stress_features(
    *,
    source_manifest: Path,
    model_path: Path,
    output_root: Path,
    runtime_loader: Callable[[], tuple[Any, Any, Any]] = _load_runtime,
) -> dict[str, Any]:
    destination = Path(output_root)
    ensure_new_directory(destination)
    _write_json_replace(destination / "status.json", {"state": "running", "stage": "load"})
    started = time.monotonic()
    try:
        Image, AutoProcessor, Qwen2_5_VLForConditionalGeneration = runtime_loader()
        source_rows = _read_manifest(Path(source_manifest), None)
        rows = build_stress_rows(source_rows)
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
                raise ValueError(f"missing Task 11A image: {image_path}")
            with Image.open(image_path) as loaded:
                transformed = transform_image(
                    loaded,
                    condition=str(row["condition"]),
                    split=str(row["split"]),
                    row_id=str(row["id"]),
                    seed=int(row["stress_seed"]),
                )
            inputs = prepare_visual_inputs(processor, transformed)
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
            manifest_path=Path(source_manifest),
            config={
                "version": "task11a-stress-feature-config-1",
                "model_path": str(Path(model_path)),
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "pooling": "post_merge_token_mean_then_l2",
                "val_transforms": {
                    "blank_rgb": [127, 127, 127],
                    "blur_radius": 6,
                    "shuffle_grid": 6,
                },
                "dev_transforms": {
                    "blank_rgb": [114, 114, 114],
                    "blur_radius": 10,
                    "shuffle_grid": 8,
                },
                "shuffle_seeds": [17, 29, 43],
            },
            model_identity=identity,
            run_metadata={
                "elapsed_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(
                    torch.cuda.max_memory_allocated(visual_parameter.device)
                ),
            },
            summary_version="task11a-stress-feature-summary-1",
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
    parser = argparse.ArgumentParser(description="Extract Task 11A stress features")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = extract_stress_features(
        source_manifest=arguments.source_manifest,
        model_path=arguments.model_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
