"""Extract global and annotation-oracle pools from one full-frame Qwen encoding."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from extract_task10b_features import (
    MAX_PIXELS,
    MIN_PIXELS,
    _model_identity,
    assert_frozen,
    mean_pool_l2,
    prepare_visual_inputs,
)
from task10_audit_common import ensure_new_directory, sha256_file, write_json_new


MASK_OCCUPANCY_THRESHOLD = 0.05


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bbox_region_indices(
    bbox: list[float], image_width: int, image_height: int, grid_height: int, grid_width: int
) -> np.ndarray:
    if min(image_width, image_height, grid_height, grid_width) <= 0 or len(bbox) != 4:
        raise ValueError("invalid bbox-grid input")
    x1, y1, x2, y2 = map(float, bbox)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > image_width or y2 > image_height:
        raise ValueError("bbox lies outside source image")
    xs = (np.arange(grid_width, dtype=np.float64) + 0.5) * image_width / grid_width
    ys = (np.arange(grid_height, dtype=np.float64) + 0.5) * image_height / grid_height
    selected = (ys[:, None] >= y1) & (ys[:, None] < y2) & (xs[None, :] >= x1) & (xs[None, :] < x2)
    flat = np.flatnonzero(selected.ravel())
    if len(flat):
        return flat.astype(np.int64)

    cell_x1 = np.arange(grid_width, dtype=np.float64) * image_width / grid_width
    cell_x2 = (np.arange(grid_width, dtype=np.float64) + 1.0) * image_width / grid_width
    cell_y1 = np.arange(grid_height, dtype=np.float64) * image_height / grid_height
    cell_y2 = (np.arange(grid_height, dtype=np.float64) + 1.0) * image_height / grid_height
    overlap_width = np.maximum(0.0, np.minimum(cell_x2[None, :], x2) - np.maximum(cell_x1[None, :], x1))
    overlap_height = np.maximum(0.0, np.minimum(cell_y2[:, None], y2) - np.maximum(cell_y1[:, None], y1))
    overlap = overlap_width * overlap_height
    if float(overlap.max()) <= 0:
        raise ValueError("bbox has no token-cell overlap")
    return np.asarray([int(overlap.argmax())], dtype=np.int64)


def mask_region_indices(mask: np.ndarray, grid_height: int, grid_width: int) -> np.ndarray:
    if mask.ndim != 2 or min(mask.shape) <= 0 or min(grid_height, grid_width) <= 0:
        raise ValueError("invalid mask-grid input")
    binary = (mask > 0).astype(np.uint8) * 255
    resized = Image.fromarray(binary, mode="L").resize(
        (grid_width, grid_height), resample=Image.Resampling.BOX
    )
    occupancy = np.asarray(resized, dtype=np.float32) / 255.0
    selected = np.flatnonzero(occupancy.ravel() >= MASK_OCCUPANCY_THRESHOLD)
    if not len(selected):
        raise ValueError("PlantSeg mask selects no post-merge token")
    return selected.astype(np.int64)


def pool_global_region(tokens: torch.Tensor, region_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tokens.ndim != 2 or not len(region_indices):
        raise ValueError("invalid oracle token pool")
    if int(region_indices.min()) < 0 or int(region_indices.max()) >= tokens.shape[0]:
        raise ValueError("oracle region index exceeds token grid")
    global_vector = mean_pool_l2(tokens).cpu().numpy().astype(np.float32, copy=False)
    region_vector = mean_pool_l2(tokens[torch.as_tensor(region_indices, dtype=torch.long)]).cpu().numpy().astype(np.float32, copy=False)
    return global_vector, region_vector


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_status(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract(*, manifest_path: Path, model_path: Path, output_root: Path) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    root = Path(output_root)
    ensure_new_directory(root)
    _write_status(root / "status.json", {"state": "running", "stage": "load"})
    started = time.monotonic()
    try:
        rows = read_jsonl(Path(manifest_path))
        if len(rows) != 144:
            raise ValueError("Task14A manifest must contain 144 rows")
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
        parameter = next(model.visual.parameters())
        merge = int(model.visual.spatial_merge_size)
        if merge <= 0:
            raise ValueError("invalid Qwen spatial merge size")
        torch.cuda.reset_peak_memory_stats(parameter.device)
        globals_: list[np.ndarray] = []
        regions: list[np.ndarray] = []
        feature_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            _write_status(root / "status.json", {"state": "running", "stage": "extract", "completed": index, "expected": len(rows)})
            image_path = Path(str(row.get("image", "")))
            if not image_path.is_file() or sha256_file(image_path) != str(row.get("source_image_sha256", "")):
                raise ValueError(f"image missing or SHA256 mismatch: {row.get('id')}")
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
            inputs = prepare_visual_inputs(processor, image)
            grid = inputs["image_grid_thw"]
            if grid.shape != (1, 3):
                raise ValueError("Task14A supports one static image per row")
            temporal, raw_h, raw_w = map(int, grid[0].tolist())
            if temporal != 1 or raw_h % merge or raw_w % merge:
                raise ValueError("unexpected Qwen image grid")
            grid_h, grid_w = raw_h // merge, raw_w // merge
            pixel_values = inputs["pixel_values"].to(device=parameter.device, dtype=parameter.dtype)
            with torch.inference_mode():
                tokens = model.visual(pixel_values, grid_thw=grid.to(parameter.device))
            if not isinstance(tokens, torch.Tensor) or tokens.shape[0] != grid_h * grid_w:
                raise ValueError("post-merge token count does not match spatial grid")
            tokens = tokens.detach().cpu()
            if row.get("target_type") == "positive":
                region_indices = bbox_region_indices(
                    list(row["evidence_bbox"]), image.width, image.height, grid_h, grid_w
                )
            elif row.get("target_type") == "real_null":
                mask_path = Path(str(row.get("mask", "")))
                if not mask_path.is_file() or sha256_file(mask_path) != str(row.get("mask_sha256", "")):
                    raise ValueError(f"mask missing or SHA256 mismatch: {row.get('id')}")
                with Image.open(mask_path) as loaded:
                    mask = np.asarray(loaded.convert("L"))
                if mask.shape != (image.height, image.width):
                    raise ValueError("PlantSeg mask/image size mismatch")
                region_indices = mask_region_indices(mask, grid_h, grid_w)
            else:
                raise ValueError("unknown Task14A target type")
            global_vector, region_vector = pool_global_region(tokens, region_indices)
            globals_.append(global_vector)
            regions.append(region_vector)
            feature_rows.append(
                {
                    **row,
                    "feature_index": index,
                    "source_size": [image.width, image.height],
                    "post_merge_grid": [grid_h, grid_w],
                    "post_merge_token_count": grid_h * grid_w,
                    "region_token_count": int(len(region_indices)),
                    "region_token_fraction": float(len(region_indices) / (grid_h * grid_w)),
                }
            )
        global_matrix = np.stack(globals_).astype(np.float32, copy=False)
        region_matrix = np.stack(regions).astype(np.float32, copy=False)
        for name, matrix in (("global", global_matrix), ("region", region_matrix)):
            if matrix.shape != (144, 2048) or not np.isfinite(matrix).all():
                raise ValueError(f"invalid {name} feature matrix")
            if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5, rtol=1e-5):
                raise ValueError(f"{name} features are not unit-normalized")
        with (root / "global_features.npy").open("xb") as handle:
            np.save(handle, global_matrix, allow_pickle=False)
        with (root / "region_features.npy").open("xb") as handle:
            np.save(handle, region_matrix, allow_pickle=False)
        _write_jsonl_new(root / "feature_rows.jsonl", feature_rows)
        write_json_new(
            root / "config.snapshot.json",
            {
                "version": "task14a-oracle-feature-config-1",
                "model_path": str(Path(model_path)),
                "model_identity": _model_identity(Path(model_path)),
                "manifest_sha256": sha256_file(Path(manifest_path)),
                "full_frame_encodings_per_image": 1,
                "post_merge_order": "row_major_after_qwen_reverse_indices",
                "positive_region": "bbox_center_cells_with_max_overlap_fallback",
                "null_region": "mask_cell_occupancy_ge_0.05",
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            },
        )
        summary = {
            "version": "task14a-oracle-feature-summary-1",
            "state": "completed",
            "feature_count": 144,
            "feature_dimension": 2048,
            "all_parameters_frozen": True,
            "full_frame_encodings_per_image": 1,
            "manifest_sha256": sha256_file(Path(manifest_path)),
            "global_features_sha256": sha256_file(root / "global_features.npy"),
            "region_features_sha256": sha256_file(root / "region_features.npy"),
            "feature_rows_sha256": sha256_file(root / "feature_rows.jsonl"),
            "elapsed_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(parameter.device)),
        }
        write_json_new(root / "run_summary.json", summary)
        signed = ["global_features.npy", "region_features.npy", "feature_rows.jsonl", "config.snapshot.json", "run_summary.json"]
        with (root / "completion.sha256").open("x", encoding="utf-8", newline="\n") as handle:
            for name in signed:
                handle.write(f"{sha256_file(root / name)}  {name}\n")
        _write_status(root / "status.json", {"state": "completed", "stage": "done"})
        return summary
    except Exception as exc:
        write_json_new(root / "failure.json", {"state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        _write_status(root / "status.json", {"state": "failed", "stage": "features"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(manifest_path=args.manifest, model_path=args.model_path, output_root=args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
