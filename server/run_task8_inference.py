from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from static_qlora_config import load_training_config
from task8_protocol import GROUPS, generation_kwargs, protocol_hash


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"row at {path}:{line_number} is not an object")
            rows.append(row)
    return rows


def validate_jobs(jobs: list[dict[str, Any]]) -> None:
    job_ids: set[str] = set()
    by_audit: dict[str, dict[str, dict[str, Any]]] = {}
    for job in jobs:
        job_id = str(job.get("job_id"))
        if job_id in job_ids:
            raise ValueError(f"duplicate job_id: {job_id}")
        job_ids.add(job_id)
        group = str(job.get("group"))
        if group not in GROUPS:
            raise ValueError(f"unknown group: {group}")
        if job.get("protocol_hash") != protocol_hash(group):
            raise ValueError(f"protocol hash mismatch: {job_id}")
        if not Path(str(job.get("image", ""))).is_file():
            raise ValueError(f"missing job image: {job_id}")
        audit_id = str(job.get("audit_id"))
        by_audit.setdefault(audit_id, {})[group] = job
    for audit_id, groups in by_audit.items():
        if set(groups) != set(GROUPS):
            raise ValueError(f"audit_id does not contain B0-B3: {audit_id}")
        b1, b2 = groups["B1"], groups["B2"]
        for key in ("prompt", "protocol_hash", "image", "image_sha256"):
            if b1.get(key) != b2.get(key):
                raise ValueError(f"B1/B2 input mismatch for {audit_id}: {key}")


def build_messages(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(job["image"])},
                {"type": "text", "text": str(job["prompt"])},
            ],
        }
    ]


def pending_jobs(jobs: list[dict[str, Any]], prediction_path: Path) -> list[dict[str, Any]]:
    known = {str(job["job_id"]) for job in jobs}
    completed: set[str] = set()
    prediction_path = Path(prediction_path)
    if prediction_path.is_file():
        for row in _load_jsonl(prediction_path):
            job_id = str(row.get("job_id"))
            if job_id not in known:
                raise ValueError(f"unknown completed job_id: {job_id}")
            if job_id in completed:
                raise ValueError(f"duplicate completed job_id: {job_id}")
            completed.add(job_id)
    return [job for job in jobs if str(job["job_id"]) not in completed]


def generate_group_predictions(
    jobs: list[dict[str, Any]],
    prediction_path: Path,
    generate_fn: Callable[[dict[str, Any]], str],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pending = pending_jobs(jobs, prediction_path)
    existing = len(jobs) - len(pending)
    generated = 0
    with prediction_path.open("a", encoding="utf-8", newline="\n") as handle:
        for job in pending:
            output = {**job, "prediction": generate_fn(job)}
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            generated += 1
            if progress_callback is not None:
                progress_callback(existing + generated, len(jobs), str(job["job_id"]))
    return {"existing": existing, "generated": generated, "total": len(jobs)}


def _load_base(config: dict[str, Any]):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

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
    model.eval()
    processor = AutoProcessor.from_pretrained(
        config["model_path"],
        min_pixels=config["vision"]["min_pixels"],
        max_pixels=config["vision"]["max_pixels"],
        use_fast=False,
        local_files_only=True,
    )
    return model, processor


def _generate_one(model, processor, job: dict[str, Any]) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = build_messages(job)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs())
    continuation = output_ids[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_inference(
    jobs: list[dict[str, Any]],
    config: dict[str, Any],
    adapter_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel

    validate_jobs(jobs)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    summaries: dict[str, Any] = {}
    for groups, use_adapter in ((('B0', 'B1'), False), (('B2', 'B3'), True)):
        grouped_jobs = {group: [job for job in jobs if job["group"] == group] for group in groups}
        if not any(pending_jobs(rows, output_root / group / "predictions.jsonl") for group, rows in grouped_jobs.items()):
            continue
        base, processor = _load_base(config)
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False) if use_adapter else base
        model.eval()
        for group, rows in grouped_jobs.items():
            group_started = time.time()
            status_path = output_root / "status.json"

            def progress(done: int, total: int, job_id: str, group_name: str = group) -> None:
                _write_json_atomic(
                    status_path,
                    {"state": "running", "group": group_name, "done": done, "total": total, "job_id": job_id},
                )

            generation = generate_group_predictions(
                rows,
                output_root / group / "predictions.jsonl",
                lambda job: _generate_one(model, processor, job),
                progress_callback=progress,
            )
            summaries[group] = {
                "generation": generation,
                "protocol_hash": protocol_hash(group),
                "elapsed_seconds": time.time() - group_started,
            }
        del model, base, processor
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "completed": all(
            not pending_jobs(
                [job for job in jobs if job["group"] == group],
                output_root / group / "predictions.jsonl",
            )
            for group in GROUPS
        ),
        "groups": summaries,
        "adapter_sha256": _sha256_file(adapter_dir / "adapter_model.safetensors"),
        "elapsed_seconds": time.time() - started,
        "peak_vram_bytes": torch.cuda.max_memory_reserved(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    _write_json_atomic(output_root / "run_summary.json", summary)
    _write_json_atomic(output_root / "status.json", {"state": "completed", "completed": summary["completed"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 8 same-protocol inference")
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--leakage-report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    leakage = json.loads(args.leakage_report.read_text(encoding="utf-8"))
    if leakage.get("passed") is not True:
        raise SystemExit("refusing Task 8 inference: leakage report did not pass")
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_inference(
            _load_jsonl(args.jobs),
            load_training_config(args.config),
            args.adapter_dir,
            args.output_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as error:
        failure = {"error": str(error), "traceback": traceback.format_exc()}
        _write_json_atomic(args.output_root / f"failure_{int(time.time())}.json", failure)
        _write_json_atomic(args.output_root / "status.json", {"state": "failed", **failure})
        raise


if __name__ == "__main__":
    main()
