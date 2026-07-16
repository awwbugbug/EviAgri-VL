import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from static_qlora_data import EXPECTED_TARGET_KEYS
from static_qlora_config import load_training_config
from static_qlora_data import JsonlDataset


def generation_kwargs() -> dict[str, Any]:
    return {"max_new_tokens": 128, "do_sample": False, "temperature": None}


def pending_records(records: list[dict[str, Any]], prediction_path: Path) -> list[dict[str, Any]]:
    source_ids = {row["id"] for row in records}
    if len(source_ids) != len(records):
        raise ValueError("duplicate source record id")
    completed: set[str] = set()
    prediction_path = Path(prediction_path)
    if prediction_path.is_file():
        with prediction_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid prediction JSON at {prediction_path}:{line_number}: {error}"
                    ) from error
                record_id = row.get("id")
                if record_id in completed:
                    raise ValueError(f"duplicate prediction id: {record_id}")
                if record_id not in source_ids:
                    raise ValueError(f"prediction id not present in source split: {record_id}")
                completed.add(record_id)
    return [row for row in records if row["id"] not in completed]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"row at {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_predictions(
    records: list[dict[str, Any]],
    prediction_path: Path,
    generate_fn: Callable[[dict[str, Any]], str],
    progress_every: int = 50,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pending = pending_records(records, prediction_path)
    existing = len(records) - len(pending)
    generated = 0
    with prediction_path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in pending:
            prediction = generate_fn(row)
            output = {
                "id": row["id"],
                "split": row.get("split"),
                "task_type": row.get("task_type"),
                "target": row["target"],
                "prediction": prediction,
            }
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            generated += 1
            done = existing + generated
            if progress_callback is not None and (
                done % progress_every == 0 or done == len(records)
            ):
                progress_callback(done, len(records), row["id"])
    return {"existing": existing, "generated": generated, "total": len(records)}


def finalize_predictions(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    prediction_path = output_dir / "predictions.jsonl"
    remaining = pending_records(records, prediction_path)
    if remaining:
        raise ValueError(f"cannot finalize: {len(remaining)} predictions are missing")
    prediction_rows = _read_jsonl(prediction_path)
    metrics = compute_metrics(prediction_rows)
    failures = metrics["parse_failures"]
    published_metrics = {key: value for key, value in metrics.items() if key != "parse_failures"}
    _write_json_atomic(output_dir / "metrics.json", published_metrics)
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return published_metrics


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


def _valid_bbox(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def bbox_iou(prediction: Any, truth: Any) -> float:
    if not _valid_bbox(prediction) or not _valid_bbox(truth):
        return 0.0
    x1 = max(prediction[0], truth[0])
    y1 = max(prediction[1], truth[1])
    x2 = min(prediction[2], truth[2])
    y2 = min(prediction[3], truth[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    prediction_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    truth_area = (truth[2] - truth[0]) * (truth[3] - truth[1])
    union = prediction_area + truth_area - intersection
    return intersection / union if union > 0 else 0.0


def pointing_game(prediction: Any, truth: Any) -> float:
    if not _valid_bbox(prediction) or not _valid_bbox(truth):
        return 0.0
    center_x = (prediction[0] + prediction[2]) / 2
    center_y = (prediction[1] + prediction[3]) / 2
    return float(truth[0] <= center_x <= truth[2] and truth[1] <= center_y <= truth[3])


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _diagnosis_id(value: Any) -> Any:
    return value.get("pest_id") if isinstance(value, dict) else None


def _macro_f1(truth: list[Any], prediction: list[Any]) -> float:
    labels = sorted(set(truth), key=str)
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(truth, prediction))
        false_positive = sum(t != label and p == label for t, p in zip(truth, prediction))
        false_negative = sum(t == label and p != label for t, p in zip(truth, prediction))
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        scores.append(_safe_divide(2 * precision * recall, precision + recall))
    return sum(scores) / len(scores)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        try:
            parsed_rows.append((row, parse_structured_json(row["prediction"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append({"id": str(row.get("id", "<missing-id>")), "error": str(error)})

    true_positive = false_positive = false_negative = 0
    for row, parsed in parsed_rows:
        truth_present = row["target"]["evidence_present"] is True
        predicted_present = parsed["evidence_present"] is True
        true_positive += int(truth_present and predicted_present)
        false_positive += int(not truth_present and predicted_present)
        false_negative += int(truth_present and not predicted_present)
    presence_precision = _safe_divide(true_positive, true_positive + false_positive)
    presence_recall = _safe_divide(true_positive, true_positive + false_negative)
    presence_f1 = _safe_divide(
        2 * presence_precision * presence_recall, presence_precision + presence_recall
    )

    parsed_by_id = {row["id"]: parsed for row, parsed in parsed_rows}
    positives = [row for row in rows if row["target"]["evidence_present"] is True]
    positive_truth_ids = [_diagnosis_id(row["target"]["diagnosis"]) for row in positives]
    positive_prediction_ids = [
        _diagnosis_id(parsed_by_id.get(row["id"], {}).get("diagnosis")) for row in positives
    ]
    diagnosis_correct = sum(
        truth == prediction
        for truth, prediction in zip(positive_truth_ids, positive_prediction_ids)
    )
    ious = [
        bbox_iou(parsed_by_id.get(row["id"], {}).get("evidence_bbox"), row["target"]["evidence_bbox"])
        for row in positives
    ]
    pointing = [
        pointing_game(
            parsed_by_id.get(row["id"], {}).get("evidence_bbox"),
            row["target"]["evidence_bbox"],
        )
        for row in positives
    ]
    valid_bbox = sum(
        _valid_bbox(parsed_by_id.get(row["id"], {}).get("evidence_bbox")) for row in positives
    )

    nulls = [row for row in rows if row["target"]["evidence_present"] is False]
    null_false_positive = 0
    evidence_bound_hallucination = 0
    for row in nulls:
        parsed = parsed_by_id.get(row["id"])
        if parsed is None:
            null_false_positive += 1
            evidence_bound_hallucination += 1
            continue
        null_false_positive += int(parsed.get("evidence_present") is not False)
        evidence_bound_hallucination += int(
            parsed.get("evidence_present") is not False
            or parsed.get("evidence_bbox") is not None
            or parsed.get("diagnosis") != "uncertain"
            or parsed.get("reliability") != "insufficient_visual_evidence"
        )

    reliability_correct = sum(
        parsed_by_id.get(row["id"], {}).get("reliability") == row["target"]["reliability"]
        for row in rows
    )
    return {
        "samples": len(rows),
        "schema_valid_count": len(parsed_rows),
        "schema_valid_rate": _safe_divide(len(parsed_rows), len(rows)),
        "parse_failure_count": len(failures),
        "parse_failures": failures,
        "evidence_presence_precision": presence_precision,
        "evidence_presence_recall": presence_recall,
        "evidence_presence_f1": presence_f1,
        "positive": {
            "samples": len(positives),
            "diagnosis_accuracy": _safe_divide(diagnosis_correct, len(positives)),
            "diagnosis_macro_f1": _macro_f1(positive_truth_ids, positive_prediction_ids),
            "bbox_valid_rate": _safe_divide(valid_bbox, len(positives)),
            "mean_iou": _safe_divide(sum(ious), len(positives)),
            "iou_at_0.5": _safe_divide(sum(iou >= 0.5 for iou in ious), len(positives)),
            "pointing_game": _safe_divide(sum(pointing), len(positives)),
        },
        "null": {
            "samples": len(nulls),
            "false_positive_rate": _safe_divide(null_false_positive, len(nulls)),
            "evidence_bound_hallucination_rate": _safe_divide(
                evidence_bound_hallucination, len(nulls)
            ),
        },
        "reliability_accuracy": _safe_divide(reliability_correct, len(rows)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_generation_stack(config: dict[str, Any], adapter_dir: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

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
    if not getattr(base, "is_loaded_in_4bit", False):
        raise RuntimeError("evaluation base model was not loaded in 4-bit mode")
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        config["model_path"],
        min_pixels=config["vision"]["min_pixels"],
        max_pixels=config["vision"]["max_pixels"],
        use_fast=False,
        local_files_only=True,
    )
    return model, processor


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
        output_ids = model.generate(**inputs, **generation_kwargs())
    continuation = output_ids[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def evaluate_adapter(
    config: dict[str, Any],
    splits: tuple[str, ...],
    output_root: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    import peft
    import bitsandbytes

    if any(split not in {"val", "test"} for split in splits):
        raise ValueError(f"evaluation splits must be val/test: {splits}")
    formal = Path(config["experiment_root"]) / "formal"
    if not (formal / "run_summary.json").is_file() or (formal / "failure.json").exists():
        raise RuntimeError("formal training is not complete and valid")
    adapter_dir = formal / "adapter"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError(f"missing final adapter: {adapter_file}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: JsonlDataset(Path(config["mixed_data_root"]) / f"{split}.jsonl") for split in splits
    }
    pending_counts = {
        split: len(pending_records(dataset.records, output_root / split / "predictions.jsonl"))
        for split, dataset in datasets.items()
        if not (output_root / split / "metrics.json").is_file()
    }
    started = time.time()
    model = processor = None
    summary: dict[str, Any] = {
        "version": "static_qlora_v1",
        "adapter": str(adapter_dir),
        "adapter_sha256": _sha256_file(adapter_file),
        "splits": {},
    }
    try:
        if any(pending_counts.values()):
            model, processor = _load_generation_stack(config, adapter_dir)
        for split in splits:
            dataset = datasets[split]
            split_output = output_root / split
            split_output.mkdir(parents=True, exist_ok=True)
            metrics_path = split_output / "metrics.json"
            if metrics_path.is_file():
                summary["splits"][split] = json.loads(metrics_path.read_text(encoding="utf-8"))
                continue

            split_started = time.time()

            def progress(done: int, total: int, record_id: str, split_name: str = split) -> None:
                elapsed = time.time() - split_started
                rate = done / elapsed if elapsed > 0 else 0.0
                status = {
                    "state": "running",
                    "split": split_name,
                    "done": done,
                    "total": total,
                    "record_id": record_id,
                    "elapsed_seconds": elapsed,
                    "samples_per_second": rate,
                    "eta_seconds": (total - done) / rate if rate > 0 else None,
                }
                _write_json_atomic(output_root / "status.json", status)
                print(json.dumps(status), flush=True)

            generation_summary = generate_predictions(
                dataset.records,
                split_output / "predictions.jsonl",
                generate_fn=lambda row: _generate_one(model, processor, row),
                progress_every=50,
                progress_callback=progress,
            )
            metrics = finalize_predictions(dataset.records, split_output)
            summary["splits"][split] = {
                "generation": generation_summary,
                "metrics": metrics,
                "elapsed_seconds": time.time() - split_started,
            }
        summary["completed"] = True
        summary["elapsed_seconds"] = time.time() - started
        summary["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        }
        _write_json_atomic(output_root / "evaluation_summary.json", summary)
        _write_json_atomic(output_root / "status.json", {"state": "completed", "summary": str(output_root / "evaluation_summary.json")})
        return summary
    except Exception as error:
        failure = {
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        _write_json_atomic(output_root / f"failure_{int(time.time())}.json", failure)
        _write_json_atomic(output_root / "status.json", {"state": "failed", **failure})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Static QLoRA v1 on val/test")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    summary = evaluate_adapter(config, tuple(args.splits), args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
