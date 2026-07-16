from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE_ORDER = ("adult", "egg", "larva", "pupa")


def extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _round_robin(groups: Iterable[list[dict[str, str]]], limit: int) -> list[dict[str, str]]:
    queues = [list(group) for group in groups if group]
    selected: list[dict[str, str]] = []
    while queues and len(selected) < limit:
        remaining: list[list[dict[str, str]]] = []
        for queue in queues:
            if len(selected) >= limit:
                break
            selected.append(queue.pop(0))
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def select_probe_rows(
    rows: list[dict[str, str]], age_limit: int, ip102_limit: int
) -> list[dict[str, str]]:
    age_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") == "ages":
            age_groups[row.get("stage_normalized", "")].append(row)
    ordered_age_groups = []
    for stage in STAGE_ORDER:
        ordered_age_groups.append(
            sorted(age_groups.get(stage, []), key=lambda row: row.get("selection_hash", ""))
        )
    age_selected = _round_robin(ordered_age_groups, age_limit)

    ip_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") == "IP102":
            ip_groups[row.get("class_name", "")].append(row)
    ordered_ip_groups = [
        sorted(group, key=lambda row: row.get("selection_hash", ""))
        for _, group in sorted(
            ip_groups.items(),
            key=lambda item: min(row.get("selection_hash", "") for row in item[1]),
        )
    ]
    ip_selected = _round_robin(ordered_ip_groups, ip102_limit)
    return age_selected + ip_selected


def build_prompt(row: dict[str, str], allowed_labels: list[str]) -> str:
    labels = " | ".join(allowed_labels)
    schema = (
        '{"diagnosis":"<allowed label or uncertain>",'
        '"stage":"<adult|egg|larva|pupa|not_applicable|uncertain>",'
        '"evidence_present":true,'
        '"evidence_bbox":[x1,y1,x2,y2],'
        '"visible_attributes":["attribute"],'
        '"reliability":"supported|insufficient_evidence"}'
    )
    common = (
        "Return JSON only, without markdown. Use this exact schema: "
        f"{schema}\n"
        "Bounding-box coordinates must be integers normalized to 0-1000. "
        "If the insect or discriminative evidence is not visible, set diagnosis=uncertain, "
        "stage=uncertain, evidence_present=false, evidence_bbox=null, "
        "visible_attributes=[], reliability=insufficient_evidence. "
        "Do not invent visual attributes."
    )
    if row.get("dataset") == "ages":
        return (
            "Perform closed-set pest species and developmental-stage diagnosis.\n"
            f"Allowed pest species: {labels}\n"
            "Allowed stages: adult|egg|larva|pupa.\n"
            f"{common}"
        )
    return (
        "Perform closed-set insect-pest diagnosis.\n"
        f"Allowed pest labels: {labels}\n"
        "Set stage=not_applicable when diagnosis is supported because this dataset has no stage label.\n"
        f"{common}"
    )


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def candidate_labels(taxonomy_rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    age = sorted(
        {row.get("species", "").strip() for row in taxonomy_rows if row.get("dataset") == "ages"}
        - {""}
    )
    ip102 = sorted(
        {
            row.get("class_name", "").strip()
            for row in taxonomy_rows
            if row.get("dataset") == "IP102"
        }
        - {""}
    )
    return age, ip102


def summarize_results(
    results: list[dict[str, Any]], model_path: str, peak_vram_gb: float
) -> dict[str, Any]:
    age = [result for result in results if result["dataset"] == "ages"]
    ip102 = [result for result in results if result["dataset"] == "IP102"]

    def accuracy(group: list[dict[str, Any]], field: str) -> float | None:
        if not group:
            return None
        return sum(bool(result[field]) for result in group) / len(group)

    parsed_count = sum(result["parsed"] is not None for result in results)
    return {
        "status": "ZERO_SHOT_PROBE_OK",
        "model_path": model_path,
        "sample_count": len(results),
        "age_count": len(age),
        "ip102_count": len(ip102),
        "parsed_count": parsed_count,
        "parse_rate": parsed_count / len(results) if results else None,
        "diagnosis_correct": sum(bool(result["diagnosis_correct"]) for result in results),
        "diagnosis_accuracy": accuracy(results, "diagnosis_correct"),
        "age_diagnosis_accuracy": accuracy(age, "diagnosis_correct"),
        "ip102_diagnosis_accuracy": accuracy(ip102, "diagnosis_correct"),
        "age_stage_correct": sum(bool(result["stage_correct"]) for result in age),
        "age_stage_accuracy": accuracy(age, "stage_correct"),
        "mean_seconds": (
            sum(float(result["seconds"]) for result in results) / len(results) if results else None
        ),
        "peak_vram_gb": peak_vram_gb,
        "protocol_note": "MVP pipeline probe only; not a final paper metric.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-limit", type=int, default=8)
    parser.add_argument("--ip102-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset_root / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    taxonomy_path = args.dataset_root / "taxonomy.csv"
    if not taxonomy_path.is_file():
        raise SystemExit(f"taxonomy file is required: {taxonomy_path}")
    with taxonomy_path.open(encoding="utf-8-sig", newline="") as handle:
        taxonomy_rows = list(csv.DictReader(handle))
    selected = select_probe_rows(rows, args.age_limit, args.ip102_limit)
    if not selected:
        raise SystemExit("no probe rows selected")

    age_labels, ip102_labels = candidate_labels(taxonomy_rows)
    if len(age_labels) != 102 or len(ip102_labels) != 102:
        raise SystemExit(
            f"expected 102 candidate labels per branch, found age={len(age_labels)} ip102={len(ip102_labels)}"
        )

    model_manifest = args.model_manifest
    if model_manifest is None:
        value = os.environ.get("MODEL_MANIFEST")
        if not value:
            raise SystemExit("MODEL_MANIFEST is not set")
        model_manifest = Path(value)
    model_record = json.loads(model_manifest.read_text(encoding="utf-8"))
    model_path = model_record["resolved_path"]

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.generation_config.temperature = None
    torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "probe_results.jsonl"
    results: list[dict[str, Any]] = []
    with result_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(selected, start=1):
            candidates = age_labels if row["dataset"] == "ages" else ip102_labels
            prompt = build_prompt(row, candidates)
            image_path = (args.dataset_root / Path(row["subset_relative"])).resolve()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
            raw = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed = extract_json(raw)
            truth_diagnosis = row["species"] if row["dataset"] == "ages" else row["class_name"]
            truth_stage = row["stage_normalized"] if row["dataset"] == "ages" else "not_applicable"
            diagnosis_correct = bool(
                parsed
                and normalize_label(parsed.get("diagnosis")) == normalize_label(truth_diagnosis)
            )
            stage_correct = bool(
                parsed and normalize_label(parsed.get("stage")) == normalize_label(truth_stage)
            )
            result = {
                "index": index,
                "dataset": row["dataset"],
                "image": row["subset_relative"],
                "truth_diagnosis": truth_diagnosis,
                "truth_stage": truth_stage,
                "raw": raw,
                "parsed": parsed,
                "diagnosis_correct": diagnosis_correct,
                "stage_correct": stage_correct,
                "seconds": seconds,
            }
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "index": index,
                        "dataset": row["dataset"],
                        "parsed": parsed is not None,
                        "diagnosis_correct": diagnosis_correct,
                        "stage_correct": stage_correct,
                        "seconds": round(seconds, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = summarize_results(
        results,
        model_path=model_path,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3,
    )
    (args.output_dir / "probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
