"""Teacher-forced token-level PDM-H audit for Task 10A."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import torch

from task10_audit_common import (
    ensure_new_directory,
    family_bootstrap_delta,
    sha256_file,
    write_json_new,
)
from task9b_protocol import build_target


TOKEN_GROUPS = (
    "evidence_present",
    "taxonomy_value",
    "bbox_value",
    "refusal_uncertain",
    "other_assistant",
)
CONDITIONS = ("original", "blank", "shuffle")
CONTROL_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class AssistantTokenSpans:
    token_ids: list[int]
    groups: dict[str, list[int]]
    coverage: float


def hellinger_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Return Hellinger distance between categorical distributions represented by logits."""
    if not torch.isfinite(logits_p).all() or not torch.isfinite(logits_q).all():
        raise FloatingPointError("non-finite PDM-H logits")
    probability_p = torch.softmax(logits_p.float(), dim=-1)
    probability_q = torch.softmax(logits_q.float(), dim=-1)
    value = torch.sqrt(
        torch.clamp(
            ((torch.sqrt(probability_p) - torch.sqrt(probability_q)) ** 2).sum(-1),
            min=0.0,
        )
    ) / math.sqrt(2.0)
    if not torch.isfinite(value).all():
        raise FloatingPointError("non-finite PDM-H")
    return value


def _value_span(text: str, key: str) -> tuple[int, int]:
    marker = json.dumps(key, ensure_ascii=False)
    key_start = text.find(marker)
    if key_start < 0 or text.find(marker, key_start + len(marker)) >= 0:
        raise ValueError(f"JSON key must occur exactly once: {key}")
    colon = text.find(":", key_start + len(marker))
    if colon < 0:
        raise ValueError(f"JSON key lacks value: {key}")
    value_start = colon + 1
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    _, relative_end = json.JSONDecoder().raw_decode(text[value_start:])
    return value_start, value_start + relative_end


def _overlaps(offset: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = offset
    return start < end and any(start < span_end and end > span_start for span_start, span_end in spans)


def assistant_token_spans(tokenizer: Any, target_text: str) -> AssistantTokenSpans:
    """Map JSON value spans to target-token indices; field names stay in `other_assistant`."""
    parsed = json.loads(target_text)
    encoded = tokenizer(
        target_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(map(int, offset)) for offset in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets) or not token_ids:
        raise ValueError("tokenizer did not return aligned target offsets")

    value_spans: dict[str, list[tuple[int, int]]] = {
        "evidence_present": [_value_span(target_text, "evidence_present")],
        "bbox_value": [_value_span(target_text, "evidence_region")],
        "taxonomy_value": [
            _value_span(target_text, "pest_id"),
            _value_span(target_text, "pest_name"),
        ],
        "refusal_uncertain": [],
    }
    diagnosis = parsed.get("diagnosis") if isinstance(parsed, dict) else None
    if isinstance(diagnosis, dict) and diagnosis.get("status") in {"abstain", "uncertain"}:
        value_spans["refusal_uncertain"].append(_value_span(target_text, "status"))
    if parsed.get("reliability") == "insufficient_visual_evidence":
        value_spans["refusal_uncertain"].append(_value_span(target_text, "reliability"))

    groups = {name: [] for name in TOKEN_GROUPS}
    valid_offsets = 0
    ordered = ("evidence_present", "bbox_value", "taxonomy_value", "refusal_uncertain")
    for token_index, offset in enumerate(offsets):
        if offset[1] <= offset[0]:
            continue
        valid_offsets += 1
        destination = next(
            (name for name in ordered if _overlaps(offset, value_spans[name])),
            "other_assistant",
        )
        groups[destination].append(token_index)
    coverage = valid_offsets / len(token_ids)
    return AssistantTokenSpans(token_ids=token_ids, groups=groups, coverage=coverage)


def assistant_token_spans_compatible(
    model_input_tokenizer: Any,
    offset_tokenizer: Any,
    target_text: str,
) -> AssistantTokenSpans:
    """Use fast offsets only after proving exact IDs match model-input tokenization."""
    spans = assistant_token_spans(offset_tokenizer, target_text)
    input_ids = model_input_tokenizer(
        target_text,
        add_special_tokens=False,
    )["input_ids"]
    if list(map(int, input_ids)) != spans.token_ids:
        raise ValueError("fast-offset and model-input token IDs differ")
    return spans


def find_token_subsequence(sequence: Iterable[int], subsequence: Iterable[int]) -> int:
    sequence_values = list(map(int, sequence))
    target = list(map(int, subsequence))
    if not target:
        raise ValueError("target token subsequence cannot be empty")
    matches = [
        index
        for index in range(len(sequence_values) - len(target) + 1)
        if sequence_values[index:index + len(target)] == target
    ]
    if len(matches) != 1:
        raise ValueError(f"target token subsequence must occur exactly once, found {len(matches)}")
    return matches[0]


def _queried_name(row: dict[str, Any]) -> str:
    try:
        prompt = str(row["messages"][1]["content"][1]["text"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"invalid positive prompt for {row.get('id')}") from exc
    match = re.search(r"queried pest '([^']+)'", prompt)
    if not match or not match.group(1).strip():
        raise ValueError(f"cannot recover queried pest name for {row.get('id')}")
    return match.group(1).strip()


def canonical_positive_target(row: dict[str, Any]) -> str:
    if (
        str(row.get("role")) != "positive"
        or str(row.get("condition")) != "original"
        or str(row.get("prompt_view")) != "canonical"
    ):
        raise ValueError("canonical target requires a canonical original positive row")
    target = build_target(
        evidence_present=True,
        evidence_region=row.get("gt_bbox"),
        pest_id=int(row["query_class_id"]),
        pest_name=_queried_name(row),
    )
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def build_condition_messages(
    original_row: dict[str, Any],
    intervention_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if str(original_row.get("family_id")) != str(intervention_row.get("family_id")):
        raise ValueError("PDM intervention must belong to the same family")
    if str(intervention_row.get("condition")) not in CONDITIONS:
        raise ValueError("PDM intervention must be original, blank, or shuffle")
    target_text = canonical_positive_target(original_row)
    conditioned = copy.deepcopy(original_row["messages"][:2])
    replacement = copy.deepcopy(intervention_row["messages"][1]["content"][0])
    conditioned[1]["content"][0] = replacement
    conditioned.append({
        "role": "assistant",
        "content": [{"type": "text", "text": target_text}],
    })
    unconditioned = copy.deepcopy(conditioned)
    unconditioned[1]["content"] = [
        item for item in unconditioned[1]["content"] if item.get("type") != "image"
    ]
    return conditioned, unconditioned, target_text


def _validate_observations(
    records: list[dict[str, Any]],
    expected_families: int,
    seeds: tuple[int, ...],
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    families = sorted({str(record.get("family_id")) for record in records})
    if len(families) != expected_families:
        failures.append(f"expected_{expected_families}_families_got_{len(families)}")
    keys = [
        (int(record.get("seed")), str(record.get("family_id")), str(record.get("condition")))
        for record in records
    ]
    if len(keys) != len(set(keys)):
        failures.append("duplicate_seed_family_condition")
    expected = {
        (seed, family, condition)
        for seed in seeds
        for family in families
        for condition in CONDITIONS
    }
    if set(keys) != expected:
        failures.append("incomplete_seed_family_condition_matrix")
    if any(float(record.get("coverage", 0.0)) < 0.95 for record in records):
        failures.append("token_span_coverage_below_0.95")
    if any(not bool(record.get("finite")) for record in records):
        failures.append("non_finite_distribution_or_pdm")
    if any(float(record.get("normalization_error", math.inf)) > 1e-5 for record in records):
        failures.append("normalization_error_above_1e-5")
    for record in records:
        values = record.get("group_mean_pdm_h")
        if not isinstance(values, dict) or set(values) != set(TOKEN_GROUPS):
            failures.append("invalid_token_group_contract")
            break
        if not all(math.isfinite(float(value)) for value in values.values()):
            failures.append("non_finite_group_mean")
            break
    return sorted(set(failures)), families


def _bootstrap_group(
    values: dict[tuple[int, str, str], float],
    *,
    seeds: tuple[int, ...],
    families: list[str],
    intervention: str,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = []
    for family in families:
        original = mean(values[(seed, family, "original")] for seed in seeds)
        changed = mean(values[(seed, family, intervention)] for seed in seeds)
        rows.append((family, original, changed))
    return family_bootstrap_delta(rows, repetitions=repetitions, seed=bootstrap_seed)


def summarize_pdm_records(
    records: Iterable[dict[str, Any]],
    *,
    repetitions: int = 1000,
    bootstrap_seed: int = 20260717,
    expected_families: int = 32,
    seeds: tuple[int, ...] = CONTROL_SEEDS,
) -> dict[str, Any]:
    rows = list(records)
    failures, families = _validate_observations(rows, expected_families, seeds)
    values_by_group: dict[str, dict[tuple[int, str, str], float]] = {
        group: {} for group in TOKEN_GROUPS
    }
    for row in rows:
        key = (int(row["seed"]), str(row["family_id"]), str(row["condition"]))
        for group in TOKEN_GROUPS:
            values_by_group[group][key] = float(row["group_mean_pdm_h"][group])

    pooled: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    complete = not any(
        failure in failures
        for failure in ("duplicate_seed_family_condition", "incomplete_seed_family_condition_matrix")
    )
    if complete and families:
        for group in TOKEN_GROUPS:
            pooled[group] = {}
            for offset, intervention in enumerate(("blank", "shuffle")):
                pooled[group][f"original_minus_{intervention}"] = _bootstrap_group(
                    values_by_group[group],
                    seeds=seeds,
                    families=families,
                    intervention=intervention,
                    repetitions=repetitions,
                    bootstrap_seed=bootstrap_seed + offset,
                )
        for seed in seeds:
            seed_report: dict[str, Any] = {}
            for group in TOKEN_GROUPS:
                seed_report[group] = {}
                for offset, intervention in enumerate(("blank", "shuffle")):
                    paired = [
                        (
                            family,
                            values_by_group[group][(seed, family, "original")],
                            values_by_group[group][(seed, family, intervention)],
                        )
                        for family in families
                    ]
                    seed_report[group][f"original_minus_{intervention}"] = (
                        family_bootstrap_delta(
                            paired,
                            repetitions=repetitions,
                            seed=bootstrap_seed + seed * 10 + offset,
                        )
                    )
            per_seed[str(seed)] = seed_report

    quality_passed = not failures
    visual_pass = False
    if quality_passed:
        candidates = [
            pooled[group][f"original_minus_{intervention}"]
            for group in ("taxonomy_value", "evidence_present")
            for intervention in ("blank", "shuffle")
        ]
        visual_pass = any(item["estimate"] > 0 and item["low"] > 0 for item in candidates)
    return {
        "version": "task10a-pdm-h-report-v1",
        "quality_passed": quality_passed,
        "visual_dependency_passed": visual_pass,
        "quality_failures": failures,
        "family_count": len(families),
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "observation_count": len(rows),
        "bootstrap_repetitions": repetitions,
        "pooled": pooled,
        "per_seed": per_seed,
    }


def _move_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _encode_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    vision_info_fn: Callable,
    device: Any,
) -> Any:
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    has_image = any(
        item.get("type") == "image"
        for item in messages[1].get("content", [])
    )
    images, videos = vision_info_fn(messages) if has_image else (None, None)
    inputs = processor(
        text=[rendered],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    return _move_to_device(inputs, device)


def teacher_forced_group_pdm(
    model: Any,
    processor: Any,
    conditioned_messages: list[dict[str, Any]],
    unconditioned_messages: list[dict[str, Any]],
    spans: AssistantTokenSpans,
    *,
    vision_info_fn: Callable,
) -> dict[str, Any]:
    """Measure grouped next-token distribution change without returning logits."""
    conditioned = _encode_messages(
        processor,
        conditioned_messages,
        vision_info_fn,
        model.device,
    )
    unconditioned = _encode_messages(
        processor,
        unconditioned_messages,
        vision_info_fn,
        model.device,
    )
    conditioned_ids = conditioned["input_ids"][0].detach().cpu().tolist()
    unconditioned_ids = unconditioned["input_ids"][0].detach().cpu().tolist()
    conditioned_start = find_token_subsequence(conditioned_ids, spans.token_ids)
    unconditioned_start = find_token_subsequence(unconditioned_ids, spans.token_ids)
    if min(conditioned_start, unconditioned_start) < 1:
        raise ValueError("assistant target lacks a causal predecessor token")

    with torch.inference_mode():
        conditioned_logits = model(**conditioned).logits[
            0,
            conditioned_start - 1:conditioned_start + len(spans.token_ids) - 1,
        ]
        unconditioned_logits = model(**unconditioned).logits[
            0,
            unconditioned_start - 1:unconditioned_start + len(spans.token_ids) - 1,
        ]
        probability_conditioned = torch.softmax(conditioned_logits.float(), dim=-1)
        probability_unconditioned = torch.softmax(unconditioned_logits.float(), dim=-1)
        normalization_error = max(
            float((probability_conditioned.sum(-1) - 1.0).abs().max().item()),
            float((probability_unconditioned.sum(-1) - 1.0).abs().max().item()),
        )
        token_values = hellinger_from_logits(
            conditioned_logits,
            unconditioned_logits,
        ).detach().cpu()

    group_means: dict[str, float] = {}
    token_counts: dict[str, int] = {}
    for group in TOKEN_GROUPS:
        indices = spans.groups[group]
        token_counts[group] = len(indices)
        group_means[group] = float(token_values[indices].mean().item()) if indices else 0.0
    return {
        "coverage": spans.coverage,
        "finite": bool(torch.isfinite(token_values).all()),
        "normalization_error": normalization_error,
        "group_mean_pdm_h": group_means,
        "token_counts": token_counts,
        "target_token_count": len(spans.token_ids),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _family_conditions(manifest: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest:
        condition = str(row.get("condition"))
        if condition not in CONDITIONS or str(row.get("prompt_view")) != "canonical":
            continue
        family = str(row.get("family_id"))
        if condition in grouped[family]:
            raise ValueError(f"duplicate PDM condition for {family}: {condition}")
        grouped[family][condition] = row
    valid = {
        family: conditions
        for family, conditions in grouped.items()
        if set(conditions) == set(CONDITIONS)
        and str(conditions["original"].get("role")) == "positive"
    }
    if len(valid) != 32:
        raise ValueError(f"PDM requires exactly 32 complete families, got {len(valid)}")
    return dict(sorted(valid.items()))


def _adapter_weight_path(adapter_dir: Path) -> Path:
    candidates = [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"expected one adapter weight file in {adapter_dir}")
    return existing[0]


def validate_pdm_run_configuration(
    adapter_paths: dict[int, Path],
    *,
    smoke: bool,
    family_limit: int | None,
) -> dict[str, Any]:
    seeds = tuple(sorted(adapter_paths))
    if smoke:
        if seeds != (29,):
            raise ValueError("PDM smoke is restricted to Control seed 29")
        if family_limit != 1:
            raise ValueError("PDM smoke must use exactly one family")
        return {
            "seeds": seeds,
            "family_limit": 1,
            "scientific_decision_valid": False,
        }
    if seeds != CONTROL_SEEDS:
        raise ValueError("formal PDM requires Control seeds 17, 29, and 43")
    if family_limit is not None:
        raise ValueError("formal PDM cannot limit families")
    return {
        "seeds": seeds,
        "family_limit": None,
        "scientific_decision_valid": True,
    }


def run_pdm_audit(
    *,
    model_path: Path,
    adapter_paths: dict[int, Path],
    manifest_path: Path,
    output_dir: Path,
    repetitions: int = 1000,
    smoke: bool = False,
    family_limit: int | None = None,
) -> dict[str, Any]:
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    configuration = validate_pdm_run_configuration(
        adapter_paths,
        smoke=smoke,
        family_limit=family_limit,
    )
    seeds = configuration["seeds"]
    ensure_new_directory(output_dir)
    manifest = _read_jsonl(manifest_path)
    families = _family_conditions(manifest)
    if configuration["family_limit"] is not None:
        families = dict(list(families.items())[:configuration["family_limit"]])
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=200704,
        max_pixels=401408,
        use_fast=False,
        local_files_only=True,
    )
    offset_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
    )
    if not bool(getattr(offset_tokenizer, "is_fast", False)):
        raise ValueError("PDM offset tokenizer must be a fast tokenizer")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    first_seed = seeds[0]
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_paths[first_seed]),
        adapter_name=f"control_{first_seed}",
        is_trainable=False,
    )
    for seed in seeds[1:]:
        model.load_adapter(str(adapter_paths[seed]), adapter_name=f"control_{seed}")
    model.eval()

    adapter_hashes = {
        seed: sha256_file(_adapter_weight_path(path))
        for seed, path in adapter_paths.items()
    }
    observations = []
    completed = 0
    expected_observations = len(seeds) * len(families) * len(CONDITIONS)
    for seed in seeds:
        model.set_adapter(f"control_{seed}")
        for family, conditions in families.items():
            original = conditions["original"]
            target_text = canonical_positive_target(original)
            spans = assistant_token_spans_compatible(
                processor.tokenizer,
                offset_tokenizer,
                target_text,
            )
            for condition in CONDITIONS:
                conditioned, unconditioned, same_target = build_condition_messages(
                    original, conditions[condition]
                )
                if same_target != target_text:
                    raise AssertionError("PDM target changed across interventions")
                measurement = teacher_forced_group_pdm(
                    model,
                    processor,
                    conditioned,
                    unconditioned,
                    spans,
                    vision_info_fn=process_vision_info,
                )
                completed += 1
                observations.append({
                    "seed": seed,
                    "family_id": family,
                    "condition": condition,
                    "adapter_sha256": adapter_hashes[seed],
                    "target_sha256": __import__("hashlib").sha256(
                        target_text.encode("utf-8")
                    ).hexdigest(),
                    **measurement,
                })
                if completed % 8 == 0:
                    write_json_new(
                        output_dir / f"progress_{completed:03d}.json",
                        {
                            "state": "running",
                            "completed": completed,
                            "expected": expected_observations,
                        },
                    )

    report = summarize_pdm_records(
        observations,
        repetitions=repetitions,
        expected_families=len(families),
        seeds=seeds,
    )
    report["mode"] = "smoke" if smoke else "formal"
    report["scientific_decision_valid"] = configuration["scientific_decision_valid"]
    if smoke:
        report["visual_dependency_passed"] = None
    report["inputs"] = {
        "manifest_sha256": sha256_file(manifest_path),
        "adapter_sha256": {str(seed): digest for seed, digest in adapter_hashes.items()},
        "model_path": str(model_path),
        "tokenizer_contract": {
            "model_input_tokenizer": type(processor.tokenizer).__name__,
            "offset_tokenizer": type(offset_tokenizer).__name__,
            "exact_target_token_ids_required": True,
        },
    }
    observations_path = output_dir / "pdm_observations.jsonl"
    report_path = output_dir / "pdm_token_report.json"
    _write_jsonl_new(observations_path, observations)
    write_json_new(report_path, report)
    completion = output_dir / "completion.sha256"
    completion.write_text(
        f"{sha256_file(observations_path)}  {observations_path.name}\n"
        f"{sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-17", type=Path)
    parser.add_argument("--adapter-29", type=Path)
    parser.add_argument("--adapter-43", type=Path)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--family-limit", type=int)
    args = parser.parse_args()
    adapters = {
        seed: path
        for seed, path in (
            (17, args.adapter_17),
            (29, args.adapter_29),
            (43, args.adapter_43),
        )
        if path is not None
    }
    report = run_pdm_audit(
        model_path=args.model_path,
        adapter_paths=adapters,
        manifest_path=args.manifest,
        output_dir=args.output,
        repetitions=args.repetitions,
        smoke=args.smoke,
        family_limit=args.family_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
