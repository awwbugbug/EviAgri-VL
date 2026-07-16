"""Score the frozen 16-class Task 10C candidates by conditional log-likelihood.

This is a diagnostic readout, not a generation shortcut: every class receives the
same compact canonical JSON continuation and is ranked by its mean answer-token
log probability under the image-and-prompt context.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from run_task10c_c2_inference import (
    build_c2_conditions,
    model_identity,
    verify_c2_adapter,
)
from run_task10c_smoke_inference import inference_messages
from task10c_c2_contract import verify_c2_protocol
from task10c_contract import CLASS_IDS, canonical_pest_id
from train_task10c_c2 import _sha256, _write_json


CANONICAL_IDS = tuple(canonical_pest_id(value) for value in CLASS_IDS)
MAX_PROJECTED_SECONDS = 30 * 60


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def candidate_targets() -> list[str]:
    return [
        json.dumps({"pest_id": pest_id}, ensure_ascii=False, separators=(",", ":"))
        for pest_id in CANONICAL_IDS
    ]


def mean_active_token_logprob(
    token_log_probs: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if token_log_probs.shape != active_mask.shape:
        raise ValueError("token log probabilities and active mask shape mismatch")
    if active_mask.dtype != torch.bool:
        active_mask = active_mask.bool()
    counts = active_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("no active answer tokens")
    if not bool(torch.isfinite(token_log_probs[active_mask]).all()):
        raise ValueError("non-finite active answer-token log probability")
    return (token_log_probs * active_mask).sum(dim=1) / counts


def rank_candidate_scores(scores: dict[str, float], *, truth: str) -> dict[str, Any]:
    if set(scores) != set(CANONICAL_IDS):
        raise ValueError("candidate set does not match the frozen 16 classes")
    if truth not in CANONICAL_IDS:
        raise ValueError("truth is outside the frozen candidate set")
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("candidate scores must be finite")
    ranking = sorted(CANONICAL_IDS, key=lambda pest_id: (-float(scores[pest_id]), pest_id))
    truth_rank = ranking.index(truth) + 1
    return {
        "prediction": ranking[0],
        "ranking": ranking,
        "truth_rank": truth_rank,
        "top1_correct": truth_rank <= 1,
        "top3_correct": truth_rank <= 3,
        "top5_correct": truth_rank <= 5,
    }


def resource_preflight_decision(
    *,
    elapsed_seconds: float,
    rows: int,
    peak_vram_bytes: int,
) -> dict[str, Any]:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("resource preflight elapsed time must be finite and positive")
    if rows <= 0 or peak_vram_bytes < 0:
        raise ValueError("resource preflight counts must be non-negative")
    projected = elapsed_seconds * rows
    passed = projected <= MAX_PROJECTED_SECONDS
    return {
        "passed": passed,
        "measured_rows": 1,
        "target_rows": rows,
        "measured_seconds_per_row": elapsed_seconds,
        "projected_seconds": projected,
        "peak_vram_bytes": int(peak_vram_bytes),
        "reason": (
            "projected runtime is within the frozen 30 minutes limit"
            if passed else
            "projected runtime exceeds the frozen 30 minutes limit"
        ),
    }


def validate_candidate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("id", "")) for row in rows]
    if len(rows) != 160 or len(set(ids)) != 160:
        raise ValueError("candidate results require 160 unique source-prompt rows")
    sources: dict[str, set[str]] = {}
    for row in rows:
        source = str(row.get("source_image_sha256", ""))
        prompt = str(row.get("prompt_variant", ""))
        sources.setdefault(source, set()).add(prompt)
        scores = row.get("scores", {})
        if not isinstance(scores, dict) or set(scores) != set(CANONICAL_IDS):
            raise ValueError("candidate result candidate set mismatch")
        if not all(math.isfinite(float(value)) for value in scores.values()):
            raise ValueError("candidate result contains non-finite scores")
        truth = str(row.get("truth", ""))
        if truth not in CANONICAL_IDS:
            raise ValueError("candidate result truth is outside candidate set")
        rank = int(row.get("truth_rank", 0))
        if not 1 <= rank <= len(CANONICAL_IDS):
            raise ValueError("candidate result truth rank is invalid")
    if len(sources) != 80 or any(prompts != {"train", "unseen"} for prompts in sources.values()):
        raise ValueError("candidate results require train and unseen prompts for 80 sources")
    return {"passed": True, "rows": 160, "sources": 80}


def _score_candidate_batch(
    *,
    model: Any,
    processor: Any,
    process_vision_info: Any,
    messages: list[dict[str, Any]],
    targets: list[str],
    device: torch.device,
) -> list[float]:
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    prompt_values: dict[str, Any] = {
        "text": [prefix], "padding": True, "return_tensors": "pt",
    }
    if images:
        prompt_values["images"] = images
    if videos:
        prompt_values["videos"] = videos
    prompt_inputs = processor(**prompt_values)
    prompt_length = int(prompt_inputs.attention_mask[0].sum().item())
    prefix_ids = prompt_inputs.input_ids[0, :prompt_length]

    values: dict[str, Any] = {
        "text": [prefix + target for target in targets],
        "padding": True,
        "return_tensors": "pt",
    }
    if images:
        values["images"] = images * len(targets)
    if videos:
        values["videos"] = videos * len(targets)
    inputs = processor(**values)
    for index in range(len(targets)):
        if not torch.equal(inputs.input_ids[index, :prompt_length], prefix_ids):
            raise ValueError("candidate continuation does not preserve the frozen prompt prefix")
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        shifted_logits = outputs.logits[:, :-1, :].float()
        shifted_labels = inputs.input_ids[:, 1:]
        token_log_probs = -F.cross_entropy(
            shifted_logits.transpose(1, 2), shifted_labels, reduction="none",
        )
    positions = torch.arange(1, inputs.input_ids.shape[1], device=device).unsqueeze(0)
    active = (positions >= prompt_length) & inputs.attention_mask[:, 1:].bool()
    return mean_active_token_logprob(token_log_probs, active).detach().cpu().tolist()


def _score_row(
    *,
    row: dict[str, Any],
    model: Any,
    processor: Any,
    process_vision_info: Any,
    device: torch.device,
    candidate_batch_size: int = 4,
) -> dict[str, Any]:
    targets = candidate_targets()
    values: list[float] = []
    messages = inference_messages(row)
    for start in range(0, len(targets), candidate_batch_size):
        values.extend(_score_candidate_batch(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            messages=messages,
            targets=targets[start:start + candidate_batch_size],
            device=device,
        ))
    scores = dict(zip(CANONICAL_IDS, (float(value) for value in values), strict=True))
    truth = canonical_pest_id(int(row["class_id"]))
    ranked = rank_candidate_scores(scores, truth=truth)
    return {
        "id": str(row["id"]),
        "source_image_id": str(row["source_image_id"]),
        "source_image_sha256": str(row["source_image_sha256"]),
        "prompt_variant": str(row["prompt_variant"]),
        "class_band": str(row["class_band"]),
        "truth": truth,
        "scores": scores,
        **ranked,
    }


def run_candidate_scoring(
    *,
    protocol_root: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    model_kind: str,
    seed: int | None = None,
    checkpoint_step: int | None = None,
    adapter_root: str | Path | None = None,
) -> dict[str, Any]:
    import peft
    import transformers
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    protocol_root, model_path, output = Path(protocol_root), Path(model_path), Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C2 candidate scoring: {output}")
    gate = verify_c2_protocol(protocol_root)
    config = json.loads((protocol_root / "config.snapshot.json").read_text(encoding="utf-8"))
    if Path(config["model_path"]).resolve() != model_path.resolve():
        raise ValueError("candidate scoring model path differs from signed protocol")
    identity = model_identity(
        model_kind=model_kind, seed=seed, checkpoint_step=checkpoint_step,
    )
    adapter_path = None
    checkpoint = None
    if model_kind == "adapter":
        if adapter_root is None or checkpoint_step != 64:
            raise ValueError("candidate scoring is frozen to a verified step-64 adapter")
        adapter_path = Path(adapter_root)
        checkpoint = verify_c2_adapter(adapter_path, seed=int(seed), step=64)
    elif adapter_root is not None:
        raise ValueError("Base candidate scoring cannot include an adapter")

    source = _read_jsonl(protocol_root / "dev.jsonl")
    all_conditions = build_c2_conditions(source, split="dev")
    rows = [row for row in all_conditions if row["image_present"]]
    if len(rows) != 160:
        raise ValueError("candidate scoring requires exactly 160 image-prompt rows")
    output.mkdir(parents=True)
    _write_json(output / "status.json", {
        "state": "running", "stage": "resource_preflight", "completed": 0,
        "expected": 160, **identity,
    })
    started = time.time()
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
            use_fast=False, local_files_only=True,
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, quantization_config=quantization,
            torch_dtype=torch.bfloat16, device_map={"": 0},
            low_cpu_mem_usage=True, local_files_only=True,
        )
        model = base if model_kind == "base" else PeftModel.from_pretrained(
            base, str(adapter_path / "adapter"), is_trainable=False,
        )
        model.eval()
        device = next(model.parameters()).device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        first_started = time.time()
        results = [_score_row(
            row=rows[0], model=model, processor=processor,
            process_vision_info=process_vision_info, device=device,
        )]
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        preflight = resource_preflight_decision(
            elapsed_seconds=time.time() - first_started,
            rows=len(rows), peak_vram_bytes=peak,
        )
        _write_json(output / "resource_preflight.json", preflight)
        if not preflight["passed"]:
            raise RuntimeError(preflight["reason"])
        _write_json(output / "status.json", {
            "state": "running", "stage": "candidate_scoring", "completed": 1,
            "expected": 160, "resource_preflight": preflight, **identity,
        })
        for index, row in enumerate(rows[1:], start=2):
            results.append(_score_row(
                row=row, model=model, processor=processor,
                process_vision_info=process_vision_info, device=device,
            ))
            _write_json(output / "status.json", {
                "state": "running", "stage": "candidate_scoring", "completed": index,
                "expected": 160, "resource_preflight": preflight, **identity,
            })
        validation = validate_candidate_results(results)
        _write_jsonl(output / "candidate_scores.jsonl", results)
        summary = {
            "completed": True, **identity,
            "rows": len(results), "sources": validation["sources"],
            "top1_accuracy": sum(row["top1_correct"] for row in results) / len(results),
            "top3_accuracy": sum(row["top3_correct"] for row in results) / len(results),
            "top5_accuracy": sum(row["top5_correct"] for row in results) / len(results),
            "protocol_manifest_sha256": gate["manifest_sha256"],
            "adapter_sha256": None if checkpoint is None else checkpoint["adapter"]["sha256"],
            "resource_preflight": preflight,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "elapsed_seconds": time.time() - started,
            "versions": {
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "peft": peft.__version__,
            },
        }
        _write_json(output / "run_summary.json", summary)
        names = ["candidate_scores.jsonl", "resource_preflight.json", "run_summary.json"]
        (output / "completion.sha256").write_text(
            "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
            encoding="utf-8",
        )
        _write_json(output / "status.json", {"state": "completed", **summary})
        return summary
    except Exception as exc:
        failure = {
            "state": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), **identity,
        }
        _write_json(output / "failure.json", failure)
        _write_json(output / "status.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-kind", choices=("base", "adapter"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--adapter-root")
    args = parser.parse_args()
    run_candidate_scoring(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
