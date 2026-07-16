"""Loss masking and loss-mass auditing for the frozen Task 9D v2.2 micro test."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F


TAXONOMY_FIELDS = ("pest_id", "pest_name")


def taxonomy_value_character_spans(target_text: str) -> list[tuple[str, int, int]]:
    """Return exact JSON-literal spans for positive taxonomy values only."""
    try:
        target = json.loads(target_text)
    except json.JSONDecodeError as exc:
        raise ValueError("assistant target is not valid JSON") from exc
    if target.get("evidence_present") is not True:
        return []
    diagnosis = target.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise ValueError("positive target lacks diagnosis object")
    spans: list[tuple[str, int, int]] = []
    for field in TAXONOMY_FIELDS:
        value = diagnosis.get(field)
        if value is None:
            raise ValueError(f"positive target has null {field}")
        literal = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*(?P<value>{re.escape(literal)})')
        matches = list(pattern.finditer(target_text))
        if len(matches) != 1:
            raise ValueError(f"expected one serialized {field} value, found {len(matches)}")
        start, end = matches[0].span("value")
        spans.append((field, start, end))
    return spans


def _tokenizer_values(tokenizer, text: str, *, offsets: bool = False) -> dict[str, Any]:
    values = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=offsets,
    )
    input_ids = values["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("tokenizer returned an unexpected batch")
        input_ids = input_ids[0]
    result: dict[str, Any] = {"input_ids": [int(item) for item in input_ids]}
    if offsets:
        mapping = values["offset_mapping"]
        if mapping and isinstance(mapping[0], list):
            mapping = mapping[0]
        result["offset_mapping"] = [(int(start), int(end)) for start, end in mapping]
    return result


def _unique_subsequence_start(values: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("assistant target tokenization is empty")
    starts = [
        start for start in range(len(values) - len(needle) + 1)
        if values[start:start + len(needle)] == needle
    ]
    if len(starts) != 1:
        raise ValueError(f"assistant target token sequence is not unique: {len(starts)} matches")
    return starts[0]


def apply_taxonomy_value_mask(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    target_text: str,
    *,
    slow_tokenizer,
    fast_tokenizer,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mask positive pest ID/name value tokens without changing example loss mass."""
    if labels.shape != input_ids.shape or labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("v2.2 taxonomy masking requires matching single-example tensors")
    slow = _tokenizer_values(slow_tokenizer, target_text)
    fast = _tokenizer_values(fast_tokenizer, target_text, offsets=True)
    if slow["input_ids"] != fast["input_ids"]:
        raise ValueError("fast/slow tokenizer IDs differ for assistant target")
    if len(fast["input_ids"]) != len(fast["offset_mapping"]):
        raise ValueError("fast tokenizer offset count differs from token count")
    full_ids = [int(item) for item in input_ids[0].tolist()]
    target_start = _unique_subsequence_start(full_ids, slow["input_ids"])
    spans = taxonomy_value_character_spans(target_text)
    before = int(labels[:, 1:].ne(-100).sum().item())
    masked = labels.clone()
    field_counts: dict[str, int] = {}
    positions: set[int] = set()
    for field, char_start, char_end in spans:
        relative = [
            index for index, (token_start, token_end) in enumerate(fast["offset_mapping"])
            if token_start < char_end and token_end > char_start
        ]
        if not relative:
            raise ValueError(f"no tokenizer offsets overlap {field} value")
        absolute = [target_start + index for index in relative]
        if any(masked[0, index].item() == -100 for index in absolute):
            raise ValueError(f"{field} value overlaps an already inactive token")
        positions.update(absolute)
        field_counts[field] = len(set(absolute))
    for position in sorted(positions):
        masked[0, position] = -100
    after = int(masked[:, 1:].ne(-100).sum().item())
    if spans and before - after != len(positions):
        raise ValueError("taxonomy mask active-token accounting mismatch")
    if after <= 0:
        raise ValueError("taxonomy mask removed every active target token")
    return masked, {
        "fast_slow_ids_equal": True,
        "target_token_count": len(slow["input_ids"]),
        "before_active_tokens": before,
        "after_active_tokens": after,
        "masked_token_count": len(positions),
        "masked_fields": [field for field, _, _ in spans],
        "masked_tokens_by_field": field_counts,
    }


def per_example_active_token_mean(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Causal-LM CE: active-token mean per example, then batch mean."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels shapes are incompatible")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shifted_labels.shape)
    active = shifted_labels.ne(-100)
    active_counts = active.sum(dim=1)
    if torch.any(active_counts == 0):
        raise ValueError("per-example reduction found an empty active-token mask")
    example_losses = (token_losses * active).sum(dim=1) / active_counts
    batch_size = labels.shape[0]
    weights = torch.full(
        (batch_size,), 1.0 / batch_size,
        dtype=example_losses.dtype,
        device=example_losses.device,
    )
    return (example_losses * weights).sum(), {
        "active_tokens": active_counts.detach(),
        "per_example_loss": example_losses.detach(),
        "per_example_loss_weight": weights.detach(),
    }


def _arm_loss_mass(rows: dict[str, dict[str, Any]], accumulation: int) -> dict[str, Any]:
    total_samples = sum(int(value["samples"]) for value in rows.values())
    if total_samples <= 0:
        raise ValueError("loss-mass audit has no samples")
    result: dict[str, Any] = {}
    for role, value in sorted(rows.items()):
        samples = int(value["samples"])
        tokens = [int(item) for item in value["active_tokens"]]
        if samples <= 0 or len(tokens) != samples or any(item <= 0 for item in tokens):
            raise ValueError(f"invalid loss-mass inputs for role {role}")
        result[role] = {
            "samples": samples,
            "active_tokens": {
                "min": min(tokens),
                "max": max(tokens),
                "mean": mean(tokens),
                "sum": sum(tokens),
            },
            "mean_example_loss_weight": 1.0,
            "mean_optimizer_window_weight": 1.0 / accumulation,
            "total_gradient_weight": samples / accumulation,
            "normalized_total_gradient_weight": samples / total_samples,
        }
    return result


def audit_loss_mass_equivalence(
    control: dict[str, dict[str, Any]],
    taxmask: dict[str, dict[str, Any]],
    *,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    """BLOCK unless Control and TaxMask retain identical role loss mass."""
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation must be positive")
    arms = {
        "Control": _arm_loss_mass(control, gradient_accumulation_steps),
        "TaxMask": _arm_loss_mass(taxmask, gradient_accumulation_steps),
    }
    reasons: list[str] = []
    if set(control) != set(taxmask):
        reasons.append("role sets differ between Control and TaxMask")
    for role in sorted(set(control) & set(taxmask)):
        if int(control[role]["samples"]) != int(taxmask[role]["samples"]):
            reasons.append(f"role sample counts differ for {role}")
            continue
        left = arms["Control"][role]["normalized_total_gradient_weight"]
        right = arms["TaxMask"][role]["normalized_total_gradient_weight"]
        if abs(left - right) > 1e-12:
            reasons.append(f"normalized total gradient weight differs for {role}")
        if arms["Control"][role]["mean_example_loss_weight"] != arms["TaxMask"][role]["mean_example_loss_weight"]:
            reasons.append(f"mean example loss weight differs for {role}")
    return {
        "reduction": "per_example_active_token_mean_then_batch_mean",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "arms": arms,
        "passed": not reasons,
        "block_reasons": reasons,
    }
