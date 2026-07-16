"""Pre-training loss-reduction and gradient-mass gate for Task 9D v2.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from task9d_data import Task9dDataset
from task9d_v22_loss import audit_loss_mass_equivalence
from task9d_v22_training import REDUCTION_CONTRACT, V22LossCollator


EXPECTED_ROLES = {"positive", "semantic_negative", "visual_counterfactual"}


def summarize_loss_reduction_audit(
    observations: list[dict[str, Any]],
    *,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    control: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": 0, "active_tokens": []})
    taxmask: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": 0, "active_tokens": []})
    reasons: list[str] = []
    for row in observations:
        role = str(row["role"])
        control[role]["samples"] += 1
        taxmask[role]["samples"] += 1
        control[role]["active_tokens"].append(int(row["control_active_tokens"]))
        taxmask[role]["active_tokens"].append(int(row["taxmask_active_tokens"]))
        if not row.get("inputs_equal"):
            reasons.append(f"model inputs differ for {row['id']}")
        if role == "positive":
            if row.get("masked_fields") != ["pest_id", "pest_name"]:
                reasons.append(f"positive taxonomy mask missing for {row['id']}")
            if int(row["taxmask_active_tokens"]) >= int(row["control_active_tokens"]):
                reasons.append(f"positive active-token count did not decrease for {row['id']}")
        else:
            if row.get("null_labels_equal") is not True:
                reasons.append(f"null labels changed for {row['id']}")
            if int(row["taxmask_active_tokens"]) != int(row["control_active_tokens"]):
                reasons.append(f"null active-token count changed for {row['id']}")
    if set(control) != EXPECTED_ROLES:
        reasons.append(f"unexpected role set: {sorted(control)}")
    mass = audit_loss_mass_equivalence(
        dict(control), dict(taxmask),
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    reasons.extend(mass["block_reasons"])
    return {
        **mass,
        "version": "task9d-v22-loss-reduction-audit-v1",
        "reduction": REDUCTION_CONTRACT,
        "observations": len(observations),
        "invariants": {
            "all_inputs_equal": all(bool(row.get("inputs_equal")) for row in observations),
            "all_null_labels_equal": all(
                row.get("null_labels_equal") is True
                for row in observations if row["role"] != "positive"
            ),
            "all_positive_masks_exact": all(
                row.get("masked_fields") == ["pest_id", "pest_name"]
                for row in observations if row["role"] == "positive"
            ),
        },
        "passed": not reasons,
        "block_reasons": reasons,
        "task8_locked_set_read": False,
        "training_started": False,
    }


def _tensors_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    keys = (set(left) | set(right)) - {"labels"}
    return all(
        key in left and key in right
        and isinstance(left[key], torch.Tensor) and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in keys
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def run_audit(
    *,
    schedule_path: Path,
    image_root: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    from transformers import AutoProcessor, AutoTokenizer

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite v2.2 loss audit: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = Task9dDataset(schedule_path, image_root)
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=200704, max_pixels=401408,
        use_fast=False, local_files_only=True,
    )
    fast = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    control = V22LossCollator(processor, 1024, arm="Control", fast_tokenizer=fast)
    taxmask = V22LossCollator(processor, 1024, arm="TaxMask", fast_tokenizer=fast)
    observations: list[dict[str, Any]] = []
    for index, envelope in enumerate(dataset.records):
        record = dataset[index]
        control_batch = control([record])
        control_audit = dict(control.last_audit or {})
        taxmask_batch = taxmask([record])
        taxmask_audit = dict(taxmask.last_audit or {})
        role = str(envelope["role"])
        observations.append({
            "id": str(envelope["id"]),
            "family_id": str(envelope["family_id"]),
            "role": role,
            "inputs_equal": _tensors_equal(control_batch, taxmask_batch),
            "control_active_tokens": int(control_audit["active_tokens"]),
            "taxmask_active_tokens": int(taxmask_audit["active_tokens"]),
            "masked_token_count": int(taxmask_audit["masked_token_count"]),
            "masked_fields": list(taxmask_audit["masked_fields"]),
            "fast_slow_ids_equal": bool(taxmask_audit["fast_slow_ids_equal"]),
            "null_labels_equal": None if role == "positive" else bool(
                torch.equal(control_batch["labels"], taxmask_batch["labels"])
            ),
        })
    report = summarize_loss_reduction_audit(
        observations, gradient_accumulation_steps=8,
    )
    report["input_sha256"] = {
        "train_schedule": _sha256(schedule_path),
        "protocol_report": _sha256(schedule_path.parent / "protocol_report.json"),
    }
    _write_json(output_root / "loss_reduction_audit.json", report)
    _write_jsonl(output_root / "observations.jsonl", observations)
    names = ["loss_reduction_audit.json", "observations.jsonl"]
    (output_root / "completion.sha256").write_text(
        "".join(f"{_sha256(output_root / name)}  {name}\n" for name in names), encoding="utf-8"
    )
    _write_json(
        output_root / ("status.json" if report["passed"] else "blocked.json"),
        {"state": "completed" if report["passed"] else "blocked", "passed": report["passed"]},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        schedule_path=args.schedule, image_root=args.image_root,
        model_path=args.model_path, output_root=args.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
