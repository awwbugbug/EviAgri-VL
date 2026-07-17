import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from run_task10c_c2_inference import (
    CONDITIONS,
    build_c2_conditions,
    model_identity,
    verify_c2_adapter,
    verify_prediction_ids,
)
from task10c_contract import CLASS_IDS, SYSTEM_PROMPT, TRAIN_PROMPT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(per_class: int) -> list[dict]:
    rows = []
    for class_id in CLASS_IDS:
        for index in range(per_class):
            rows.append({
                "id": f"row-{class_id:03d}-{index}",
                "class_id": class_id,
                "class_band": "head" if class_id in {16, 22, 24, 45, 50, 101} else (
                    "medium" if class_id in {10, 68, 71, 87, 99} else "tail"
                ),
                "source_image_id": f"IP{class_id:03d}{index:06d}",
                "source_image_sha256": f"{class_id:03d}{index:061d}"[-64:],
                "near_duplicate_component_id": f"component-{class_id}-{index}",
                "model": {"messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": f"/opaque/image-{class_id}-{index}.jpg"},
                        {"type": "text", "text": TRAIN_PROMPT},
                    ]},
                ]},
            })
    return rows


def _signed_checkpoint(tmp_path: Path, *, seed: int = 17, step: int = 16) -> Path:
    root = tmp_path / f"step_{step:03d}"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    model = adapter / "adapter_model.safetensors"
    model.write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    report = {"path": str(model), "sha256": _sha256(model), "bytes": model.stat().st_size}
    summary = {
        "completed": True,
        "seed": seed,
        "optimizer_steps": step,
        "actual_exposures": step * 8,
        "loss_reduction": "per_example_active_token_mean_then_batch_mean",
        "log_history": [{"loss": 1.0, "grad_norm": 0.5, "step": step}],
        "trainable_parameters": [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
        ],
        "adapter": report,
        "authorize_reuse": False,
    }
    (root / "checkpoint_summary.json").write_text(
        json.dumps(summary) + "\n", encoding="utf-8"
    )
    (root / "adapter.sha256.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    (root / "status.json").write_text('{"state":"completed"}\n', encoding="utf-8")
    names = [
        "adapter/adapter_model.safetensors", "adapter/adapter_config.json",
        "checkpoint_summary.json", "adapter.sha256.json", "status.json",
    ]
    (root / "completion.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    return root


def test_c2_conditions_are_exact_for_smoke_and_full_dev():
    smoke = build_c2_conditions(_rows(1), split="smoke_dev")
    full = build_c2_conditions(_rows(5), split="dev")

    assert len(smoke) == 64
    assert len(full) == 320
    assert Counter(row["condition"] for row in full) == {name: 80 for name in CONDITIONS}
    assert len({row["id"] for row in full}) == 320
    for row in full:
        prompt = json.dumps(row["messages"], ensure_ascii=False)
        assert row["source_image_id"] not in prompt
        assert row["source_image_sha256"] not in prompt


def test_c2_conditions_reject_wrong_split_quota():
    with pytest.raises(ValueError, match="class quota"):
        build_c2_conditions(_rows(1), split="dev")
    with pytest.raises(ValueError, match="split"):
        build_c2_conditions(_rows(1), split="test")


def test_model_identity_is_unambiguous():
    assert model_identity(model_kind="base") == {
        "model_id": "D0_base", "model_kind": "base", "seed": None, "checkpoint_step": 0,
    }
    assert model_identity(model_kind="adapter", seed=29, checkpoint_step=32) == {
        "model_id": "D1_seed_29_step_032", "model_kind": "adapter",
        "seed": 29, "checkpoint_step": 32,
    }
    with pytest.raises(ValueError, match="adapter identity"):
        model_identity(model_kind="adapter", seed=None, checkpoint_step=32)


def test_adapter_gate_requires_declared_seed_step_and_hash(tmp_path):
    checkpoint = _signed_checkpoint(tmp_path, seed=17, step=16)
    assert verify_c2_adapter(checkpoint, seed=17, step=16)["optimizer_steps"] == 16
    with pytest.raises(ValueError, match="checkpoint step"):
        verify_c2_adapter(checkpoint, seed=17, step=32)
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="completion SHA256"):
        verify_c2_adapter(checkpoint, seed=17, step=16)


def test_prediction_ids_must_match_manifest_exactly():
    rows = build_c2_conditions(_rows(1), split="smoke_dev")
    predictions = [{"id": row["id"]} for row in rows]
    verify_prediction_ids(rows, predictions)
    with pytest.raises(ValueError, match="prediction ID"):
        verify_prediction_ids(rows, predictions[:-1])
