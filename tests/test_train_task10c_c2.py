import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task10c_contract import EXPECTED_MANIFEST_SHA256
from train_task10c_c2 import (
    C2CheckpointCallback,
    save_c2_checkpoint,
    training_directory,
    validate_c2_run_summary,
)


SAFE_TRAINABLES = [
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
]


class FakeAdapterModel:
    def __init__(self, payload: bytes):
        self.payload = payload

    def save_pretrained(self, path, safe_serialization=True):
        assert safe_serialization is True
        path = Path(path)
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(self.payload)
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")


class FakeState:
    def __init__(self, step: int):
        self.global_step = step
        self.log_history = [
            {"step": value, "loss": 2.0 - value / 100, "grad_norm": 1.0}
            for value in range(1, step + 1)
        ]

    def save_to_json(self, path):
        Path(path).write_text(
            json.dumps({"global_step": self.global_step, "log_history": self.log_history}),
            encoding="utf-8",
        )


def _summary(steps=(8, 16, 32, 64)) -> dict:
    return {
        "completed": True,
        "seed": 17,
        "optimizer_steps": 64,
        "actual_exposures": 512,
        "training_rows": 64,
        "continued_from_c1": False,
        "loss_reduction": "per_example_active_token_mean_then_batch_mean",
        "log_history": [{"step": 64, "loss": 1.0, "grad_norm": 0.5}],
        "trainable_parameters": SAFE_TRAINABLES,
        "checkpoints": {
            str(step): {
                "optimizer_steps": step,
                "actual_exposures": step * 8,
                "adapter": {"sha256": f"{step:064x}"[-64:], "bytes": 10},
            }
            for step in steps
        },
    }


def test_training_directory_is_seed_scoped_and_frozen(tmp_path):
    assert training_directory(tmp_path, 17) == tmp_path / "training" / "seed_17"
    with pytest.raises(ValueError, match="frozen seed"):
        training_directory(tmp_path, 31)


def test_checkpoint_writer_saves_exact_step_hash_and_refuses_overwrite(tmp_path):
    result = save_c2_checkpoint(
        model=FakeAdapterModel(b"adapter-step-8"),
        state=FakeState(8),
        training_root=tmp_path,
        seed=17,
        step=8,
        trainable_parameters=SAFE_TRAINABLES,
        protocol_manifest_sha256=EXPECTED_MANIFEST_SHA256,
    )

    checkpoint = tmp_path / "checkpoints" / "step_008"
    assert result["optimizer_steps"] == 8
    assert result["actual_exposures"] == 64
    assert len(result["adapter"]["sha256"]) == 64
    assert (checkpoint / "completion.sha256").is_file()
    assert not (tmp_path / "checkpoints" / "step_008.tmp").exists()
    with pytest.raises(FileExistsError):
        save_c2_checkpoint(
            model=FakeAdapterModel(b"again"),
            state=FakeState(8),
            training_root=tmp_path,
            seed=17,
            step=8,
            trainable_parameters=SAFE_TRAINABLES,
            protocol_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        )


def test_checkpoint_callback_saves_only_frozen_steps_once(tmp_path):
    callback = C2CheckpointCallback(
        tmp_path,
        seed=17,
        trainable_parameters=SAFE_TRAINABLES,
        protocol_manifest_sha256=EXPECTED_MANIFEST_SHA256,
    )
    model = FakeAdapterModel(b"adapter")
    for step in (1, 8, 8, 16):
        callback.on_step_end(None, FakeState(step), None, model=model)
    assert set(callback.saved) == {8, 16}
    assert sorted(path.name for path in (tmp_path / "checkpoints").iterdir()) == [
        "step_008", "step_016"
    ]


def test_run_summary_requires_exact_final_counts_and_all_checkpoints():
    assert validate_c2_run_summary(_summary())["passed"] is True

    missing = _summary(steps=(8, 16, 64))
    with pytest.raises(ValueError, match="checkpoint set"):
        validate_c2_run_summary(missing)

    continued = _summary()
    continued["continued_from_c1"] = True
    with pytest.raises(ValueError, match="continued"):
        validate_c2_run_summary(continued)

    wrong_rows = _summary()
    wrong_rows["training_rows"] = 192
    with pytest.raises(ValueError, match="64-row"):
        validate_c2_run_summary(wrong_rows)
