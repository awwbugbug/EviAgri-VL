import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from train_task9d import (
    adapter_hash_report,
    ensure_new_output,
    final_checkpoint_rationale,
    role_exposure_counts,
    run_directory,
    task9d_training_arguments,
    validate_finite_history,
)


def test_run_paths_are_variant_seed_isolated_and_output_is_immutable(tmp_path):
    assert run_directory(tmp_path, "A", 17) == tmp_path / "runs" / "A" / "seed_17"
    assert run_directory(tmp_path, "C", 43) == tmp_path / "runs" / "C" / "seed_43"
    output = run_directory(tmp_path, "A", 17)
    output.mkdir(parents=True)
    (output / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing"):
        ensure_new_output(output)


def test_training_arguments_use_fixed_budget_and_final_checkpoint_only():
    formal = task9d_training_arguments(seed=29, mode="formal")
    assert formal["max_steps"] == 192
    assert formal["gradient_accumulation_steps"] == 8
    assert formal["eval_steps"] == 64
    assert formal["save_steps"] == 192
    assert formal["save_total_limit"] == 1
    assert formal["load_best_model_at_end"] is False
    smoke = task9d_training_arguments(seed=17, mode="smoke")
    assert smoke["max_steps"] == 3
    assert smoke["save_strategy"] == "no"
    assert final_checkpoint_rationale(192) == {
        "selected_step": 192, "rule": "fixed_final_step_no_dev_selection", "early_stopping": False
    }


def test_nonfinite_history_fails_closed():
    validate_finite_history([{"loss": 1.2}, {"eval_loss": 1.1}])
    with pytest.raises(FloatingPointError, match="non-finite"):
        validate_finite_history([{"loss": float("nan")}])


def test_adapter_hash_and_exposure_report_are_exact(tmp_path):
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    report = adapter_hash_report(adapter)
    assert report["sha256"] == hashlib.sha256(b"adapter").hexdigest()
    assert report["bytes"] == 7
    assert role_exposure_counts([
        {"role": "positive"}, {"role": "semantic_negative"}, {"role": "positive"}
    ]) == {"positive": 2, "semantic_negative": 1}
