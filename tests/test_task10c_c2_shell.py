from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_c2_shell_runs_frozen_matrix_and_stops_after_evaluation():
    text = (ROOT / "server/run_task10c_c2.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "for seed in 17 29 43" in text
    assert "for step in 8 16 32 64" in text
    assert "train_task10c_c2.py" in text
    assert "run_task10c_c2_inference.py" in text
    assert "score_task10c_c2_candidates.py" in text
    assert "evaluate_task10c_c2.py" in text
    assert "PASS_C1_ENGINEERING" in text
    assert "--repetitions 1000" in text
    assert "--bootstrap-seed 20260717" in text
    forbidden = ("task8", "official_test", "shutdown", "poweroff", "7b", "sam2", "task10d")
    assert not any(token in text.lower() for token in forbidden)


def test_c2_shell_verifies_every_completion_and_never_retries_or_overwrites():
    text = (ROOT / "server/run_task10c_c2.sh").read_text(encoding="utf-8")
    assert 'test ! -e "$ROOT"' in text
    assert "sha256sum -c completion.sha256" in text
    assert 'verify_completion "$ROOT/training/seed_$seed/checkpoints/step_$step_padded"' in text
    assert 'verify_completion "$ROOT/inference/smoke/seed_$seed/step_$step_padded"' in text
    assert 'verify_completion "$ROOT/inference/dev/seed_$seed"' in text
    assert 'verify_completion "$ROOT/candidates/seed_$seed"' in text
    assert "while " not in text
    assert "rm " not in text
    assert "TASK10C_C2_COMPLETED_STOP" in text
