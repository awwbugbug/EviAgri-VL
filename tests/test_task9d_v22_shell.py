from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_v22_matrix_shell_preserves_gate_and_frozen_six_runs():
    text = (ROOT / "server/run_task9d_v22_matrix.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "sha256sum -c completion.sha256" in text
    assert "loss_reduction_audit.json" in text
    assert 'for arm in Control TaxMask' in text
    assert 'for seed in 17 29 43' in text
    assert "test ! -e" in text
    assert "train_task9d_v22.py" in text
    assert "task9d_v22_micro/2026-07-16" in text
    assert "task9d/2026-07-16/runs/B/seed_29/config.snapshot.json" in text
    assert "shutdown" not in text
    assert "poweroff" not in text
    assert "task8" not in text.lower()
    assert "run_task9d_inference" not in text
