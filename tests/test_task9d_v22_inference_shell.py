from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_v22_inference_shell_requires_all_training_and_runs_six_paired_groups():
    text = (ROOT / "server/run_task9d_v22_inference.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "matrix_status.json" in text
    assert 'for arm in Control TaxMask' in text
    assert 'for seed in 17 29 43' in text
    assert "sha256sum -c completion.sha256" in text
    assert "run_task9d_inference.py" in text
    assert "evaluation_manifest.jsonl" in text
    assert "run_task9d_v22_evaluation.py" in text
    assert "test ! -e" in text
    assert "Base" not in text
    assert "shutdown" not in text
    assert "poweroff" not in text
    assert "task8" not in text.lower()
