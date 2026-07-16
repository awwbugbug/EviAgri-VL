import subprocess
from pathlib import Path


def test_shell_runs_exact_smoke_matrix_and_cannot_start_c2():
    path = Path("server/run_task10c_c0_c1.sh")
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "for seed in 17 29 43" in text
    assert "task10c_contract.py" in text
    assert "train_task10c_smoke.py" in text
    assert "run_task10c_smoke_inference.py" in text
    assert "evaluate_task10c_smoke.py" in text
    assert text.count("sha256sum -c completion.sha256") >= 4
    assert "test ! -e \"$ROOT\"" in text
    assert "set -euo pipefail" in text
    for forbidden in (
        "task8",
        "official_test",
        "max_steps 64",
        "task10d",
        "shutdown",
        "poweroff",
        "train_task10c_formal",
    ):
        assert forbidden not in lowered


def test_shell_is_executable():
    entry = subprocess.run(
        ["git", "ls-files", "--stage", "server/run_task10c_c0_c1.sh"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert entry.startswith("100755 ")
