from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_task10b_formal_shell_requires_protocol_and_smoke_before_formal_run():
    text = (ROOT / "server/run_task10b_v2.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "sha256sum -c completion.sha256" in text
    assert "task10b_v2" in text
    assert 'PROTOCOL="$ROOT/protocol"' in text
    assert 'SMOKE="$ROOT/smoke_8_r3"' in text
    assert "test ! -e \"$FEATURE_OUT\"" in text
    assert "test ! -e \"$EVAL_OUT\"" in text
    assert "/root/miniconda3/envs/eviagri/bin/python" in text
    assert "/root/miniconda3/bin/python" in text
    assert "extract_task10b_features.py" in text
    assert "evaluate_task10b_probe.py" in text
    assert "--repetitions 1000" in text
    assert "--limit" not in text
    assert "shutdown" not in text
    assert "poweroff" not in text
    assert "task10c" not in text.lower()
    assert "task8" not in text.lower()
