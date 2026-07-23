from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_task11a_shell_is_fail_closed_and_reuses_signed_features():
    text = (ROOT / "server/run_task11a_confidence_router.sh").read_text(
        encoding="utf-8"
    )
    assert "set -euo pipefail" in text
    assert "test ! -e \"$ROOT\"" in text
    assert "sha256sum -c completion.sha256" in text
    assert "EXPECTED_BASE_SHA" in text
    assert "extract_task11a_stress_features.py" in text
    assert "evaluate_task11a_confidence_router.py" in text
    assert "task8" not in text.lower()
    assert "shutdown" not in text.lower()
