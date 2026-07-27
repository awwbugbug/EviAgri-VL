from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_task11a3_router_shell_is_one_time_and_fail_closed():
    text = (ROOT / "server" / "run_task11a3_plantseg_router.sh").read_text(encoding="utf-8")
    assert "test ! -e \"$ROOT\"" in text
    assert "formal_v2" in text
    assert "PROJECT=/root/EviAgri-VL/task11a3_router_code_20260727_v3" in text
    assert "EXPECTED_FINAL_AUDIT_SHA" in text
    assert "sha256sum -c completion.sha256" in text
    assert "--summary-version task11a3-plantseg-feature-summary-1" in text
    assert "--repetitions 1000" in text
    assert "task8_locked_set_read\": false" in text
    assert "task11b_started\": false" in text
    assert "shutdown" not in text and "poweroff" not in text
