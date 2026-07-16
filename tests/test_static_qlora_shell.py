import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "server" / "run_static_qlora_formal.sh"
STATUS = ROOT / "server" / "check_static_qlora_status.sh"
EVAL_LAUNCHER = ROOT / "server" / "run_static_qlora_evaluation.sh"
EVAL_STATUS = ROOT / "server" / "check_static_qlora_evaluation_status.sh"
SHUTDOWN_GUARD = ROOT / "server" / "run_evaluation_shutdown_guard.sh"


class StaticQloraShellTest(unittest.TestCase):
    def test_formal_launcher_requires_passed_smoke_gate(self):
        text = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("smoke_gate.json", text)
        self.assertIn('"passed": true', text)
        self.assertIn("screen -L", text)
        self.assertIn("-dmS static_qlora_v1", text)
        self.assertIn("--mode formal", text)
        self.assertIn("20 * 1024 * 1024 * 1024", text)

    def test_status_checker_writes_machine_readable_state(self):
        text = STATUS.read_text(encoding="utf-8")

        self.assertIn("status.json", text)
        self.assertIn("screen_running", text)
        self.assertIn("latest_loss", text)
        self.assertIn("progress_matches", text)
        self.assertIn("total_steps", text)
        self.assertIn("checkpoints", text)
        self.assertIn("nvidia-smi", text)

    def test_evaluation_launcher_requires_completed_formal_adapter(self):
        text = EVAL_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("run_summary.json", text)
        self.assertIn('"completed"', text)
        self.assertIn("adapter_model.safetensors", text)
        self.assertIn("static_qlora_eval", text)
        self.assertIn("--splits val test", text)
        self.assertIn("screen -L", text)

    def test_evaluation_status_is_read_only_and_progress_aware(self):
        text = EVAL_STATUS.read_text(encoding="utf-8")

        self.assertIn("evaluation/status.json", text)
        self.assertIn("predictions.jsonl", text)
        self.assertIn("screen_running", text)
        self.assertIn("nvidia-smi", text)

    def test_shutdown_guard_is_failure_safe_and_uses_platform_shutdown(self):
        text = SHUTDOWN_GUARD.read_text(encoding="utf-8")

        self.assertIn("evaluation_shutdown_guard.py", text)
        self.assertIn("/usr/bin/shutdown", text)
        self.assertIn("sync", text)
        self.assertIn("static_qlora_shutdown_guard", text)
        self.assertIn("guard_blocked", text)


if __name__ == "__main__":
    unittest.main()
