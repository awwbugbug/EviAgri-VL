import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluation_shutdown_guard import (
    BLOCKED,
    READY,
    WAITING,
    evaluate_shutdown_gate,
    screen_session_active,
    write_checksum_manifest,
)


class EvaluationShutdownGuardTest(unittest.TestCase):
    def _complete_evaluation(self, root: Path) -> None:
        (root / "val").mkdir(parents=True)
        (root / "test").mkdir(parents=True)
        (root / "evaluation_summary.json").write_text(
            json.dumps({"completed": True}), encoding="utf-8"
        )
        for split, count in (("val", 2), ("test", 3)):
            split_dir = root / split
            (split_dir / "predictions.jsonl").write_text(
                "".join(json.dumps({"id": f"{split}-{index}"}) + "\n" for index in range(count)),
                encoding="utf-8",
            )
            (split_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
            (split_dir / "failures.jsonl").write_text("", encoding="utf-8")

    def test_dead_screen_is_not_active(self):
        output = "123.static_qlora_eval (Dead ???)\n"
        self.assertFalse(screen_session_active(output, "static_qlora_eval"))
        self.assertTrue(
            screen_session_active("123.static_qlora_eval (Detached)\n", "static_qlora_eval")
        )

    def test_complete_exact_artifacts_are_ready_only_after_evaluator_exits(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            root = Path(tmp_string)
            self._complete_evaluation(root)

            running = evaluate_shutdown_gate(root, {"val": 2, "test": 3}, evaluator_active=True)
            ready = evaluate_shutdown_gate(root, {"val": 2, "test": 3}, evaluator_active=False)

            self.assertEqual(running.state, WAITING)
            self.assertEqual(ready.state, READY)

    def test_missing_predictions_block_if_evaluator_has_stopped(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            root = Path(tmp_string)
            self._complete_evaluation(root)
            (root / "test" / "predictions.jsonl").write_text("{}\n", encoding="utf-8")

            report = evaluate_shutdown_gate(root, {"val": 2, "test": 3}, evaluator_active=False)

            self.assertEqual(report.state, BLOCKED)
            self.assertIn("test predictions", " ".join(report.reasons))

    def test_failure_report_always_blocks_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            root = Path(tmp_string)
            self._complete_evaluation(root)
            (root / "failure_123.json").write_text("{}\n", encoding="utf-8")

            report = evaluate_shutdown_gate(root, {"val": 2, "test": 3}, evaluator_active=True)

            self.assertEqual(report.state, BLOCKED)

    def test_checksum_manifest_covers_all_completion_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            root = Path(tmp_string)
            self._complete_evaluation(root)

            manifest = write_checksum_manifest(root)
            text = manifest.read_text(encoding="utf-8")

            self.assertIn("evaluation_summary.json", text)
            self.assertIn("val/predictions.jsonl", text)
            self.assertIn("test/metrics.json", text)
            self.assertIn("test/failures.jsonl", text)


if __name__ == "__main__":
    unittest.main()
