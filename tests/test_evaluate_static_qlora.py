import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from evaluate_static_qlora import (
    bbox_iou,
    compute_metrics,
    finalize_predictions,
    generate_predictions,
    generation_kwargs,
    parse_structured_json,
    pending_records,
    pointing_game,
)


def target(present: bool, pest_id: int | None = None) -> dict:
    return {
        "evidence_present": present,
        "evidence_bbox": [10, 10, 30, 30] if present else None,
        "visible_attributes": [],
        "diagnosis": {"pest_id": pest_id, "pest_name": f"pest-{pest_id}"} if present else "uncertain",
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }


def prediction(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


def fixture_predictions() -> list[dict]:
    positive_correct = target(True, 0)
    positive_missed = target(False)
    null_correct = target(False)
    return [
        {"id": "p0", "target": target(True, 0), "prediction": prediction(positive_correct)},
        {"id": "p1", "target": target(True, 1), "prediction": prediction(positive_missed)},
        {"id": "n0", "target": target(False), "prediction": prediction(null_correct)},
        {"id": "n1", "target": target(False), "prediction": "not-json"},
    ]


class EvaluateStaticQloraTest(unittest.TestCase):
    def test_parses_only_exact_ordered_evidence_first_schema(self):
        value = target(True, 0)

        self.assertEqual(parse_structured_json(f"```json\n{prediction(value)}\n```"), value)
        reordered = {"diagnosis": value["diagnosis"], **value}
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_structured_json(prediction(reordered))

    def test_bbox_iou_and_pointing_game(self):
        truth = [10, 10, 30, 30]

        self.assertEqual(bbox_iou(truth, truth), 1.0)
        self.assertAlmostEqual(bbox_iou([20, 20, 40, 40], truth), 100 / 700)
        self.assertEqual(pointing_game([12, 12, 20, 20], truth), 1.0)
        self.assertEqual(pointing_game([30, 30, 40, 40], truth), 0.0)

    def test_metrics_separate_positive_and_null_behavior(self):
        metrics = compute_metrics(fixture_predictions())

        self.assertEqual(metrics["schema_valid_rate"], 0.75)
        self.assertAlmostEqual(metrics["evidence_presence_f1"], 2 / 3)
        self.assertEqual(metrics["positive"]["diagnosis_accuracy"], 0.5)
        self.assertEqual(metrics["positive"]["diagnosis_macro_f1"], 0.5)
        self.assertEqual(metrics["positive"]["mean_iou"], 0.5)
        self.assertEqual(metrics["positive"]["iou_at_0.5"], 0.5)
        self.assertEqual(metrics["positive"]["pointing_game"], 0.5)
        self.assertEqual(metrics["null"]["false_positive_rate"], 0.5)
        self.assertEqual(metrics["parse_failure_count"], 1)

    def test_resume_skips_only_unique_completed_prediction_ids(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            path = Path(tmp_string) / "predictions.jsonl"
            path.write_text(json.dumps({"id": "row-1", "prediction": "{}"}) + "\n", encoding="utf-8")
            records = [{"id": "row-1"}, {"id": "row-2"}]

            remaining = pending_records(records, path)

            self.assertEqual(remaining, [{"id": "row-2"}])
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "row-1", "prediction": "{}"}) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate prediction id"):
                pending_records(records, path)

    def test_generation_is_deterministic_and_bounded(self):
        self.assertEqual(
            generation_kwargs(),
            {"max_new_tokens": 128, "do_sample": False, "temperature": None},
        )

    def test_generation_loop_is_resume_safe_and_finalizes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            output = Path(tmp_string)
            records = [
                {"id": "p0", "split": "val", "task_type": "pest_evidence_grounding", "target": target(True, 0)},
                {"id": "n0", "split": "val", "task_type": "prompt_conflict_null_evidence", "target": target(False)},
            ]
            calls = []

            summary = generate_predictions(
                records,
                output / "predictions.jsonl",
                generate_fn=lambda row: calls.append(row["id"]) or prediction(row["target"]),
            )
            resumed = generate_predictions(
                records,
                output / "predictions.jsonl",
                generate_fn=lambda row: self.fail(f"regenerated completed row {row['id']}"),
            )
            metrics = finalize_predictions(records, output)

            self.assertEqual(summary, {"existing": 0, "generated": 2, "total": 2})
            self.assertEqual(resumed, {"existing": 2, "generated": 0, "total": 2})
            self.assertEqual(calls, ["p0", "n0"])
            self.assertEqual(metrics["schema_valid_rate"], 1.0)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "failures.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
