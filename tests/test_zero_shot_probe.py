import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "server" / "zero_shot_probe.py"
SPEC = importlib.util.spec_from_file_location("zero_shot_probe", MODULE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


class ZeroShotProbeHelpersTest(unittest.TestCase):
    def test_extract_json_accepts_fenced_object(self):
        raw = '```json\n{"diagnosis":"x","evidence_present":false}\n```'
        self.assertEqual(PROBE.extract_json(raw)["diagnosis"], "x")

    def test_select_probe_rows_balances_age_stages_and_datasets(self):
        rows = []
        for stage in ("adult", "egg", "larva", "pupa"):
            for index in range(3):
                rows.append(
                    {
                        "dataset": "ages",
                        "stage_normalized": stage,
                        "selection_hash": f"{index}{stage}",
                        "subset_relative": f"ages/{stage}/{index}.jpg",
                    }
                )
        for index in range(8):
            rows.append(
                {
                    "dataset": "IP102",
                    "class_name": f"pest-{index}",
                    "selection_hash": f"{index:02d}",
                    "subset_relative": f"IP102/{index}.jpg",
                }
            )

        selected = PROBE.select_probe_rows(rows, age_limit=8, ip102_limit=5)
        age = [row for row in selected if row["dataset"] == "ages"]
        ip102 = [row for row in selected if row["dataset"] == "IP102"]

        self.assertEqual(len(age), 8)
        self.assertEqual(len(ip102), 5)
        self.assertEqual({row["stage_normalized"] for row in age}, {"adult", "egg", "larva", "pupa"})
        self.assertEqual(len({row["class_name"] for row in ip102}), 5)

    def test_build_prompt_requires_closed_set_json(self):
        row = {"dataset": "ages"}
        prompt = PROBE.build_prompt(row, ["pest a", "pest b"])
        self.assertIn('"diagnosis"', prompt)
        self.assertIn("pest a", prompt)
        self.assertIn("adult|egg|larva|pupa", prompt)
        self.assertIn("JSON only", prompt)

    def test_candidate_labels_include_empty_age_classes_from_taxonomy(self):
        taxonomy = [
            {"dataset": "ages", "species": "pest a", "class_name": ""},
            {"dataset": "ages", "species": "pest b", "class_name": ""},
            {"dataset": "IP102", "species": "", "class_name": "ip pest"},
        ]
        age, ip102 = PROBE.candidate_labels(taxonomy)
        self.assertEqual(age, ["pest a", "pest b"])
        self.assertEqual(ip102, ["ip pest"])

    def test_summarize_results_uses_age_only_for_stage_accuracy(self):
        results = [
            {"dataset": "ages", "parsed": {}, "diagnosis_correct": True, "stage_correct": True, "seconds": 1.0},
            {"dataset": "ages", "parsed": {}, "diagnosis_correct": False, "stage_correct": False, "seconds": 3.0},
            {"dataset": "IP102", "parsed": {}, "diagnosis_correct": True, "stage_correct": False, "seconds": 2.0},
        ]
        summary = PROBE.summarize_results(results, model_path="model", peak_vram_gb=7.5)
        self.assertEqual(summary["age_diagnosis_accuracy"], 0.5)
        self.assertEqual(summary["ip102_diagnosis_accuracy"], 1.0)
        self.assertEqual(summary["age_stage_accuracy"], 0.5)
        self.assertNotIn("stage_accuracy", summary)


if __name__ == "__main__":
    unittest.main()
