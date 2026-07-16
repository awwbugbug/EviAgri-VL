import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "server" / "detection_grounding_probe.py"
SPEC = importlib.util.spec_from_file_location("detection_grounding_probe", MODULE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


class DetectionGroundingHelpersTest(unittest.TestCase):
    def test_normalized_bbox_to_pixels(self):
        self.assertEqual(PROBE.normalized_bbox_to_pixels([100, 200, 900, 800], 200, 100), [20.0, 20.0, 180.0, 80.0])
        self.assertIsNone(PROBE.normalized_bbox_to_pixels([100, 200, 1001, 800], 200, 100))
        self.assertIsNone(PROBE.normalized_bbox_to_pixels([500, 200, 400, 800], 200, 100))

    def test_iou_and_pointing(self):
        predicted = [0.0, 0.0, 10.0, 10.0]
        truth = [5.0, 5.0, 15.0, 15.0]
        self.assertAlmostEqual(PROBE.bbox_iou(predicted, truth), 25 / 175)
        self.assertTrue(PROBE.pointing_hit(predicted, [truth]))
        self.assertFalse(PROBE.pointing_hit([0.0, 0.0, 2.0, 2.0], [truth]))

    def test_select_targets_uses_distinct_classes(self):
        records = [
            {"image_id": "a", "class_ids": "0", "image_sha256": "03"},
            {"image_id": "b", "class_ids": "0|1", "image_sha256": "01"},
            {"image_id": "c", "class_ids": "2", "image_sha256": "02"},
        ]
        selected = PROBE.select_targets(records, limit=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({target["target_class_id"] for target in selected}), 3)


if __name__ == "__main__":
    unittest.main()
