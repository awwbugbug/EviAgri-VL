import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from build_evidence_first_jsonl import build_detection_records, load_class_map, write_jsonl_bundle


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(40, 90, 30)).save(path)


def annotation(image_id: str, class_id: int, box: tuple[int, int, int, int]) -> str:
    xmin, ymin, xmax, ymax = box
    return f"""<annotation><filename>{image_id}</filename>
    <size><width>100</width><height>80</height><depth>3</depth></size>
    <object><name>{class_id}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>
    <xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>"""


class BuildEvidenceFirstJsonlTest(unittest.TestCase):
    def test_builds_grounded_positive_and_absent_class_null_samples(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            voc = tmp / "VOC2007"
            annotations = voc / "Annotations"
            splits = voc / "ImageSets" / "Main"
            annotations.mkdir(parents=True)
            splits.mkdir(parents=True)
            for image_id in ("img1", "img2", "img3"):
                write_image(voc / "JPEGImages" / f"{image_id}.jpg")
            (annotations / "img1.xml").write_text(annotation("img1", 0, (1, 2, 20, 30)), encoding="utf-8")
            (annotations / "img2.xml").write_text(annotation("img2", 0, (5, 6, 5, 20)), encoding="utf-8")
            (annotations / "img3.xml").write_text(annotation("img3", 1, (3, 4, 40, 50)), encoding="utf-8")
            (splits / "trainval.txt").write_text("img1\nimg2\n", encoding="utf-8")
            (splits / "test.txt").write_text("img3\n", encoding="utf-8")
            classes = tmp / "classes.txt"
            classes.write_text("1 pest zero\n2 pest one\n3 pest two\n", encoding="utf-8")

            class_map = load_class_map(classes)
            bundle, summary = build_detection_records(
                voc,
                class_map,
                image_prefix="/data/VOC2007/JPEGImages",
                val_percent=0,
                seed="fixture",
            )

            self.assertEqual(summary["valid_images"], 2)
            self.assertEqual(summary["skipped_without_valid_boxes"], 1)
            self.assertEqual(len(bundle["train"]["positive"]), 1)
            self.assertEqual(len(bundle["train"]["null"]), 1)
            self.assertEqual(len(bundle["test"]["positive"]), 1)
            positive = bundle["train"]["positive"][0]
            null = bundle["train"]["null"][0]
            self.assertEqual(list(positive["target"]), [
                "evidence_present", "evidence_bbox", "visible_attributes", "diagnosis", "reliability"
            ])
            self.assertEqual(positive["target"]["evidence_bbox"], [1, 2, 20, 30])
            self.assertEqual(positive["target"]["diagnosis"]["pest_id"], 0)
            self.assertFalse(null["target"]["evidence_present"])
            self.assertIsNone(null["target"]["evidence_bbox"])
            self.assertNotEqual(null["query_pest_id"], 0)
            self.assertEqual(json.loads(positive["messages"][1]["content"][0]["text"]), positive["target"])

    def test_writes_positive_and_null_files_separately(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            output = Path(tmp_string)
            bundle = {
                split: {"positive": [], "null": []} for split in ("train", "val", "test")
            }
            bundle["train"]["positive"] = [{"id": "positive"}]
            bundle["train"]["null"] = [{"id": "null"}]
            write_jsonl_bundle(bundle, output, {"version": "fixture"})

            self.assertEqual(json.loads((output / "vlm_sft" / "train_evidence_positive.jsonl").read_text()), {"id": "positive"})
            self.assertEqual(json.loads((output / "hallucination" / "train_prompt_conflict.jsonl").read_text()), {"id": "null"})
            self.assertTrue((output / "metadata" / "build_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
