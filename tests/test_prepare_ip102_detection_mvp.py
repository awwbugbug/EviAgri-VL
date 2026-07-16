import csv
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_ip102_detection_mvp.py"


def annotation(image_id: str, class_id: int, box: tuple[int, int, int, int]) -> str:
    xmin, ymin, xmax, ymax = box
    return f"""<annotation><filename>{image_id}</filename><size><width>100</width><height>80</height><depth>3</depth></size><object><name>{class_id}</name><difficult>0</difficult><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>"""


class PrepareDetectionMvpTest(unittest.TestCase):
    def test_repairs_duplicate_root_and_drops_degenerate_box(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            tmp = Path(tmp_string)
            voc = tmp / "VOC2007"
            annotations = voc / "Annotations"
            image_sets = voc / "ImageSets" / "Main"
            annotations.mkdir(parents=True)
            image_sets.mkdir(parents=True)

            (annotations / "img1.xml").write_text(annotation("img1", 0, (1, 2, 20, 30)))
            duplicate = annotation("img2", 1, (2, 3, 40, 50))
            (annotations / "img2.xml").write_text(duplicate + "\n" + duplicate)
            (annotations / "img3.xml").write_text(annotation("img3", 0, (5, 6, 5, 20)))
            (image_sets / "trainval.txt").write_text("img1\nimg2\nimg3\n")
            (image_sets / "test.txt").write_text("")

            image_source = tmp / "JPEGImages"
            image_source.mkdir()
            for image_id in ("img1", "img2", "img3"):
                (image_source / f"{image_id}.jpg").write_bytes(f"jpeg-{image_id}".encode())
            with tarfile.open(voc / "JPEGImages.tar", "w") as archive:
                archive.add(image_source, arcname="JPEGImages")

            classes = tmp / "classes.txt"
            classes.write_text("1 class zero\n2 class one\n")
            output = tmp / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--voc-root",
                    str(voc),
                    "--classes-file",
                    str(classes),
                    "--output-root",
                    str(output),
                    "--per-class",
                    "1",
                    "--seed",
                    "fixture-seed",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(audit["duplicate_root_repairs"], 1)
            self.assertEqual(audit["invalid_boxes_dropped"], 1)
            self.assertEqual(audit["annotations_without_valid_boxes"], 1)
            self.assertEqual(audit["selected_images"], 2)
            self.assertEqual(audit["covered_classes"], 2)

            with (output / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["image_id"] for row in rows}, {"img1", "img2"})
            self.assertTrue((output / "images" / "img1.jpg").is_file())
            self.assertTrue((output / "images" / "img2.jpg").is_file())
            self.assertFalse((output / "images" / "img3.jpg").exists())


if __name__ == "__main__":
    unittest.main()
