import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from audit_full_datasets import audit_classification_dataset, audit_detection_dataset


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=(20, 80, 40)).save(path)


def write_xml(path: Path, image_id: str, class_id: int, box: tuple[int, int, int, int]) -> None:
    xmin, ymin, xmax, ymax = box
    path.write_text(
        f"""<annotation><filename>{image_id}</filename>
        <size><width>16</width><height>12</height><depth>3</depth></size>
        <object><name>{class_id}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>""",
        encoding="utf-8",
    )


class AuditFullDatasetsTest(unittest.TestCase):
    def test_audits_class_directories_and_unreadable_images(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            root = Path(tmp_string) / "ages"
            write_image(root / "train" / "Species-adult" / "a.jpg")
            write_image(root / "val" / "Species-larva" / "b.jpg")
            corrupt = root / "test" / "Species-adult" / "bad.jpg"
            corrupt.parent.mkdir(parents=True)
            corrupt.write_bytes(b"not-an-image")

            audit = audit_classification_dataset(root, class_directories=True)

            self.assertEqual(audit["split_image_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(audit["class_count"], 2)
            self.assertEqual(audit["species_count"], 1)
            self.assertEqual(audit["stages"], ["adult", "larva"])
            self.assertEqual(len(audit["unreadable_images"]), 1)

    def test_audits_flat_classification_manifests(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            root = Path(tmp_string) / "IP102"
            root.mkdir()
            (root / "classes.txt").write_text("1 class zero\n2 class one\n", encoding="utf-8")
            write_image(root / "train" / "0" / "a.jpg")
            write_image(root / "val" / "1" / "b.jpg")
            write_image(root / "test" / "1" / "c.jpg")
            (root / "train.txt").write_text("a.jpg 0\n", encoding="utf-8")
            (root / "val.txt").write_text("missing.jpg 1\n", encoding="utf-8")
            (root / "test.txt").write_text("c.jpg 1\n", encoding="utf-8")

            audit = audit_classification_dataset(root, class_directories=False)

            self.assertEqual(audit["declared_class_count"], 2)
            self.assertEqual(audit["manifest_missing_files"], ["val/1/missing.jpg"])
            self.assertEqual(audit["images_not_in_manifest"], ["val/1/b.jpg"])
            self.assertEqual(audit["invalid_manifest_labels"], [])

    def test_audits_detection_splits_and_degenerate_boxes(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            voc = Path(tmp_string) / "VOC2007"
            annotations = voc / "Annotations"
            image_sets = voc / "ImageSets" / "Main"
            annotations.mkdir(parents=True)
            image_sets.mkdir(parents=True)
            write_image(voc / "JPEGImages" / "img1.jpg")
            write_image(voc / "JPEGImages" / "img2.jpg")
            write_xml(annotations / "img1.xml", "img1", 0, (1, 1, 10, 10))
            write_xml(annotations / "img2.xml", "img2", 1, (5, 2, 5, 8))
            (image_sets / "trainval.txt").write_text("img1\nmissing\n", encoding="utf-8")
            (image_sets / "test.txt").write_text("img2\n", encoding="utf-8")

            audit = audit_detection_dataset(voc)

            self.assertEqual(audit["image_count"], 2)
            self.assertEqual(audit["annotation_count"], 2)
            self.assertEqual(audit["box_count"], 2)
            self.assertEqual(audit["degenerate_box_count"], 1)
            self.assertEqual(audit["split_missing_images"], ["trainval/missing"])
            self.assertEqual(audit["observed_class_ids"], [0, 1])


if __name__ == "__main__":
    unittest.main()
