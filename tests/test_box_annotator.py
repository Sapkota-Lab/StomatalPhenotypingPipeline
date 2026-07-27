import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from box_annotator import AnnotationStore, safe_image_name, validate_boxes


class ValidationTests(unittest.TestCase):
    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            safe_image_name("../image.jpg")

    def test_rejects_zero_area_box(self):
        with self.assertRaises(ValueError):
            validate_boxes([{"x1": 5, "y1": 5, "x2": 5, "y2": 10}], 100, 100)

    def test_normalizes_numeric_coordinates(self):
        boxes = validate_boxes(
            [{"x1": "1.234", "y1": 2, "x2": 30, "y2": 40}], 100, 100
        )
        self.assertEqual(
            boxes, [{"x1": 1.23, "y1": 2.0, "x2": 30.0, "y2": 40.0}]
        )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.images = root / "images"
        self.images.mkdir()
        Image.new("RGB", (80, 60), "white").save(self.images / "sample.jpg")
        self.store = AnnotationStore(self.images, root / "annotations")

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trip(self):
        saved = self.store.save(
            {
                "image": "sample.jpg",
                "status": "complete",
                "notes": "clear field",
                "boxes": [{"x1": 10, "y1": 12, "x2": 30, "y2": 42}],
            }
        )
        loaded = self.store.load("sample.jpg")
        self.assertEqual(saved["count"], 1)
        self.assertEqual(loaded["boxes"], saved["boxes"])
        self.assertEqual(loaded["status"], "complete")

    def test_provisional_status_round_trips(self):
        saved = self.store.save(
            {"image": "sample.jpg", "status": "provisional", "boxes": []}
        )
        self.assertEqual(saved["status"], "provisional")

    def test_saved_json_has_oriented_dimensions(self):
        self.store.save(
            {"image": "sample.jpg", "status": "in_progress", "boxes": []}
        )
        path = self.store.annotation_path("sample.jpg")
        payload = json.loads(path.read_text())
        self.assertEqual((payload["width"], payload["height"]), (80, 60))

    def test_priority_csv_controls_first_images(self):
        Image.new("RGB", (20, 20), "white").save(self.images / "other.jpg")
        priority = Path(self.directory.name) / "priority.csv"
        priority.write_text("filename\nother.jpg\n")
        store = AnnotationStore(self.images, Path(self.directory.name) / "out", priority)
        self.assertEqual(store.image_names(), ["other.jpg", "sample.jpg"])


if __name__ == "__main__":
    unittest.main()
