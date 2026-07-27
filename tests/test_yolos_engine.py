import tempfile
import unittest
from pathlib import Path

from PIL import Image

from stomata_engine import Detection, global_nms
from yolos_stomata_engine import open_oriented_image


class OrientedImageTests(unittest.TestCase):
    def test_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (30, 20), "white")
            exif = image.getexif()
            exif[274] = 8  # Rotate 90 degrees counter-clockwise for display.
            image.save(path, exif=exif)

            loaded = open_oriented_image(path)

            self.assertEqual(loaded.size, (20, 30))


class CenterDistanceNmsTests(unittest.TestCase):
    def test_nearly_coincident_low_iou_boxes_are_deduplicated(self):
        detections = [
            Detection(10, 10, 30, 90, 0.98),
            Detection(10, 55, 30, 95, 0.95),
        ]

        kept = global_nms(
            detections,
            threshold=0.30,
            center_distance_threshold=0.55,
        )

        self.assertEqual(kept, [detections[0]])

    def test_neighboring_objects_are_preserved(self):
        detections = [
            Detection(10, 10, 30, 90, 0.98),
            Detection(55, 10, 75, 90, 0.95),
        ]

        kept = global_nms(
            detections,
            threshold=0.30,
            center_distance_threshold=0.55,
        )

        self.assertEqual(kept, detections)

    def test_smaller_higher_confidence_box_still_suppresses_duplicate(self):
        detections = [
            Detection(10, 55, 30, 95, 0.99),
            Detection(10, 10, 30, 90, 0.95),
        ]

        kept = global_nms(
            detections,
            threshold=0.30,
            center_distance_threshold=0.55,
        )

        self.assertEqual(kept, [detections[0]])


if __name__ == "__main__":
    unittest.main()
