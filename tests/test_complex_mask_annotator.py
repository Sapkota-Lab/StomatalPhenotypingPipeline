import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from complex_mask_annotator import (
    HTML,
    balanced_queue,
    polygon_area,
    quality_queue,
    quality_refill_queue,
    validate_polygon,
)


class PolygonTests(unittest.TestCase):
    def test_area(self):
        self.assertEqual(
            polygon_area([[0, 0], [10, 0], [10, 5], [0, 5]]),
            50,
        )

    def test_rejects_out_of_bounds_point(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_polygon([[0, 0], [10, 0], [11, 5]], 10, 10)


class InterfaceTests(unittest.TestCase):
    def test_item_response_and_proposal_use_distinct_javascript_variables(self):
        self.assertIn("let response=await fetch", HTML)
        self.assertIn("let proposal=item.proposal_summary", HTML)
        self.assertNotIn("let q=await fetch", HTML)


class QueueTests(unittest.TestCase):
    def test_queue_is_balanced_and_deterministic(self):
        manifest = [
            {
                "stomata_image": "A.JPG",
                "plant_unit_id": "A",
                "genotype_id": "V01",
                "variety": "A",
                "water_status": "watered",
            },
            {
                "stomata_image": "B.JPG",
                "plant_unit_id": "B",
                "genotype_id": "V01",
                "variety": "A",
                "water_status": "drought_stressed",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("A", "B"):
                (root / f"{name}.json").write_text(json.dumps({
                    "status": "complete",
                    "boxes": [
                        {"x1": i, "y1": i, "x2": i + 2, "y2": i + 2}
                        for i in range(10)
                    ],
                }))
            first = balanced_queue(manifest, root, 3, 42)
            second = balanced_queue(manifest, root, 3, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(
            {item["image"] for item in first},
            {"A.JPG", "B.JPG"},
        )
        self.assertEqual(
            [item["image"] for item in first[:4]],
            ["A.JPG", "B.JPG", "A.JPG", "B.JPG"],
        )

    def test_quality_queue_prefers_locally_sharp_object(self):
        manifest = [{
            "stomata_image": "A.JPG",
            "plant_unit_id": "A",
            "genotype_id": "V01",
            "variety": "A",
            "water_status": "watered",
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (320, 160), "gray")
            draw = ImageDraw.Draw(image)
            for y in range(25, 136, 10):
                for x in range(25, 136, 10):
                    color = "white" if (x // 10 + y // 10) % 2 else "black"
                    draw.rectangle((x, y, x + 9, y + 9), fill=color)
            image.save(root / "A.JPG", quality=100)
            boxes = root / "boxes"
            boxes.mkdir()
            (boxes / "A.json").write_text(json.dumps({
                "status": "complete",
                "boxes": [
                    {"x1": 35, "y1": 35, "x2": 125, "y2": 125},
                    {"x1": 195, "y1": 35, "x2": 285, "y2": 125},
                ],
            }))
            selected = quality_queue(manifest, boxes, root, 1)
        self.assertEqual(selected[0]["expert_box_index"], 1)
        self.assertEqual(selected[0]["selection_method"], "local_image_quality_v1")

    def test_quality_refill_excludes_already_reviewed_candidates(self):
        manifest = [{
            "stomata_image": "A.JPG",
            "plant_unit_id": "A",
            "genotype_id": "V01",
            "variety": "A",
            "water_status": "watered",
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (320, 160), "gray").save(root / "A.JPG")
            boxes = root / "boxes"
            boxes.mkdir()
            (boxes / "A.json").write_text(json.dumps({
                "status": "complete",
                "boxes": [
                    {"x1": 35, "y1": 35, "x2": 125, "y2": 125},
                    {"x1": 195, "y1": 35, "x2": 285, "y2": 125},
                ],
            }))
            output = root / "output"
            output.mkdir()
            (output / "A_0001.json").write_text(json.dumps({
                "object_id": "A:0001",
                "image": "A.JPG",
                "status": "skipped",
            }))
            selected = quality_refill_queue(
                manifest,
                boxes,
                root,
                output,
                target_complete_per_image=1,
            )
        self.assertEqual([item["object_id"] for item in selected], ["A:0002"])
        self.assertEqual(
            selected[0]["selection_method"],
            "local_image_quality_refill_v1",
        )


if __name__ == "__main__":
    unittest.main()
