import unittest

import cv2
import numpy as np

from download_external_validation import (
    QUALITY_ORDER,
    encode_jpeg,
    filter_coco,
    stratified_selection,
)


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_stratified(self):
        names = [f"im{i:03d}" for i in range(12)]
        qualities = {
            name: QUALITY_ORDER[index // 4]
            for index, name in enumerate(names)
        }

        first = stratified_selection(names, qualities, 2, seed=7)
        second = stratified_selection(names, qualities, 2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(
            {quality: sum(item[1] == quality for item in first)
             for quality in QUALITY_ORDER},
            {quality: 2 for quality in QUALITY_ORDER},
        )

    def test_selection_rejects_undersized_stratum(self):
        with self.assertRaises(ValueError):
            stratified_selection(
                ["im000"],
                {"im000": "imperfect"},
                per_quality=2,
                seed=1,
            )


class CocoFilterTests(unittest.TestCase):
    def test_filter_keeps_only_selected_images_and_annotations(self):
        coco = {
            "info": {"description": "test"},
            "categories": [{"id": 1, "name": "stomata"}],
            "images": [
                {"id": 10, "file_name": "im010.png"},
                {"id": 11, "file_name": "im011.png"},
            ],
            "annotations": [
                {"id": 1, "image_id": 10},
                {"id": 2, "image_id": 11},
                {"id": 3, "image_id": 11},
            ],
        }

        result = filter_coco(coco, {"im011"})

        self.assertEqual([item["id"] for item in result["images"]], [11])
        self.assertEqual(
            [item["id"] for item in result["annotations"]],
            [2, 3],
        )
        self.assertEqual(result["info"], {"description": "test"})


class EncodingTests(unittest.TestCase):
    def test_png_is_reencoded_without_changing_dimensions(self):
        image = np.zeros((11, 17, 3), dtype=np.uint8)
        success, png = cv2.imencode(".png", image)
        self.assertTrue(success)

        jpeg, width, height = encode_jpeg(png.tobytes(), quality=90)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        self.assertEqual((width, height), (17, 11))
        self.assertEqual(decoded.shape[:2], (11, 17))


if __name__ == "__main__":
    unittest.main()
