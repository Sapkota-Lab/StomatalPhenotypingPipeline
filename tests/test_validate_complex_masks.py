import unittest

from validate_complex_masks import compare_polygons, gate


class ComplexMaskValidationTests(unittest.TestCase):
    def test_identical_polygons_are_perfect(self):
        polygon = [[0, 0], [20, 0], [20, 10], [0, 10]]
        metrics = compare_polygons(polygon, polygon)
        self.assertEqual(metrics["mask_iou"], 1)
        self.assertEqual(metrics["mask_dice"], 1)
        self.assertEqual(metrics["relative_area_error"], 0)

    def test_larger_proposal_has_positive_area_error(self):
        manual = [[0, 0], [10, 0], [10, 10], [0, 10]]
        proposed = [[0, 0], [12, 0], [12, 10], [0, 10]]
        metrics = compare_polygons(manual, proposed)
        self.assertGreater(metrics["relative_area_error"], 0)
        self.assertLess(metrics["mask_iou"], 1)

    def test_small_sample_cannot_pass_gate(self):
        rows = [{
            "plant_unit_id": "plant",
            "water_status": "watered",
            "mask_iou": 1,
            "mask_dice": 1,
            "relative_area_error": 0,
            "absolute_percentage_area_error": 0,
        }]
        self.assertFalse(gate(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
