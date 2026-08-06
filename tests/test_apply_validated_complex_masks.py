import unittest
from pathlib import Path

from apply_validated_complex_masks import apply_masks


class ApplyValidatedMasksTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "stomata_image": "IMG_0001.JPG",
            "detection_id": "1",
            "edge_censored": "False",
        }
        self.proposal = {
            "object_id": "IMG_0001:0001",
            "status": "proposed",
            "polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
            "model_sha256": "hash",
            "model_confidence": 0.9,
            "geometry_method": "convex",
        }
        self.report = {
            "measurement_gate": {"passed": True},
            "proposal_model_hashes": ["hash"],
            "proposal_geometry_methods": ["convex"],
        }

    def test_populates_area_only_after_passed_gate(self):
        output = apply_masks(
            [self.row],
            {"IMG_0001:0001": (self.proposal, Path("mask.json"))},
            self.report,
        )
        self.assertEqual(output[0]["complex_area_px2"], 200)
        self.assertTrue(output[0]["measurement_quality_passed"])

    def test_rejects_unpassed_gate(self):
        self.report["measurement_gate"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "has not passed"):
            apply_masks(
                [self.row],
                {},
                self.report,
            )

    def test_edge_object_is_excluded(self):
        self.row["edge_censored"] = "True"
        output = apply_masks(
            [self.row],
            {"IMG_0001:0001": (self.proposal, Path("mask.json"))},
            self.report,
        )
        self.assertFalse(output[0]["measurement_quality_passed"])
        self.assertEqual(
            output[0]["measurement_exclusion_reason"],
            "complex_mask_edge_censored",
        )


if __name__ == "__main__":
    unittest.main()
