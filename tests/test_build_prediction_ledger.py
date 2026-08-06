import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from apply_tomato_calibration import calibrated_probability
from build_prediction_ledger import build_prediction_ledger


class PredictionLedgerContractTests(unittest.TestCase):
    def test_probability_contract_is_bounded(self):
        for score in (0.001, 0.2, 0.8, 0.999):
            probability = calibrated_probability(score, 1.0, 0.0)
            self.assertGreater(probability, 0)
            self.assertLess(probability, 1)

    def test_rejected_candidates_do_not_affect_spatial_metrics(self):
        manifest = [{
            "study_id": "study",
            "genotype_id": "genotype",
            "plant_unit_id": "plant",
            "variety": "Variety",
            "water_status": "watered",
            "water_ml": "100",
            "stomata_image": "IMG_0001.JPG",
        }]
        predictions = [
            {
                "filename": "IMG_0001.JPG",
                "x1": "10", "y1": "10", "x2": "20", "y2": "20",
                "raw_detector_score": "0.9",
                "p_stoma": "0.95",
                "decision_status": "accepted",
            },
            {
                "filename": "IMG_0001.JPG",
                "x1": "22", "y1": "10", "x2": "32", "y2": "20",
                "raw_detector_score": "0.1",
                "p_stoma": "0.01",
                "decision_status": "rejected",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "IMG_0001.JPG"
            cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
            rows = build_prediction_ledger(
                manifest,
                predictions,
                Path(directory),
                run_id="run",
                calibration_id="calibration",
            )
        accepted, rejected = rows
        self.assertEqual(
            accepted["nearest_neighbor_center_distance_px"],
            "",
        )
        self.assertEqual(rejected["neighbor_radius_px"], "")


if __name__ == "__main__":
    unittest.main()
