import unittest

from calibrate_tomato_confidence import (
    calibration_metrics,
    label_rows,
    select_threshold,
)
from evaluate_external_predictions import Box


class TomatoConfidenceTests(unittest.TestCase):
    def test_labels_one_prediction_per_ground_truth(self):
        rows = [
            {
                "filename": "a.jpg",
                "confidence": "0.9",
                "x1": "0",
                "y1": "0",
                "x2": "10",
                "y2": "10",
            },
            {
                "filename": "a.jpg",
                "confidence": "0.8",
                "x1": "1",
                "y1": "1",
                "x2": "11",
                "y2": "11",
            },
        ]
        labeled = label_rows(rows, {"a.jpg": [Box(0, 0, 10, 10)]})
        self.assertEqual(
            [row["match_is_true_positive"] for row in labeled],
            [1, 0],
        )

    def test_precision_threshold_prefers_high_recall_valid_prefix(self):
        rows = [
            {"p_stoma": 0.9, "match_is_true_positive": 1},
            {"p_stoma": 0.8, "match_is_true_positive": 1},
            {"p_stoma": 0.7, "match_is_true_positive": 0},
        ]
        self.assertEqual(
            select_threshold(rows, 2, minimum_precision=0.90),
            0.8,
        )

    def test_calibration_metrics_reward_correct_probabilities(self):
        rows = [
            {"match_is_true_positive": 1, "p": 0.9},
            {"match_is_true_positive": 0, "p": 0.1},
        ]
        metrics = calibration_metrics(rows, "p")
        self.assertAlmostEqual(metrics["brier_score"], 0.01)


if __name__ == "__main__":
    unittest.main()
