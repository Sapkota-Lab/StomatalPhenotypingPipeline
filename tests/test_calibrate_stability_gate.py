import unittest

from calibrate_stability_gate import (
    accuracy_gate,
    accurate_filenames,
    apply_thresholds,
    calibrate,
)


class StabilityCalibrationTests(unittest.TestCase):
    def test_accuracy_gate_must_pass_before_stability(self):
        summary = {
            "ground_truth": 100,
            "images": 10,
            "precision": 0.95,
            "recall": 0.90,
            "count_mae": 2.0,
            "count_bias": -1.0,
        }
        passed, _limits = accuracy_gate(summary)
        self.assertTrue(passed)
        summary["recall"] = 0.8
        self.assertFalse(accuracy_gate(summary)[0])

    def test_calibrates_and_applies_directional_thresholds(self):
        accuracy = [
            {
                "filename": f"image{index}.jpg",
                "ground_truth_count": "20",
                "true_positive": "19",
                "false_positive": "1",
                "false_negative": "1",
                "count_error": "0",
            }
            for index in range(5)
        ]
        stability = [
            {
                "filename": f"image{index}.jpg",
                "geometric_count_cv": str(0.01 * index),
                "geometric_max_relative_deviation": str(0.02 * index),
                "identity_repeatability": str(1.0 - 0.01 * index),
                "geometric_consensus_fraction": str(1.0 - 0.02 * index),
                "contrast_dice": str(1.0 - 0.03 * index),
            }
            for index in range(5)
        ]
        eligible = accurate_filenames(accuracy)
        thresholds = calibrate(stability, eligible)
        target = [{
            **stability[0],
            "filename": "target.jpg",
            "identity_repeatability": "0.1",
        }]
        audited = apply_thresholds(target, thresholds)

        self.assertEqual(len(eligible), 5)
        self.assertEqual(audited[0]["stability_status"], "review")
        self.assertIn(
            "identity_repeatability",
            audited[0]["failed_stability_metrics"],
        )


if __name__ == "__main__":
    unittest.main()
