import unittest

from apply_tomato_calibration import (
    apply_calibration,
    calibrated_probability,
)


class ApplyTomatoCalibrationTests(unittest.TestCase):
    def test_probability_is_monotonic(self):
        low = calibrated_probability(0.1, 1.2, -0.3)
        high = calibrated_probability(0.9, 1.2, -0.3)
        self.assertLess(low, high)

    def test_assigns_all_decision_tiers(self):
        report = {
            "coefficient": 1.0,
            "intercept": 0.0,
            "accepted_threshold_precision_target_0_90": 0.8,
            "review_floor": 0.1,
        }
        rows = [
            {"confidence": "0.9"},
            {"confidence": "0.5"},
            {"confidence": "0.05"},
        ]
        result = apply_calibration(rows, report)
        self.assertEqual(
            [row["decision_status"] for row in result],
            ["accepted", "review", "rejected"],
        )


if __name__ == "__main__":
    unittest.main()
