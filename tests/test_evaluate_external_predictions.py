import tempfile
import unittest
from pathlib import Path

from evaluate_external_predictions import (
    Box,
    box_iou,
    evaluate,
    interpolated_ap,
    load_predictions,
    match_image,
)


class MatchingTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(
            box_iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)),
            1.0,
        )
        self.assertEqual(
            box_iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)),
            0.0,
        )

    def test_one_ground_truth_cannot_match_twice(self):
        ground_truth = [Box(0, 0, 10, 10)]
        predictions = [
            Box(0, 0, 10, 10, 0.9),
            Box(1, 1, 9, 9, 0.8),
        ]

        matches = match_image(predictions, ground_truth, 0.5)

        self.assertEqual([item[1] for item in matches], [True, False])


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_have_perfect_metrics(self):
        predictions = {"image.jpg": [Box(0, 0, 10, 10, 0.9)]}
        ground_truth = {"image.jpg": [Box(0, 0, 10, 10)]}

        summary, rows = evaluate(
            predictions,
            ground_truth,
            {"image.jpg": "default"},
            0.5,
        )

        self.assertEqual(summary["precision"], 1.0)
        self.assertEqual(summary["recall"], 1.0)
        self.assertEqual(summary["f1"], 1.0)
        self.assertEqual(summary["ap50"], 1.0)
        self.assertEqual(rows[0]["count_error"], 0)

    def test_empty_predictions_report_false_negative(self):
        summary, _rows = evaluate(
            {},
            {"image.jpg": [Box(0, 0, 10, 10)]},
            {},
            0.5,
        )

        self.assertEqual(summary["false_negative"], 1)
        self.assertEqual(summary["recall"], 0.0)
        self.assertEqual(interpolated_ap([], 1), 0.0)

    def test_load_predictions_filters_tta_support_and_area(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.csv"
            path.write_text(
                "filename,x1,y1,x2,y2,confidence,view_support,area_ratio\n"
                "image.jpg,0,0,10,10,0.9,5,1.0\n"
                "image.jpg,20,20,30,30,0.9,2,1.0\n"
                "image.jpg,40,40,50,50,0.9,5,2.0\n"
            )
            predictions = load_predictions(
                path,
                status=None,
                source=None,
                confidence=0.5,
                min_view_support=4,
                min_area_ratio=0.5,
                max_area_ratio=1.5,
            )
            self.assertEqual(1, len(predictions["image.jpg"]))

    def test_load_predictions_can_use_named_score_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.csv"
            path.write_text(
                "filename,x1,y1,x2,y2,confidence,morphology_confidence\n"
                "image.jpg,0,0,10,10,0.99,0.2\n"
                "image.jpg,20,20,30,30,0.1,0.8\n"
            )
            predictions = load_predictions(
                path,
                status=None,
                source=None,
                confidence=0.5,
                score_field="morphology_confidence",
            )
            self.assertEqual(1, len(predictions["image.jpg"]))
            self.assertAlmostEqual(0.8, predictions["image.jpg"][0].score)


if __name__ == "__main__":
    unittest.main()
