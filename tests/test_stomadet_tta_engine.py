import unittest

import numpy as np

from stomadet_tta_engine import (
    TTAResult,
    VIEW_NAMES,
    ViewDetection,
    cluster_views,
    inverse_box,
    stability_metrics,
    transform_image,
)
from stomata_engine import Detection


class TransformTests(unittest.TestCase):
    def test_all_transforms_and_inverses_preserve_box(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        original = Detection(10, 15, 30, 35, 0.9)
        transformed_boxes = {
            "identity": Detection(10, 15, 30, 35, 0.9),
            "gray_clahe": Detection(10, 15, 30, 35, 0.9),
            "horizontal_flip": Detection(70, 15, 90, 35, 0.9),
            "rotate90": Detection(25, 10, 45, 30, 0.9),
            "rotate180": Detection(70, 25, 90, 45, 0.9),
            "rotate270": Detection(15, 70, 35, 90, 0.9),
        }

        for view in VIEW_NAMES:
            transformed = transform_image(image, view)
            if view in {"rotate90", "rotate270"}:
                self.assertEqual(transformed.shape[:2], (100, 60))
            else:
                self.assertEqual(transformed.shape[:2], (60, 100))
            restored = inverse_box(
                transformed_boxes[view],
                view,
                width=100,
                height=60,
            )
            self.assertEqual(restored.box, original.box)


class ClusteringTests(unittest.TestCase):
    def test_one_detection_per_view_forms_one_cluster(self):
        detections = [
            ViewDetection(
                view,
                Detection(10 + offset, 10, 30 + offset, 40, 0.9),
            )
            for offset, view in enumerate(VIEW_NAMES)
        ]

        clusters = cluster_views(detections)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), len(VIEW_NAMES))

    def test_neighboring_objects_remain_separate(self):
        detections = [
            ViewDetection("identity", Detection(0, 0, 20, 30, 0.9)),
            ViewDetection("rotate90", Detection(1, 0, 21, 30, 0.8)),
            ViewDetection("identity", Detection(60, 0, 80, 30, 0.9)),
            ViewDetection("rotate90", Detection(61, 0, 81, 30, 0.8)),
        ]

        clusters = cluster_views(detections)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(len(item) for item in clusters), [2, 2])


class StabilityMetricTests(unittest.TestCase):
    def test_perfectly_repeatable_views_score_one(self):
        by_view = {
            view: [
                Detection(10, 10, 20, 30, 0.9),
                Detection(40, 40, 50, 60, 0.8),
            ]
            for view in VIEW_NAMES
        }
        clusters = [
            [
                ViewDetection(view, detections[index])
                for view, detections in by_view.items()
            ]
            for index in range(2)
        ]
        metrics = stability_metrics(TTAResult([], clusters, by_view))

        self.assertEqual(metrics["geometric_count_cv"], 0.0)
        self.assertEqual(metrics["identity_repeatability"], 1.0)
        self.assertEqual(metrics["geometric_consensus_fraction"], 1.0)
        self.assertEqual(metrics["contrast_dice"], 1.0)

    def test_single_view_noise_reduces_repeatability(self):
        identity = Detection(10, 10, 20, 30, 0.9)
        repeated = Detection(40, 40, 50, 60, 0.8)
        by_view = {view: [] for view in VIEW_NAMES}
        by_view["identity"] = [identity, repeated]
        by_view["rotate90"] = [repeated]
        clusters = [
            [ViewDetection("identity", identity)],
            [
                ViewDetection("identity", repeated),
                ViewDetection("rotate90", repeated),
            ],
        ]
        metrics = stability_metrics(TTAResult([], clusters, by_view))

        self.assertEqual(metrics["identity_repeatability"], 0.5)
        self.assertEqual(metrics["geometric_consensus_fraction"], 0.0)
        self.assertGreater(metrics["geometric_count_cv"], 0.0)
        self.assertEqual(metrics["contrast_dice"], 0.0)


if __name__ == "__main__":
    unittest.main()
