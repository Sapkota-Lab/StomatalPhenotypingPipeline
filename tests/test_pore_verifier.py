import unittest

import cv2
import numpy as np

from pore_verifier import mask_metrics, proposal_patch


class PoreVerifierTests(unittest.TestCase):
    def test_empty_mask_has_zero_score(self):
        metrics = mask_metrics(np.zeros((64, 64), dtype=np.float32))
        self.assertEqual(metrics["pore_component_count"], 0)
        self.assertEqual(metrics["pore_score"], 0.0)

    def test_centered_ellipse_has_pore_metrics(self):
        probability = np.zeros((64, 64), dtype=np.float32)
        cv2.ellipse(probability, (32, 32), (14, 4), 20, 0, 360, 0.9, -1)
        metrics = mask_metrics(probability)
        self.assertEqual(metrics["pore_component_count"], 1)
        self.assertGreater(metrics["pore_area_ratio"], 0)
        self.assertLess(metrics["pore_aspect_ratio"], 0.5)
        self.assertLess(metrics["pore_centroid_offset"], 0.05)
        self.assertGreater(metrics["pore_score"], 0.5)

    def test_proposal_patch_is_rgb_64_square(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        patch = proposal_patch(image, (20, 30, 60, 70))
        self.assertEqual(patch.shape, (64, 64, 3))
        self.assertEqual(patch[32, 32, 0], 255)


if __name__ == "__main__":
    unittest.main()
