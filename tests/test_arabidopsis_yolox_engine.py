import unittest

import numpy as np

from arabidopsis_yolox_engine import nms, preprocess


class ArabidopsisYoloXTests(unittest.TestCase):
    def test_preprocess_letterboxes_without_normalizing_pixels(self):
        image = np.full((100, 200, 3), 50, dtype=np.uint8)
        tensor, ratio = preprocess(image, (200, 200))
        self.assertEqual(tensor.shape, (3, 200, 200))
        self.assertEqual(ratio, 1.0)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tensor[0, 50, 50], 50)
        self.assertEqual(tensor[0, 150, 50], 114)

    def test_nms_keeps_highest_scoring_overlap(self):
        boxes = np.asarray([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
        ], dtype=np.float32)
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(nms(boxes, scores, 0.5), [0, 2])


if __name__ == "__main__":
    unittest.main()
