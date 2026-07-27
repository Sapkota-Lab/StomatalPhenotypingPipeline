import unittest

import numpy as np

from morphology_verifier import (
    PATCH_SIZE,
    box_iou,
    expanded_crop,
    normalize_patch,
    patch_features,
    transform_patch,
)


class MorphologyVerifierTests(unittest.TestCase):
    def test_box_iou(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_expanded_crop_clips_to_image(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        crop = expanded_crop(image, (0, 0, 10, 10), scale=2.0)
        self.assertGreater(crop.size, 0)
        self.assertLessEqual(crop.shape[0], image.shape[0])
        self.assertLessEqual(crop.shape[1], image.shape[1])

    def test_features_have_stable_shape_for_all_views(self):
        crop = np.arange(30 * 50, dtype=np.uint8).reshape(30, 50)
        patch = normalize_patch(crop)
        self.assertEqual(patch.shape, (PATCH_SIZE, PATCH_SIZE))
        shapes = {
            patch_features(transform_patch(patch, view)).shape
            for view in (
                "identity",
                "rotate90",
                "rotate180",
                "rotate270",
                "hflip",
            )
        }
        self.assertEqual(len(shapes), 1)


if __name__ == "__main__":
    unittest.main()
