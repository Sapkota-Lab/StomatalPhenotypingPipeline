import unittest

import numpy as np

from propose_complex_masks import (
    box_iou,
    candidate_rank,
    crop_bounds,
    select_prompted_mask,
)


class ProposalGeometryTests(unittest.TestCase):
    def test_identical_boxes_have_unit_iou(self):
        box = np.asarray([1, 2, 11, 12], dtype=float)
        self.assertEqual(box_iou(box, box), 1)

    def test_centered_candidate_ranks_above_displaced_candidate(self):
        expert = np.asarray([10, 10, 30, 30], dtype=float)
        centered = np.asarray([11, 11, 29, 29], dtype=float)
        displaced = np.asarray([25, 25, 45, 45], dtype=float)
        self.assertGreater(
            candidate_rank(centered, expert, 0.8)[0],
            candidate_rank(displaced, expert, 0.8)[0],
        )

    def test_crop_is_clipped_to_image(self):
        box = {"x1": 1, "y1": 2, "x2": 11, "y2": 12}
        self.assertEqual(crop_bounds(box, 20, 20, 1), (0, 0, 20, 20))

    def test_prompted_mask_uses_polygon_bounds(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def __float__(self):
                return float(self.value)

        class Boxes:
            conf = [Tensor(0.9)]

            def __len__(self):
                return 1

        class Masks:
            data = [
                Tensor(np.pad(
                    np.ones((20, 20)),
                    ((10, 10), (10, 10)),
                ))
            ]

        result = type("Result", (), {"masks": Masks(), "boxes": Boxes()})()
        selected = select_prompted_mask(
            result,
            np.asarray([10, 10, 30, 30], dtype=float),
        )
        self.assertGreater(selected["box_iou_with_expert"], 0.9)


if __name__ == "__main__":
    unittest.main()
