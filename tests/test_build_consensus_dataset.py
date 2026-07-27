import unittest

from build_consensus_dataset import (
    centered_window,
    clip_box,
    stable_split,
    yolo_line,
)


class ConsensusDatasetTests(unittest.TestCase):
    def test_split_is_stable_for_source_image(self):
        first = stable_split("IMG_3012.JPG", 0.2, "seed")
        second = stable_split("IMG_3012.JPG", 0.2, "seed")

        self.assertEqual(first, second)
        self.assertIn(first, {"train", "val"})

    def test_window_stays_inside_image(self):
        self.assertEqual(
            centered_window(10, 20, 100, 600, 400),
            (0, 0, 100, 100),
        )
        self.assertEqual(
            centered_window(590, 390, 100, 600, 400),
            (500, 300, 600, 400),
        )

    def test_box_is_included_when_center_is_in_window(self):
        self.assertEqual(
            clip_box((90, 90, 130, 130), (100, 100, 200, 200)),
            (0, 0, 30, 30),
        )

    def test_box_is_excluded_when_center_is_outside_window(self):
        self.assertIsNone(
            clip_box((80, 80, 100, 100), (100, 100, 200, 200))
        )

    def test_yolo_coordinates_are_normalized(self):
        self.assertEqual(
            yolo_line((25, 25, 75, 75), 100, 100),
            "0 0.50000000 0.50000000 0.50000000 0.50000000",
        )


if __name__ == "__main__":
    unittest.main()
