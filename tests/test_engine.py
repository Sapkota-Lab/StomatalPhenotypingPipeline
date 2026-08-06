import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from stomata_engine import (
    Detection,
    global_nms,
    image_paths,
    image_quality,
    split_filenames,
    tile_regions,
    tile_starts,
)


class TileTests(unittest.TestCase):
    def test_small_image_has_one_tile(self):
        self.assertEqual(tile_starts(600, 1216, 0.2), [0])

    def test_last_tile_reaches_image_edge(self):
        starts = tile_starts(6000, 1216, 0.2)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + 1216, 6000)

    def test_tile_cores_partition_image_width(self):
        regions = list(tile_regions(3000, 1000, 1216, 0.2))
        first_row = [region for region in regions if region[1] == 0]
        self.assertEqual(first_row[0][2], 0)
        self.assertEqual(first_row[-1][4], 3000)
        for left_region, right_region in zip(first_row, first_row[1:]):
            self.assertEqual(left_region[4], right_region[2])


class NmsTests(unittest.TestCase):
    def test_duplicate_boxes_keep_higher_confidence(self):
        detections = [
            Detection(10, 10, 50, 50, 0.9),
            Detection(12, 12, 51, 51, 0.6),
            Detection(100, 100, 140, 140, 0.7),
        ]
        kept = global_nms(detections, 0.45)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0].confidence, 0.9)
        np.testing.assert_allclose(kept[1].box, (100, 100, 140, 140))

    def test_center_distance_suppresses_different_size_duplicate(self):
        detections = [
            Detection(10, 10, 50, 50, 0.9),
            Detection(0, 0, 60, 60, 0.7),
        ]
        kept = global_nms(
            detections,
            threshold=0.8,
            center_distance_threshold=0.25,
        )
        self.assertEqual(len(kept), 1)


class SplitManifestTests(unittest.TestCase):
    def test_filters_input_to_requested_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("a.jpg", "b.jpg"):
                Image.new("RGB", (10, 10)).save(root / filename)
            manifest = root / "split.csv"
            manifest.write_text(
                "filename,split\n"
                "a.jpg,val\n"
                "b.jpg,test\n"
            )
            allowed = split_filenames(manifest, "test")
            self.assertEqual(
                [path.name for path in image_paths(root, allowed)],
                ["b.jpg"],
            )
            self.assertEqual(
                split_filenames(manifest, "all"),
                {"a.jpg", "b.jpg"},
            )


class QualityTests(unittest.TestCase):
    def test_flat_image_is_low_quality(self):
        image = np.full((300, 400, 3), 128, dtype=np.uint8)
        quality = image_quality(image)
        self.assertEqual(quality["image_quality"], "low")
        self.assertEqual(quality["focus_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
