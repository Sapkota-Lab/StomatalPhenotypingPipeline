import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dev_selector import Selector


class SelectorViewportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "image.jpg"
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.imwrite(str(self.path), image)
        self.selector = Selector(
            self.path,
            Path(self.directory.name) / "output",
            max_width=300,
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_keeps_full_resolution_image_and_mask(self):
        self.assertEqual(self.selector.image.shape[:2], (400, 600))
        self.assertEqual(self.selector.selection.shape, (400, 600))
        self.assertEqual(self.selector.display_size, (300, 200))

    def test_render_uses_display_dimensions(self):
        rendered = self.selector.render()

        self.assertEqual(rendered.shape[:2], (200, 300))

    def test_zoom_maps_display_center_to_full_resolution_center(self):
        self.selector.zoom_at(150, 100, 2.0)

        self.assertAlmostEqual(self.selector.zoom, 2.0)
        self.assertEqual(self.selector.to_base(150, 100), (300, 200))


if __name__ == "__main__":
    unittest.main()
