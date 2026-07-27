import unittest
from unittest.mock import patch

import numpy as np

from stomadet_engine import infer_stomadet_tiled
from stomata_engine import Detection


class TiledStomaDetTests(unittest.TestCase):
    def test_zero_tile_size_uses_full_image(self):
        image = np.zeros((100, 180, 3), dtype=np.uint8)
        expected = [Detection(1, 2, 3, 4, 0.9)]
        with patch("stomadet_engine.infer_stomadet", return_value=expected) as infer:
            result = infer_stomadet_tiled(None, image, 0, 0.2, 0.45)
        self.assertEqual(result, expected)
        infer.assert_called_once()

    def test_tile_coordinates_return_to_full_image(self):
        image = np.zeros((100, 180, 3), dtype=np.uint8)

        def fake_infer(_predictor, tile):
            if tile.shape[1] == 100:
                return [Detection(10, 20, 30, 40, 0.9)]
            return []

        with patch("stomadet_engine.infer_stomadet", side_effect=fake_infer):
            result = infer_stomadet_tiled(None, image, 100, 0.2, 0.45)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].box, (90, 20, 110, 40))
        self.assertEqual(result[1].box, (10, 20, 30, 40))


if __name__ == "__main__":
    unittest.main()
