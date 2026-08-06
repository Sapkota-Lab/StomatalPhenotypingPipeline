import unittest

from build_stomata_ledger import (
    add_spatial_metrics,
    axis_aligned_edge_distance,
)


def object_row(x1, y1, x2, y2):
    return {
        "center_x_px": (x1 + x2) / 2,
        "center_y_px": (y1 + y2) / 2,
        "bbox_x1_px": x1,
        "bbox_y1_px": y1,
        "bbox_x2_px": x2,
        "bbox_y2_px": y2,
        "bbox_width_px": x2 - x1,
        "bbox_height_px": y2 - y1,
    }


class SpatialMetricTests(unittest.TestCase):
    def test_axis_aligned_edge_distance(self):
        self.assertEqual(
            axis_aligned_edge_distance((0, 0, 10, 10), (13, 14, 20, 20)),
            5.0,
        )
        self.assertEqual(
            axis_aligned_edge_distance((0, 0, 10, 10), (5, 5, 20, 20)),
            0.0,
        )

    def test_nearest_neighbor_and_bearing(self):
        rows = [
            object_row(0, 0, 10, 10),
            object_row(20, 0, 30, 10),
            object_row(0, 40, 10, 50),
        ]
        add_spatial_metrics(rows)
        self.assertEqual(rows[0]["nearest_neighbor_center_distance_px"], 20.0)
        self.assertEqual(rows[0]["nearest_neighbor_bearing_deg"], 0.0)
        self.assertEqual(rows[0]["nearest_neighbor_edge_distance_px"], 10.0)

    def test_single_object_has_no_nearest_neighbor(self):
        rows = [object_row(0, 0, 10, 10)]
        add_spatial_metrics(rows)
        self.assertEqual(rows[0]["nearest_neighbor_center_distance_px"], "")
        self.assertEqual(rows[0]["neighbors_within_radius"], 0)


if __name__ == "__main__":
    unittest.main()
