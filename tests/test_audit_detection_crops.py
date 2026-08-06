import unittest
import tempfile
from pathlib import Path

import numpy as np

from audit_detection_crops import crop_detection, load_rows


class LedgerCompatibilityTests(unittest.TestCase):
    def test_load_rows_filters_water_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.csv"
            path.write_text(
                "decision_status,proposal_source,water_status\n"
                "accepted,expert_box_annotation,watered\n"
                "accepted,expert_box_annotation,drought_stressed\n"
            )
            rows = load_rows(
                path,
                status="accepted",
                source="expert_box_annotation",
                water_status="drought_stressed",
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["water_status"], "drought_stressed")

    def test_crop_supports_ledger_coordinate_fields(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_detection(
            image,
            {
                "bbox_x1_px": "20",
                "bbox_y1_px": "30",
                "bbox_x2_px": "40",
                "bbox_y2_px": "50",
            },
            padding=0,
        )
        self.assertEqual(crop.shape, (20, 20, 3))


if __name__ == "__main__":
    unittest.main()
