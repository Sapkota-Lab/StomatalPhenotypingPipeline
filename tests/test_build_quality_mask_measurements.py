import unittest
from pathlib import Path

from build_quality_mask_measurements import (
    additional_polygon_traits,
    select_measurements,
)


class QualityMaskMeasurementTests(unittest.TestCase):
    def test_polygon_traits_include_centroid_and_circularity(self):
        traits = additional_polygon_traits(
            [[0, 0], [20, 0], [20, 10], [0, 10]]
        )
        self.assertEqual(traits["complex_convex_area_px2"], 200)
        self.assertEqual(traits["complex_centroid_x_px"], 10)
        self.assertEqual(traits["complex_centroid_y_px"], 5)
        self.assertGreater(traits["complex_circularity"], 0)

    def test_selects_at_most_three_completed_masks_per_image(self):
        manifest = [{
            "stomata_image": "A.JPG",
            "study_id": "study",
            "plant_unit_id": "A-W",
            "genotype_id": "V01",
            "variety": "A",
            "water_status": "watered",
            "water_ml": "100",
        }]
        candidates = [{
            "object_id": f"A:{index:04d}",
            "image": "A.JPG",
            "plant_unit_id": "A-W",
            "genotype_id": "V01",
            "variety": "A",
            "water_status": "watered",
            "expert_box_index": index,
            "quality_rank_within_image": index,
            "local_quality_score": 10 - index,
            "local_focus_score": 10 - index,
            "local_contrast_5_95": 50,
            "local_clipped_fraction": 0,
            "edge_margin_px": 100,
        } for index in range(1, 5)]
        polygon = [[0, 0], [20, 0], [20, 10], [0, 10]]
        annotations = {
            item["object_id"]: ({
                "status": "complete",
                "polygon": polygon,
                "reviewer": "tester",
            }, Path(f"{item['object_id']}.json"))
            for item in candidates
        }
        rows, report = select_measurements(
            manifest,
            candidates,
            annotations,
            max_per_image=3,
            min_per_image=2,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["expert_box_index"] for row in rows],
            [1, 2, 3],
        )
        self.assertEqual(report["coverage_by_image"]["A.JPG"], {
            "usable_completed_masks": 4,
            "selected_masks": 3,
        })


if __name__ == "__main__":
    unittest.main()
