import unittest

from build_combined_study_manifest import (
    build_rows,
    canonical_treatment,
    canonical_variety,
)


class CombinedStudyTests(unittest.TestCase):
    def test_only_explicit_source_labels_are_canonicalized(self):
        self.assertEqual(canonical_variety("AlisaCraig"), "Ailsa Craig")
        self.assertEqual(canonical_variety("Ailsa Craig"), "Ailsa Craig")
        with self.assertRaises(ValueError):
            canonical_variety("ailsa-craig")

    def test_only_explicit_treatment_labels_are_canonicalized(self):
        self.assertEqual(canonical_treatment("Control"), "control")
        self.assertEqual(canonical_treatment("regular"), "regular")
        with self.assertRaises(ValueError):
            canonical_treatment("drought")

    def test_regular_and_control_map_to_water_status(self):
        stomata = [
            {
                "filename": "IMG_1.JPG",
                "photo_serial": "1",
                "plant_name": "NC264G",
                "treatment": "regular",
                "mapping_status": "mapped",
            },
            {
                "filename": "IMG_2.JPG",
                "photo_serial": "2",
                "plant_name": "NC264G",
                "treatment": "control",
                "mapping_status": "mapped",
            },
        ]
        thermal = [
            {
                "session_id": "S20260715",
                "plant_id": "P02",
                "variety": "NC264G",
                "treatment": "Regular",
                "group_type": "breeding_line",
                "source_file": "regular.JPG",
                "capture_time": "",
                "mean_c": "25",
                "median_c": "25",
                "min_c": "24",
                "max_c": "26",
                "std_c": "0.3",
                "p10_c": "24.5",
                "p90_c": "25.5",
                "delta_t_air": "0",
                "pixel_count": "6000",
            },
            {
                "session_id": "S20260715",
                "plant_id": "P01",
                "variety": "NC264G",
                "treatment": "Control",
                "group_type": "breeding_line",
                "source_file": "control.JPG",
                "capture_time": "",
                "mean_c": "26",
                "median_c": "26",
                "min_c": "25",
                "max_c": "27",
                "std_c": "0.3",
                "p10_c": "25.5",
                "p90_c": "26.5",
                "delta_t_air": "1",
                "pixel_count": "7000",
            },
        ]

        rows = build_rows(stomata, thermal, require_complete=False)
        by_status = {row["water_status"]: row for row in rows}
        self.assertEqual(by_status["watered"]["water_ml"], 100)
        self.assertEqual(by_status["drought_stressed"]["water_ml"], 0)
        self.assertEqual(
            by_status["watered"]["plant_unit_id"],
            "TOM26-V01-W",
        )
        self.assertEqual(
            by_status["drought_stressed"]["plant_unit_id"],
            "TOM26-V01-D",
        )
        self.assertEqual(
            by_status["watered"]["join_method"],
            "exact_canonical_variety_and_treatment",
        )

    def test_join_requires_both_variety_and_treatment(self):
        stomata = [{
            "filename": "IMG_1.JPG",
            "photo_serial": "1",
            "plant_name": "NC264G",
            "treatment": "regular",
            "mapping_status": "mapped",
        }]
        thermal = [{
            "session_id": "S20260715",
            "plant_id": "P01",
            "variety": "NC264G",
            "treatment": "Control",
            "group_type": "breeding_line",
            "source_file": "control.JPG",
            "capture_time": "",
            "mean_c": "26",
            "median_c": "26",
            "min_c": "25",
            "max_c": "27",
            "std_c": "0.3",
            "p10_c": "25.5",
            "p90_c": "26.5",
            "delta_t_air": "1",
            "pixel_count": "7000",
        }]
        with self.assertRaisesRegex(
            ValueError,
            "No July 15 thermal record for NC264G / regular",
        ):
            build_rows(stomata, thermal, require_complete=False)


if __name__ == "__main__":
    unittest.main()
