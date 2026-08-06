import unittest

from build_integrated_product_dataset import build_products


def plant(image, variety, treatment):
    return {
        "study_id": "study",
        "join_method": "exact_canonical_variety_and_treatment",
        "join_key_variety": variety,
        "join_key_treatment": treatment,
        "stomata_source_variety": variety,
        "stomata_source_treatment": treatment,
        "thermal_source_variety": variety,
        "thermal_source_treatment": treatment,
        "genotype_id": "V01",
        "plant_unit_id": f"{variety}-{treatment}",
        "variety": variety,
        "variety_class": "line",
        "treatment_code": treatment,
        "water_status": "watered",
        "water_ml": "100",
        "stomata_image": image,
        "stomata_photo_serial": "1",
        "thermal_session_id": "S1",
        "thermal_session_plant_id": "P1",
        "thermal_source_file": "thermal.jpg",
    }


class IntegratedProductTests(unittest.TestCase):
    def test_duplicate_composite_key_fails(self):
        plants = [
            plant("A.JPG", "NC264G", "regular"),
            plant("B.JPG", "NC264G", "regular"),
        ]
        with self.assertRaisesRegex(
            ValueError,
            r"Duplicate canonical variety\+treatment key",
        ):
            build_products(plants, [], [])

    def test_unmatched_detection_fails(self):
        plants = [
            plant(
                f"IMG_{index:04d}_{treatment}.JPG",
                f"V{index:02d}",
                treatment,
            )
            for index in range(1, 17)
            for treatment in ("control", "regular")
        ]
        detection = {
            "filename": "UNKNOWN.JPG",
            "detection_id": "1",
            "status": "accepted",
            "confidence": "0.9",
            "raw_detector_score": "0.9",
            "p_stoma": "0.95",
            "decision_status": "accepted",
            "x1": "1",
            "y1": "1",
            "x2": "2",
            "y2": "2",
        }
        with self.assertRaisesRegex(ValueError, "Unmatched detection image"):
            build_products(plants, [detection], [])


if __name__ == "__main__":
    unittest.main()
