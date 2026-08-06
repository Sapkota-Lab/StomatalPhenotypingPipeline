import unittest

from build_combined_phenotypes import build_tables


class CombinedPhenotypeTests(unittest.TestCase):
    def test_joins_thermal_detection_and_area_by_plant(self):
        manifest = []
        area_plants = []
        area_varieties = [{
            "genotype_id": "V01",
            "variety": "A",
            "complete_pair": "True",
        }]
        for status, suffix, water, image in (
            ("watered", "W", "100", "w.jpg"),
            ("drought_stressed", "D", "0", "d.jpg"),
        ):
            manifest.append({
                "study_id": "study",
                "genotype_id": "V01",
                "plant_unit_id": f"V01-{suffix}",
                "variety": "A",
                "water_status": status,
                "water_ml": water,
                "stomata_image": image,
                "thermal_mask_qc_passed": "True",
                "canopy_mean_c": "25",
                "canopy_delta_t_air_c": "1",
                "projected_plant_area_px2": "100",
                "projected_plant_area_fraction": "0.1",
            })
            area_plants.append({
                "plant_unit_id": f"V01-{suffix}",
                "area_unit": "px2",
                "mean_area": "200",
                "median_area": "190",
                "area_sd": "10",
                "area_q25": "185",
                "area_q75": "205",
            })
        stomata = [{
            "stomata_image": "w.jpg",
            "measurement_quality_passed": "True",
        }]
        predictions = [{
            "stomata_image": "w.jpg",
            "decision_status": "accepted",
            "p_stoma": "0.9",
        }]
        plants, varieties, treatments = build_tables(
            manifest,
            area_plants,
            area_varieties,
            [
                {"water_status": "watered"},
                {"water_status": "drought_stressed"},
            ],
            stomata,
            predictions,
        )
        watered = next(
            row for row in plants if row["water_status"] == "watered"
        )
        self.assertEqual(watered["stomatal_complex_area_mean"], 200)
        self.assertEqual(watered["stomata_candidates_accepted"], 1)
        self.assertEqual(watered["stomata_objects_measured"], 1)
        self.assertEqual(len(varieties), 1)
        self.assertEqual(len(treatments), 2)


if __name__ == "__main__":
    unittest.main()
