import tempfile
import unittest
from pathlib import Path

from analyze_stomatal_area import analyze, write_figures


class StomatalAreaAnalysisTests(unittest.TestCase):
    def test_uses_plant_means_not_pooled_stomata_as_replicates(self):
        manifest = [
            {
                "study_id": "study",
                "genotype_id": "V01",
                "plant_unit_id": "V01-W",
                "variety": "A",
                "water_status": "watered",
                "water_ml": "100",
                "stomata_image": "w1.jpg",
            },
            {
                "study_id": "study",
                "genotype_id": "V01",
                "plant_unit_id": "V01-D",
                "variety": "A",
                "water_status": "drought_stressed",
                "water_ml": "0",
                "stomata_image": "d1.jpg",
            },
            {
                "study_id": "study",
                "genotype_id": "V02",
                "plant_unit_id": "V02-W",
                "variety": "B",
                "water_status": "watered",
                "water_ml": "100",
                "stomata_image": "w2.jpg",
            },
            {
                "study_id": "study",
                "genotype_id": "V02",
                "plant_unit_id": "V02-D",
                "variety": "B",
                "water_status": "drought_stressed",
                "water_ml": "0",
                "stomata_image": "d2.jpg",
            },
        ]
        stomata = [
            {
                "filename": "w1.jpg",
                "complex_area_px2": "10",
                "decision_status": "accepted",
                "measurement_quality_passed": "true",
            },
            {
                "filename": "w1.jpg",
                "complex_area_px2": "10",
                "decision_status": "accepted",
                "measurement_quality_passed": "true",
            },
            {
                "filename": "d1.jpg",
                "complex_area_px2": "20",
                "decision_status": "accepted",
                "measurement_quality_passed": "true",
            },
            {
                "filename": "w2.jpg",
                "complex_area_px2": "30",
                "decision_status": "accepted",
                "measurement_quality_passed": "true",
            },
            {
                "filename": "d2.jpg",
                "complex_area_px2": "50",
                "decision_status": "accepted",
                "measurement_quality_passed": "true",
            },
        ]

        plant, variety, treatment, report = analyze(
            manifest,
            stomata,
            area_field="complex_area_px2",
            area_unit="px2",
            status_field="decision_status",
            accepted_status="accepted",
            quality_field="measurement_quality_passed",
        )

        watered = next(
            row for row in treatment if row["water_status"] == "watered"
        )
        self.assertEqual(watered["mean_of_plant_mean_area"], 20)
        self.assertEqual(watered["n_plants"], 2)
        self.assertEqual(watered["n_stomata"], 3)
        self.assertEqual(
            report["paired_treatment_comparison"]["mean_paired_difference"],
            15,
        )
        self.assertEqual(len(plant), 4)
        self.assertEqual(len(variety), 2)

    def test_excludes_review_and_failed_measurements(self):
        manifest = [
            {
                "study_id": "study",
                "genotype_id": "V01",
                "plant_unit_id": "V01-W",
                "variety": "A",
                "water_status": "watered",
                "water_ml": "100",
                "stomata_image": "w.jpg",
            },
            {
                "study_id": "study",
                "genotype_id": "V01",
                "plant_unit_id": "V01-D",
                "variety": "A",
                "water_status": "drought_stressed",
                "water_ml": "0",
                "stomata_image": "d.jpg",
            },
        ]
        stomata = [
            {
                "filename": "w.jpg",
                "complex_area_px2": "10",
                "decision_status": "review",
                "measurement_quality_passed": "true",
            },
            {
                "filename": "d.jpg",
                "complex_area_px2": "20",
                "decision_status": "accepted",
                "measurement_quality_passed": "false",
            },
        ]
        plant, _variety, _treatment, report = analyze(
            manifest,
            stomata,
            area_field="complex_area_px2",
            area_unit="px2",
            status_field="decision_status",
            accepted_status="accepted",
            quality_field="measurement_quality_passed",
        )
        self.assertTrue(all(row["n_stomata"] == 0 for row in plant))
        self.assertEqual(
            report["object_inclusion"]["excluded_object_rows"],
            2,
        )

    def test_writes_requested_area_figures(self):
        rows = [{
            "variety": "A",
            "watered_mean_area": 100,
            "drought_mean_area": 120,
            "difference_drought_minus_watered": 20,
            "complete_pair": True,
            "area_unit": "px2",
        }]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_figures(output, rows)
            self.assertTrue((output / "paired_treatment_area.png").is_file())
            self.assertTrue((output / "paired_area_differences.png").is_file())
            self.assertTrue((output / "variety_treatment_area.png").is_file())


if __name__ == "__main__":
    unittest.main()
