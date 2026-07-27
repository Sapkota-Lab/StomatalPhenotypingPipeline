import unittest
import tempfile
from pathlib import Path

from ensemble_stomata_engine import (
    COUNT_FIELDS,
    DETECTION_FIELDS,
    classify_ensemble,
    load_checkpoint,
    load_scope_manifest,
    write_csv_atomic,
)
from stomata_engine import Detection


class EnsembleClassificationTests(unittest.TestCase):
    def test_single_model_detection_is_review_only(self):
        stomaquant = [Detection(10, 10, 30, 40, 0.30)]

        result = classify_ensemble(stomaquant, [])

        self.assertEqual(result[0].status, "review")
        self.assertEqual(result[0].source, "stomaquant_only")

    def test_low_stomaquant_detection_is_promoted_by_agreement(self):
        stomaquant = [Detection(10, 10, 30, 40, 0.08)]
        yolos = [Detection(11, 11, 31, 41, 0.95)]

        result = classify_ensemble(stomaquant, yolos)

        self.assertEqual(result[0].status, "accepted")
        self.assertEqual(result[0].source, "stomaquant_yolos")
        self.assertEqual(result[0].yolos_confidence, 0.95)

    def test_low_confidence_yolos_does_not_promote(self):
        stomaquant = [Detection(10, 10, 30, 40, 0.08)]
        yolos = [Detection(11, 11, 31, 41, 0.50)]

        result = classify_ensemble(stomaquant, yolos)

        self.assertEqual(result[0].status, "review")
        self.assertEqual(result[0].source, "stomaquant_only")

    def test_high_confidence_unmatched_yolos_is_review_only(self):
        stomaquant = [Detection(10, 10, 30, 40, 0.30)]
        yolos = [Detection(100, 100, 130, 150, 0.99)]

        result = classify_ensemble(stomaquant, yolos)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].status, "review")
        self.assertEqual(result[1].source, "yolos_only")

    def test_multiple_yolos_matches_do_not_duplicate_count(self):
        stomaquant = [Detection(10, 10, 30, 40, 0.08)]
        yolos = [
            Detection(11, 11, 31, 41, 0.90),
            Detection(9, 9, 29, 39, 0.95),
        ]

        result = classify_ensemble(stomaquant, yolos)

        accepted = [item for item in result if item.status == "accepted"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].yolos_confidence, 0.95)

    def test_stomadet_and_yolos_agreement_adds_detection(self):
        yolos = [Detection(100, 100, 140, 180, 0.95)]
        stomadet = [Detection(102, 102, 142, 182, 0.60)]

        result = classify_ensemble([], yolos, stomadet)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, "accepted")
        self.assertEqual(result[0].source, "stomadet_yolos")
        self.assertEqual(result[0].yolos_confidence, 0.95)
        self.assertEqual(result[0].stomadet_confidence, 0.60)

    def test_stomadet_alone_does_not_add_detection(self):
        stomadet = [Detection(100, 100, 140, 180, 0.99)]

        result = classify_ensemble([], [], stomadet)

        self.assertEqual(result, [])

    def test_stomadet_pair_uses_stricter_yolos_threshold(self):
        yolos = [Detection(100, 100, 140, 180, 0.90)]
        stomadet = [Detection(102, 102, 142, 182, 0.99)]

        result = classify_ensemble([], yolos, stomadet)

        self.assertEqual(result, [])

    def test_third_model_does_not_duplicate_existing_detection(self):
        stomaquant = [Detection(100, 100, 140, 180, 0.30)]
        yolos = [Detection(102, 102, 142, 182, 0.95)]
        stomadet = [Detection(101, 101, 141, 181, 0.60)]

        result = classify_ensemble(stomaquant, yolos, stomadet)

        accepted = [item for item in result if item.status == "accepted"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].source, "stomaquant_yolos")


class CheckpointTests(unittest.TestCase):
    def test_compatible_checkpoint_round_trips(self):
        count = {field: "value" for field in COUNT_FIELDS}
        count["filename"] = "image.jpg"
        detection = {field: "value" for field in DETECTION_FIELDS}
        detection["filename"] = "image.jpg"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_csv_atomic(output / "counts.csv", COUNT_FIELDS, [count])
            write_csv_atomic(
                output / "detections.csv",
                DETECTION_FIELDS,
                [detection],
            )

            counts, detections = load_checkpoint(output)

        self.assertEqual(counts, [count])
        self.assertEqual(detections, [detection])

    def test_incompatible_checkpoint_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_csv_atomic(output / "counts.csv", ["wrong"], [{"wrong": "x"}])
            write_csv_atomic(output / "detections.csv", ["wrong"], [{"wrong": "x"}])

            counts, detections = load_checkpoint(output)

        self.assertEqual(counts, [])
        self.assertEqual(detections, [])


class ScopeManifestTests(unittest.TestCase):
    def test_valid_manifest_loads_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "filename,analysis_scope\n"
                "leaf.jpg,eligible\n"
                "fungus.jpg,out_of_scope\n"
            )

            scopes = load_scope_manifest(path)

        self.assertEqual(
            scopes,
            {"leaf.jpg": "eligible", "fungus.jpg": "out_of_scope"},
        )

    def test_invalid_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("filename,analysis_scope\nimage.jpg,maybe\n")

            with self.assertRaises(ValueError):
                load_scope_manifest(path)


if __name__ == "__main__":
    unittest.main()
