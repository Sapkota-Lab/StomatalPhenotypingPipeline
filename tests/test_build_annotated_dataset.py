import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_annotated_dataset import (
    assign_group_splits,
    boxes_in_window,
    build_dataset,
    grid_origins,
    load_annotations,
    load_metadata,
    yolo_line,
)


class GroupSplitTests(unittest.TestCase):
    def test_group_members_cannot_cross_splits(self):
        groups = ["plant_a", "plant_a", "plant_b", "plant_c", "plant_d"]
        assignment = assign_group_splits(groups, 0.2, 0.2, 7)
        self.assertEqual(set(assignment), set(groups))
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})

    def test_split_is_deterministic(self):
        groups = ["a", "b", "c", "d", "e"]
        self.assertEqual(
            assign_group_splits(groups, 0.2, 0.2, 99),
            assign_group_splits(list(reversed(groups)), 0.2, 0.2, 99),
        )

    def test_requires_three_independent_groups(self):
        with self.assertRaises(ValueError):
            assign_group_splits(["a", "b"], 0.2, 0.2, 1)


class TileTests(unittest.TestCase):
    def test_grid_reaches_far_edge(self):
        origins = grid_origins(2500, 1024, 0.2)
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 1476)

    def test_box_center_controls_window_membership(self):
        boxes = ((90.0, 90.0, 130.0, 130.0),)
        self.assertEqual(len(boxes_in_window(boxes, 0, 0, 120, 120)), 1)
        self.assertEqual(len(boxes_in_window(boxes, 120, 0, 120, 120)), 0)

    def test_yolo_coordinates_are_normalized(self):
        values = yolo_line((20, 30, 60, 70), 0, 0, 100, 100).split()
        self.assertEqual(values[0], "0")
        self.assertEqual([float(value) for value in values[1:]], [0.4, 0.5, 0.4, 0.4])


class DatasetIntegrationTests(unittest.TestCase):
    def test_out_of_scope_metadata_rows_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations"
            annotations.mkdir()
            metadata = root / "metadata.csv"
            metadata.write_text(
                "filename,specimen_id,mapping_status\n"
                "mapped.jpg,plant_a,mapped\n"
                "extra.jpg,,not_in_workbook\n"
            )
            payload = {
                "width": 100,
                "height": 100,
                "status": "complete",
                "boxes": [{"x1": 10, "y1": 10, "x2": 20, "y2": 20}],
            }
            for filename in ("mapped.jpg", "extra.jpg"):
                (annotations / f"{Path(filename).stem}.json").write_text(
                    json.dumps(payload | {"image": filename})
                )
            rows = load_annotations(
                annotations,
                load_metadata(metadata, "specimen_id"),
                allow_image_groups=False,
            )
            self.assertEqual([row.filename for row in rows], ["mapped.jpg"])

    def test_builds_non_leaking_three_way_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            annotations = root / "annotations"
            images.mkdir()
            annotations.mkdir()
            metadata = root / "metadata.csv"
            metadata.write_text(
                "filename,specimen_id\n"
                "one.jpg,plant_a\n"
                "two.jpg,plant_b\n"
                "three.jpg,plant_c\n"
            )
            for filename in ("one.jpg", "two.jpg", "three.jpg"):
                Image.new("RGB", (200, 200), "white").save(images / filename)
                payload = {
                    "image": filename,
                    "width": 200,
                    "height": 200,
                    "status": "complete",
                    "boxes": [{"x1": 80, "y1": 80, "x2": 120, "y2": 120}],
                }
                (annotations / f"{Path(filename).stem}.json").write_text(
                    json.dumps(payload)
                )
            output = root / "dataset"
            args = argparse.Namespace(
                images=images,
                annotations=annotations,
                metadata=metadata,
                group_column="specimen_id",
                output=output,
                tile_size=128,
                overlap=0.2,
                negative_ratio=0.0,
                zero_image_negatives=1,
                val_fraction=0.2,
                test_fraction=0.2,
                seed=7,
                allow_image_groups=False,
                include_provisional=False,
            )
            summary = build_dataset(args)
            self.assertEqual(summary["source_images"], 3)
            self.assertEqual(summary["stomata"], 3)
            self.assertEqual(
                {
                    path.parent.name
                    for path in (output / "images").glob("*/*.jpg")
                },
                {"train", "val", "test"},
            )
            manifest = (output / "split_manifest.csv").read_text()
            self.assertIn("plant_a", manifest)
            self.assertTrue((output / "dataset.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
