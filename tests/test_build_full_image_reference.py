import json
import tempfile
import unittest
from pathlib import Path

from build_full_image_reference import build_references


class FullImageReferenceTests(unittest.TestCase):
    def test_builds_unique_source_annotations_by_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, count in (("a.jpg", 2), ("b.jpg", 1)):
                payload = {
                    "image": filename,
                    "width": 100,
                    "height": 80,
                    "status": "complete",
                    "boxes": [
                        {
                            "x1": 10 + index * 20,
                            "y1": 10,
                            "x2": 20 + index * 20,
                            "y2": 30,
                        }
                        for index in range(count)
                    ],
                }
                (root / f"{Path(filename).stem}.json").write_text(
                    json.dumps(payload)
                )
            references = build_references(
                root,
                {"a.jpg": "val", "b.jpg": "test"},
            )
            self.assertEqual(len(references["val"]["images"]), 1)
            self.assertEqual(len(references["val"]["annotations"]), 2)
            self.assertEqual(len(references["test"]["annotations"]), 1)


if __name__ == "__main__":
    unittest.main()
