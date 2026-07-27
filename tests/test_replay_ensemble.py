import tempfile
import unittest
from pathlib import Path

from replay_ensemble import load_detection_csv


class DetectionCsvTests(unittest.TestCase):
    def test_loads_and_filters_by_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.csv"
            path.write_text(
                "filename,confidence,x1,y1,x2,y2\n"
                "a.jpg,0.9,1,2,3,4\n"
                "a.jpg,0.1,5,6,7,8\n"
            )

            result = load_detection_csv(path, minimum_confidence=0.5)

        self.assertEqual(list(result), ["a.jpg"])
        self.assertEqual(len(result["a.jpg"]), 1)
        self.assertEqual(result["a.jpg"][0].confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
