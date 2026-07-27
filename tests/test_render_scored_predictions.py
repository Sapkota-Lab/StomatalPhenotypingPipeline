import tempfile
import unittest
from pathlib import Path

from render_scored_predictions import load_scored


class RenderScoredPredictionsTests(unittest.TestCase):
    def test_threshold_splits_accepted_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            path.write_text(
                "filename,x1,y1,x2,y2,pore_score\n"
                "image.jpg,0,0,10,10,0.8\n"
                "image.jpg,20,20,30,30,0.2\n"
            )
            accepted, review = load_scored(path, "pore_score", 0.6)["image.jpg"]
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(review), 1)


if __name__ == "__main__":
    unittest.main()
