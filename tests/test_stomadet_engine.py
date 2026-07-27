import unittest

from stomata_engine import Detection
from stomadet_engine import infer_stomadet


class _Boxes:
    def __init__(self, values):
        self.tensor = values


class _Instances:
    def __init__(self, torch):
        self.pred_boxes = _Boxes(torch.tensor([
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
        ]))
        self.scores = torch.tensor([0.9, 0.8])
        self.pred_classes = torch.tensor([0, 1])

    def to(self, _device):
        return self


class StomaDetInferenceTests(unittest.TestCase):
    def test_only_stomata_class_is_returned(self):
        import torch

        class Predictor:
            def __call__(self, _image):
                return {"instances": _Instances(torch)}

        detections = infer_stomadet(
            Predictor(),
            torch.zeros((10, 10, 3)).numpy(),
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box, (10.0, 20.0, 30.0, 40.0))
        self.assertAlmostEqual(detections[0].confidence, 0.9)
