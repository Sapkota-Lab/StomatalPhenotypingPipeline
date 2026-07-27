from evaluate_external_predictions import Box
from calibrate_external_threshold import select_f1_threshold


def test_select_f1_threshold_prefers_prefix_with_best_f1():
    truth = {"a.jpg": [Box(0, 0, 10, 10), Box(20, 20, 30, 30)]}
    predictions = {
        "a.jpg": [
            Box(0, 0, 10, 10, 0.9),
            Box(40, 40, 50, 50, 0.8),
            Box(20, 20, 30, 30, 0.7),
            Box(60, 60, 70, 70, 0.1),
        ]
    }
    assert select_f1_threshold(predictions, truth) == 0.7
