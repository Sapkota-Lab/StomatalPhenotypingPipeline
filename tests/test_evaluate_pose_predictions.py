import numpy as np

from evaluate_pose_predictions import (
    GroundTruth,
    Prediction,
    box_iou,
    evaluate,
    pair_invariant_keypoint_errors,
)


def prediction(image, box, confidence=0.9, plausible=True):
    keypoints = np.array(
        [[0, 0, 1], [10, 0, 1], [5, -5, 1], [5, 5, 1]],
        dtype=float,
    )
    return Prediction(image, np.asarray(box, dtype=float), confidence, keypoints, plausible)


def truth(box):
    keypoints = np.array(
        [[0, 0, 2], [10, 0, 2], [5, -5, 2], [5, 5, 2]],
        dtype=float,
    )
    return GroundTruth(np.asarray(box, dtype=float), keypoints)


def test_box_iou():
    assert box_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert box_iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0


def test_keypoint_error_is_invariant_to_axis_endpoint_order():
    expected = truth([0, 0, 10, 10]).keypoints
    predicted = expected[[1, 0, 3, 2]].copy()
    assert np.allclose(pair_invariant_keypoint_errors(predicted, expected), 0.0)


def test_evaluate_reports_tp_fp_fn_and_count_error():
    ground_truth = {"a.jpg": [truth([0, 0, 10, 10]), truth([20, 20, 30, 30])]}
    predictions = [
        prediction("a.jpg", [0, 0, 10, 10], 0.9),
        prediction("a.jpg", [40, 40, 50, 50], 0.8),
    ]
    result = evaluate(
        ground_truth,
        predictions,
        confidence=0.0,
        morphology_only=False,
    )
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["count_mae"] == 0.0


def test_morphology_filter_excludes_review_predictions():
    ground_truth = {"a.jpg": [truth([0, 0, 10, 10])]}
    predictions = [prediction("a.jpg", [0, 0, 10, 10], plausible=False)]
    result = evaluate(
        ground_truth,
        predictions,
        confidence=0.0,
        morphology_only=True,
    )
    assert result["predictions"] == 0
    assert result["fn"] == 1
