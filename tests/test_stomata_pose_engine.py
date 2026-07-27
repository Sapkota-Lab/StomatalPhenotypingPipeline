import numpy as np

from stomata_pose_engine import (
    MorphologyThresholds,
    box_iou,
    is_morphology_plausible,
    make_detection,
    morphology_measurements,
    nms,
    preprocess_image,
    tile_starts,
)


def test_morphology_measurements_for_centered_perpendicular_cross():
    keypoints = np.array(
        [
            [10.0, 30.0, 0.90],
            [50.0, 30.0, 0.95],
            [30.0, 20.0, 0.80],
            [30.0, 40.0, 0.85],
        ]
    )
    measured = morphology_measurements(keypoints, [8, 18, 52, 42])
    assert measured["length_px"] == 40.0
    assert measured["width_px"] == 20.0
    assert measured["length_width_ratio"] == 2.0
    assert measured["perpendicular_deviation_deg"] == 0.0
    assert measured["center_offset_norm"] == 0.0
    assert measured["keypoints_inside_fraction"] == 1.0


def test_morphology_plausibility_rejects_round_degenerate_geometry():
    thresholds = MorphologyThresholds()
    plausible = {
        "keypoint_min_conf": 0.9,
        "length_width_ratio": 1.4,
        "perpendicular_deviation_deg": 4.0,
        "center_offset_norm": 0.05,
        "keypoints_inside_fraction": 1.0,
    }
    assert is_morphology_plausible(plausible, thresholds)
    assert not is_morphology_plausible(
        {**plausible, "perpendicular_deviation_deg": 45.0},
        thresholds,
    )
    assert not is_morphology_plausible(
        {**plausible, "center_offset_norm": 0.4},
        thresholds,
    )


def test_nms_keeps_high_confidence_duplicate():
    thresholds = MorphologyThresholds()
    keypoints = np.array(
        [[10, 20, 0.9], [30, 20, 0.9], [20, 14, 0.9], [20, 26, 0.9]],
        dtype=float,
    )
    high = make_detection([5, 10, 35, 30], 0.9, keypoints, thresholds)
    low = make_detection([6, 11, 36, 31], 0.5, keypoints + [1, 1, 0], thresholds)
    far = make_detection([100, 100, 130, 120], 0.7, keypoints + [95, 90, 0], thresholds)
    assert box_iou(high, low) > 0.5
    kept = nms([low, far, high], 0.5)
    assert kept == [high, far]


def test_tile_starts_cover_final_edge():
    assert tile_starts(1000, 1280, 0.2) == [0]
    starts = tile_starts(3000, 1280, 0.2)
    assert starts[0] == 0
    assert starts[-1] == 1720


def test_preprocessing_preserves_shape_and_none_is_identity():
    image = np.full((32, 40, 3), 100, dtype=np.uint8)
    assert preprocess_image(image, "none") is image
    assert preprocess_image(image, "clahe").shape == image.shape
    assert preprocess_image(image, "gray-clahe").shape == image.shape
