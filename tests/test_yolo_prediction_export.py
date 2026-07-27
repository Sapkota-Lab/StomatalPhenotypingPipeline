from yolo_prediction_export import owned_by_core


def test_owned_by_core_uses_global_box_center():
    assert owned_by_core(
        [40, 40, 60, 60],
        origin_x=500,
        origin_y=300,
        core=(500, 300, 600, 400),
    )
    assert not owned_by_core(
        [110, 40, 130, 60],
        origin_x=500,
        origin_y=300,
        core=(500, 300, 600, 400),
    )

