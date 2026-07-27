import json

from build_external_domain_dataset import (
    load_coco,
    tile_regions,
    tile_yolo_labels,
    yolo_labels,
)


def test_yolo_labels_normalize_coco_box():
    image = {"width": 200, "height": 100}
    labels = yolo_labels(image, [{"bbox": [20, 10, 40, 20]}])
    assert labels == ["0 0.20000000 0.20000000 0.20000000 0.20000000"]


def test_load_coco_groups_annotations(tmp_path):
    document = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 2, "image_id": 1, "bbox": [1, 2, 3, 4]}],
    }
    (tmp_path / "coco.json").write_text(json.dumps(document))
    images, annotations = load_coco(tmp_path)
    assert images[1]["file_name"] == "a.jpg"
    assert annotations[1][0]["bbox"] == [1, 2, 3, 4]


def test_tile_regions_have_non_overlapping_ownership_cores():
    regions = list(tile_regions(1280, 1024, 640, 0.2))
    assert len(regions) == 6
    assert regions[0] == (0, 0, 0.0, 0.0, 576.0, 512.0)
    assert regions[-1] == (640, 384, 896.0, 512.0, 1280.0, 1024.0)


def test_tile_labels_assign_each_center_once():
    annotations = [
        {"bbox": [560, 500, 20, 20]},
        {"bbox": [700, 700, 30, 20]},
    ]
    assigned = []
    for x0, y0, left, top, right, bottom in tile_regions(1280, 1024, 640, 0.2):
        labels = tile_yolo_labels(
            annotations,
            x0=x0,
            y0=y0,
            crop_width=640,
            crop_height=640,
            ownership=(left, top, right, bottom),
        )
        assigned.extend(labels)
    assert len(assigned) == len(annotations)
