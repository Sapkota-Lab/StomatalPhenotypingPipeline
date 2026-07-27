#!/usr/bin/env python3
"""Build leave-one-domain-out YOLO datasets from the expert keypoint benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("tile_overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_regions(width: int, height: int, tile_size: int, overlap: float):
    """Yield crop bounds and a non-overlapping center-ownership core."""
    xs = tile_starts(width, tile_size, overlap)
    ys = tile_starts(height, tile_size, overlap)
    for yi, y in enumerate(ys):
        top = 0.0 if yi == 0 else (ys[yi - 1] + tile_size + y) / 2
        bottom = float(height) if yi == len(ys) - 1 else (
            y + tile_size + ys[yi + 1]
        ) / 2
        for xi, x in enumerate(xs):
            left = 0.0 if xi == 0 else (xs[xi - 1] + tile_size + x) / 2
            right = float(width) if xi == len(xs) - 1 else (
                x + tile_size + xs[xi + 1]
            ) / 2
            yield x, y, left, top, right, bottom


def tile_yolo_labels(
    annotations: list[dict],
    *,
    x0: int,
    y0: int,
    crop_width: int,
    crop_height: int,
    ownership: tuple[float, float, float, float],
) -> list[str]:
    """Return labels whose original centers belong to this tile's core.

    Overlapping crops make every small benchmark box fully visible in its
    owner tile. Boxes at an outer image edge are clipped to valid crop bounds.
    """
    left, top, right, bottom = ownership
    labels: list[str] = []
    for annotation in annotations:
        x, y, box_width, box_height = (
            float(value) for value in annotation["bbox"]
        )
        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        if not (left <= center_x < right and top <= center_y < bottom):
            continue

        clipped_x1 = max(float(x0), x)
        clipped_y1 = max(float(y0), y)
        clipped_x2 = min(float(x0 + crop_width), x + box_width)
        clipped_y2 = min(float(y0 + crop_height), y + box_height)
        clipped_width = clipped_x2 - clipped_x1
        clipped_height = clipped_y2 - clipped_y1
        if clipped_width <= 0 or clipped_height <= 0:
            continue
        local_center_x = (
            (clipped_x1 + clipped_x2) / 2.0 - x0
        ) / crop_width
        local_center_y = (
            (clipped_y1 + clipped_y2) / 2.0 - y0
        ) / crop_height
        labels.append(
            "0 "
            f"{local_center_x:.8f} {local_center_y:.8f} "
            f"{clipped_width / crop_width:.8f} "
            f"{clipped_height / crop_height:.8f}"
        )
    return labels


def load_coco(split_dir: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    document = json.loads((split_dir / "coco.json").read_text())
    images = {int(image["id"]): image for image in document["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    return images, dict(annotations)


def yolo_labels(image: dict, annotations: list[dict]) -> list[str]:
    width = float(image["width"])
    height = float(image["height"])
    labels = []
    for annotation in annotations:
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        center_x = (x + box_width / 2.0) / width
        center_y = (y + box_height / 2.0) / height
        labels.append(
            "0 "
            f"{center_x:.8f} {center_y:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
    return labels


def reset_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output / "full_images" / split).mkdir(parents=True, exist_ok=True)


def write_filtered_coco(
    source_dir: Path,
    image_ids: list[int],
    destination: Path,
) -> None:
    document = json.loads((source_dir / "coco.json").read_text())
    selected_ids = set(image_ids)
    images = []
    for image in document["images"]:
        if int(image["id"]) not in selected_ids:
            continue
        normalized = dict(image)
        normalized["file_name"] = Path(image["file_name"]).name
        images.append(normalized)
    filtered = {
        "images": images,
        "annotations": [
            annotation
            for annotation in document["annotations"]
            if int(annotation["image_id"]) in selected_ids
        ],
        "categories": document["categories"],
    }
    destination.write_text(json.dumps(filtered, indent=2) + "\n")


def materialize_image(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve())


def add_split(
    *,
    source_dir: Path,
    image_ids: list[int],
    destination: Path,
    destination_split: str,
    source_domain: str,
    tile_size: int | None = None,
    tile_overlap: float = 0.2,
    jpeg_quality: int = 95,
) -> list[dict[str, object]]:
    images, annotations = load_coco(source_dir)
    rows: list[dict[str, object]] = []
    for image_id in image_ids:
        image = images[image_id]
        filename = Path(image["file_name"]).name
        source_image = source_dir / "images" / filename
        if not source_image.exists():
            raise FileNotFoundError(source_image)
        materialize_image(
            source_image,
            destination / "full_images" / destination_split / filename,
        )

        if tile_size is None or destination_split == "test":
            target_image = destination / "images" / destination_split / filename
            materialize_image(source_image, target_image)
            target_label = (
                destination
                / "labels"
                / destination_split
                / f"{Path(filename).stem}.txt"
            )
            labels = yolo_labels(image, annotations.get(image_id, []))
            target_label.write_text("\n".join(labels) + ("\n" if labels else ""))
            rows.append(
                {
                    "filename": filename,
                    "source_filename": filename,
                    "source_domain": source_domain,
                    "dataset_split": destination_split,
                    "tile_x": "",
                    "tile_y": "",
                    "objects": len(labels),
                }
            )
            continue

        pixels = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f"Could not read {source_image}")
        height, width = pixels.shape[:2]
        expected_width = int(image["width"])
        expected_height = int(image["height"])
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"COCO dimensions for {filename} are "
                f"{expected_width}x{expected_height}, image is {width}x{height}"
            )
        source_annotations = annotations.get(image_id, [])
        assigned_objects = 0
        for x0, y0, left, top, right, bottom in tile_regions(
            width,
            height,
            tile_size,
            tile_overlap,
        ):
            crop = pixels[y0:min(y0 + tile_size, height), x0:min(x0 + tile_size, width)]
            crop_height, crop_width = crop.shape[:2]
            tile_name = (
                f"{Path(filename).stem}__x{x0:04d}_y{y0:04d}"
                f"{Path(filename).suffix.lower()}"
            )
            labels = tile_yolo_labels(
                source_annotations,
                x0=x0,
                y0=y0,
                crop_width=crop_width,
                crop_height=crop_height,
                ownership=(left, top, right, bottom),
            )
            target_image = destination / "images" / destination_split / tile_name
            if not cv2.imwrite(
                str(target_image),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise OSError(f"Could not write {target_image}")
            target_label = (
                destination
                / "labels"
                / destination_split
                / f"{Path(tile_name).stem}.txt"
            )
            target_label.write_text("\n".join(labels) + ("\n" if labels else ""))
            assigned_objects += len(labels)
            rows.append(
                {
                    "filename": tile_name,
                    "source_filename": filename,
                    "source_domain": source_domain,
                    "dataset_split": destination_split,
                    "tile_x": x0,
                    "tile_y": y0,
                    "objects": len(labels),
                }
            )
        if assigned_objects != len(source_annotations):
            raise ValueError(
                f"Assigned {assigned_objects} of {len(source_annotations)} "
                f"annotations for {filename}"
            )
    return rows


def build_dataset(
    benchmark: Path,
    output: Path,
    train_domain: str,
    test_domain: str,
    validation_images: int,
    seed: int,
    tile_size: int | None = None,
    tile_overlap: float = 0.2,
    jpeg_quality: int = 95,
) -> list[dict[str, object]]:
    train_dir = benchmark / "datasets" / train_domain
    test_dir = benchmark / "datasets" / test_domain
    train_images, _ = load_coco(train_dir)
    test_images, _ = load_coco(test_dir)
    if validation_images <= 0 or validation_images >= len(train_images):
        raise ValueError("validation_images must leave at least one training image")

    train_ids = sorted(train_images)
    random.Random(seed).shuffle(train_ids)
    validation_ids = sorted(train_ids[:validation_images])
    fitting_ids = sorted(train_ids[validation_images:])
    test_ids = sorted(test_images)

    reset_output(output)
    rows = []
    rows.extend(
        add_split(
            source_dir=train_dir,
            image_ids=fitting_ids,
            destination=output,
            destination_split="train",
            source_domain=train_domain,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            jpeg_quality=jpeg_quality,
        )
    )
    rows.extend(
        add_split(
            source_dir=train_dir,
            image_ids=validation_ids,
            destination=output,
            destination_split="val",
            source_domain=train_domain,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            jpeg_quality=jpeg_quality,
        )
    )
    rows.extend(
        add_split(
            source_dir=test_dir,
            image_ids=test_ids,
            destination=output,
            destination_split="test",
            source_domain=test_domain,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            jpeg_quality=jpeg_quality,
        )
    )
    write_filtered_coco(train_dir, validation_ids, output / "val_coco.json")
    write_filtered_coco(test_dir, test_ids, output / "test_coco.json")

    (output / "dataset.yaml").write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: stomata\n"
    )
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("external_validation/stomata_keypoint_benchmark"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-domain", choices=("T-SBGH", "T-MSAA"), required=True)
    parser.add_argument("--test-domain", choices=("T-SBGH", "T-MSAA"), required=True)
    parser.add_argument("--validation-images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--tile-size",
        type=int,
        help="Crop train/validation images into ownership-core tiles.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_domain == args.test_domain:
        raise SystemExit("train-domain and test-domain must be different")
    rows = build_dataset(
        args.benchmark,
        args.output,
        args.train_domain,
        args.test_domain,
        args.validation_images,
        args.seed,
        args.tile_size,
        args.tile_overlap,
        args.jpeg_quality,
    )
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["dataset_split"] == split]
        print(
            f"{split}: {len(subset)} images, "
            f"{sum(int(row['objects']) for row in subset)} expert boxes"
        )


if __name__ == "__main__":
    main()
