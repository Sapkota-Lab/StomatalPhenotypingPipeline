#!/usr/bin/env python3
"""Build a source-image-split YOLO dataset from consensus detections.

This is a bootstrap dataset, not biological ground truth. It is intended for
domain-adaptation experiments whose predictions must still be evaluated on
independently annotated held-out images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2


def stable_split(filename: str, validation_fraction: float, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{filename}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if value < validation_fraction else "train"


def centered_window(
    center_x: float,
    center_y: float,
    crop_size: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    width = min(crop_size, image_width)
    height = min(crop_size, image_height)
    left = round(center_x - width / 2)
    top = round(center_y - height / 2)
    left = min(max(left, 0), image_width - width)
    top = min(max(top, 0), image_height - height)
    return left, top, left + width, top + height


def clip_box(
    box: tuple[float, float, float, float],
    window: tuple[int, int, int, int],
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    left, top, right, bottom = window
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    if not (left <= center_x < right and top <= center_y < bottom):
        return None
    return (
        max(x1, left) - left,
        max(y1, top) - top,
        min(x2, right) - left,
        min(y2, bottom) - top,
    )


def yolo_line(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return (
        f"0 {center_x:.8f} {center_y:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="stomata-v1")
    args = parser.parse_args()
    if args.crop_size <= 0:
        parser.error("--crop-size must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    with args.detections.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "accepted"
        ]
    by_filename: dict[str, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )
    for row in rows:
        by_filename[row["filename"]].append(tuple(
            float(row[name]) for name in ("x1", "y1", "x2", "y2")
        ))

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_images: dict[str, set[str]] = defaultdict(set)
    crop_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)
    for filename, boxes in sorted(by_filename.items()):
        image = cv2.imread(str(args.images / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {args.images / filename}")
        split = stable_split(filename, args.validation_fraction, args.seed)
        split_images[split].add(filename)
        for index, anchor in enumerate(boxes, 1):
            center_x = (anchor[0] + anchor[2]) / 2
            center_y = (anchor[1] + anchor[3]) / 2
            window = centered_window(
                center_x,
                center_y,
                args.crop_size,
                image.shape[1],
                image.shape[0],
            )
            left, top, right, bottom = window
            crop = image[top:bottom, left:right]
            crop_name = f"{Path(filename).stem}_{index:04d}"
            image_path = args.output / "images" / split / f"{crop_name}.jpg"
            label_path = args.output / "labels" / split / f"{crop_name}.txt"
            if not cv2.imwrite(str(image_path), crop):
                raise OSError(f"Could not write {image_path}")
            labels = [
                yolo_line(clipped, crop.shape[1], crop.shape[0])
                for box in boxes
                if (clipped := clip_box(box, window)) is not None
            ]
            label_path.write_text("\n".join(labels) + "\n")
            crop_counts[split] += 1
            label_counts[split] += len(labels)

    dataset_yaml = (
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: stoma\n"
    )
    (args.output / "dataset.yaml").write_text(dataset_yaml)
    summary = {
        "warning": (
            "Bootstrap pseudo-labels from model consensus; not biological "
            "ground truth."
        ),
        "crop_size": args.crop_size,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "source_images": {
            split: sorted(names) for split, names in split_images.items()
        },
        "crop_counts": dict(crop_counts),
        "label_counts": dict(label_counts),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
