#!/usr/bin/env python3
"""Export low-threshold Ultralytics detections to the common CSV schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from stomata_engine import Detection, global_nms, tile_regions
from stomata_pose_engine import image_paths


FIELDS = ["filename", "detection_id", "confidence", "x1", "y1", "x2", "y2"]


def result_rows(filename: str, result) -> list[dict[str, object]]:
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    rows = []
    for box, score, class_id in zip(boxes, scores, classes):
        if int(class_id) != 0:
            continue
        rows.append(
            {
                "filename": filename,
                "detection_id": len(rows) + 1,
                "confidence": float(score),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            }
        )
    return rows


def owned_by_core(
    box: tuple[float, float, float, float] | list[float],
    *,
    origin_x: int,
    origin_y: int,
    core: tuple[float, float, float, float],
) -> bool:
    x1, y1, x2, y2 = (float(value) for value in box)
    center_x = (x1 + x2) / 2.0 + origin_x
    center_y = (y1 + y2) / 2.0 + origin_y
    left, top, right, bottom = core
    return left <= center_x < right and top <= center_y < bottom


def tiled_rows(
    filename: str,
    model: YOLO,
    image,
    *,
    tile_size: int,
    overlap: float,
    image_size: int,
    confidence: float,
    iou: float,
    global_iou: float,
    max_detections: int,
    device: str,
    tile_batch: int,
) -> list[dict[str, object]]:
    height, width = image.shape[:2]
    tiles = []
    tile_metadata = []
    for x0, y0, left, top, right, bottom in tile_regions(
        width,
        height,
        tile_size,
        overlap,
    ):
        tiles.append(image[y0:min(y0 + tile_size, height), x0:min(x0 + tile_size, width)])
        tile_metadata.append((x0, y0, (left, top, right, bottom)))

    detections: list[Detection] = []
    for start in range(0, len(tiles), tile_batch):
        batch = tiles[start:start + tile_batch]
        results = model.predict(
            source=batch,
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            max_det=max_detections,
            device=device,
            verbose=False,
        )
        for result, (x0, y0, core) in zip(
            results,
            tile_metadata[start:start + tile_batch],
        ):
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            for box, score, class_id in zip(boxes, scores, classes):
                if int(class_id) != 0:
                    continue
                if not owned_by_core(
                    box,
                    origin_x=x0,
                    origin_y=y0,
                    core=core,
                ):
                    continue
                detections.append(
                    Detection(
                        x1=max(0.0, float(box[0]) + x0),
                        y1=max(0.0, float(box[1]) + y0),
                        x2=min(float(width), float(box[2]) + x0),
                        y2=min(float(height), float(box[3]) + y0),
                        confidence=float(score),
                    )
                )

    merged = global_nms(detections, global_iou)
    return [
        {
            "filename": filename,
            "detection_id": index,
            "confidence": detection.confidence,
            "x1": detection.x1,
            "y1": detection.y1,
            "x2": detection.x2,
            "y2": detection.y2,
        }
        for index, detection in enumerate(merged, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-detections", type=int, default=1000)
    parser.add_argument(
        "--tile-size",
        type=int,
        help="Run overlapping tiled inference instead of resizing the full image.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--global-iou", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = image_paths(args.input)
    if not paths:
        raise SystemExit(f"No images found under {args.input}")
    model = YOLO(str(args.model))
    rows = []
    if args.tile_size:
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read {path}")
            image_rows = tiled_rows(
                path.name,
                model,
                image,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                image_size=args.image_size,
                confidence=args.confidence,
                iou=args.iou,
                global_iou=args.global_iou,
                max_detections=args.max_detections,
                device=args.device,
                tile_batch=args.tile_batch,
            )
            rows.extend(image_rows)
            print(f"{path.name}: {len(image_rows)} proposals")
    else:
        results = model.predict(
            source=[str(path) for path in paths],
            imgsz=args.image_size,
            conf=args.confidence,
            iou=args.iou,
            max_det=args.max_detections,
            device=args.device,
            verbose=False,
            stream=True,
        )
        for path, result in zip(paths, results):
            image_rows = result_rows(path.name, result)
            rows.extend(image_rows)
            print(f"{path.name}: {len(image_rows)} proposals")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
