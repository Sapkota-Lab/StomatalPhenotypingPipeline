#!/usr/bin/env python3
"""Morphology-aware stomata detection with four-keypoint pose predictions.

The model predicts:
  p0, p1: polar tips along the stomatal length axis
  p2, p3: lateral endpoints along the stomatal width axis

The derived measurements are intended for validation and review.  A prediction
is not considered biologically correct merely because its keypoints form a
plausible cross; the operating thresholds must pass held-out object-level
evaluation before this engine can be used for unattended counting.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class PoseDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    keypoints: np.ndarray
    keypoint_min_conf: float
    keypoint_median_conf: float
    length_px: float
    width_px: float
    length_width_ratio: float
    perpendicular_deviation_deg: float
    center_offset_norm: float
    keypoints_inside_fraction: float
    morphology_plausible: bool = False


@dataclass(frozen=True)
class MorphologyThresholds:
    min_keypoint_conf: float = 0.50
    min_length_width_ratio: float = 0.90
    max_length_width_ratio: float = 2.10
    max_perpendicular_deviation_deg: float = 20.0
    max_center_offset_norm: float = 0.16
    min_keypoints_inside_fraction: float = 0.75


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_exif_normalized(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        return cv2.cvtColor(
            cv2.merge((enhanced, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
    if mode == "gray-clahe":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def morphology_measurements(
    keypoints: np.ndarray,
    box: Sequence[float],
) -> dict[str, float]:
    points = np.asarray(keypoints, dtype=float)
    if points.shape != (4, 3):
        raise ValueError(f"Expected keypoints with shape (4, 3), got {points.shape}")

    length_axis = points[1, :2] - points[0, :2]
    width_axis = points[3, :2] - points[2, :2]
    length = float(np.linalg.norm(length_axis))
    width = float(np.linalg.norm(width_axis))

    if length <= 1e-9 or width <= 1e-9:
        ratio = math.inf
        deviation = 90.0
        center_offset = math.inf
    else:
        ratio = length / width
        absolute_cosine = float(
            np.clip(
                abs(np.dot(length_axis, width_axis) / (length * width)),
                0.0,
                1.0,
            )
        )
        acute_angle = math.degrees(math.acos(absolute_cosine))
        deviation = 90.0 - acute_angle
        length_midpoint = (points[0, :2] + points[1, :2]) / 2.0
        width_midpoint = (points[2, :2] + points[3, :2]) / 2.0
        center_offset = float(
            np.linalg.norm(length_midpoint - width_midpoint) / max(length, width)
        )

    x1, y1, x2, y2 = (float(value) for value in box)
    inside = (
        (points[:, 0] >= x1)
        & (points[:, 0] <= x2)
        & (points[:, 1] >= y1)
        & (points[:, 1] <= y2)
    )
    confidences = points[:, 2]
    return {
        "keypoint_min_conf": float(np.min(confidences)),
        "keypoint_median_conf": float(np.median(confidences)),
        "length_px": length,
        "width_px": width,
        "length_width_ratio": ratio,
        "perpendicular_deviation_deg": deviation,
        "center_offset_norm": center_offset,
        "keypoints_inside_fraction": float(np.mean(inside)),
    }


def is_morphology_plausible(
    measurements: dict[str, float],
    thresholds: MorphologyThresholds,
) -> bool:
    return bool(
        measurements["keypoint_min_conf"] >= thresholds.min_keypoint_conf
        and thresholds.min_length_width_ratio
        <= measurements["length_width_ratio"]
        <= thresholds.max_length_width_ratio
        and measurements["perpendicular_deviation_deg"]
        <= thresholds.max_perpendicular_deviation_deg
        and measurements["center_offset_norm"] <= thresholds.max_center_offset_norm
        and measurements["keypoints_inside_fraction"]
        >= thresholds.min_keypoints_inside_fraction
    )


def make_detection(
    box: Sequence[float],
    confidence: float,
    keypoints: np.ndarray,
    thresholds: MorphologyThresholds,
) -> PoseDetection:
    measurements = morphology_measurements(keypoints, box)
    return PoseDetection(
        x1=float(box[0]),
        y1=float(box[1]),
        x2=float(box[2]),
        y2=float(box[3]),
        confidence=float(confidence),
        keypoints=np.asarray(keypoints, dtype=float),
        morphology_plausible=is_morphology_plausible(measurements, thresholds),
        **measurements,
    )


def box_iou(a: PoseDetection, b: PoseDetection) -> float:
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections: Iterable[PoseDetection], threshold: float) -> list[PoseDetection]:
    pending = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
    kept: list[PoseDetection] = []
    while pending:
        chosen = pending.pop(0)
        kept.append(chosen)
        pending = [candidate for candidate in pending if box_iou(chosen, candidate) < threshold]
    return kept


def infer_array(
    model: YOLO,
    image: np.ndarray,
    *,
    confidence: float,
    iou: float,
    image_size: int,
    device: str,
    thresholds: MorphologyThresholds,
) -> list[PoseDetection]:
    result = model.predict(
        source=image,
        imgsz=image_size,
        conf=confidence,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    if result.boxes is None or result.keypoints is None:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    keypoints = result.keypoints.data.detach().cpu().numpy()
    return [
        make_detection(box, score, points, thresholds)
        for box, score, points in zip(boxes, scores, keypoints)
    ]


def infer_tiled(
    model: YOLO,
    image: np.ndarray,
    *,
    tile_size: int,
    overlap: float,
    confidence: float,
    model_iou: float,
    global_iou: float,
    image_size: int,
    device: str,
    thresholds: MorphologyThresholds,
) -> list[PoseDetection]:
    height, width = image.shape[:2]
    detections: list[PoseDetection] = []
    for y0 in tile_starts(height, tile_size, overlap):
        for x0 in tile_starts(width, tile_size, overlap):
            crop = image[y0 : min(y0 + tile_size, height), x0 : min(x0 + tile_size, width)]
            for detection in infer_array(
                model,
                crop,
                confidence=confidence,
                iou=model_iou,
                image_size=image_size,
                device=device,
                thresholds=thresholds,
            ):
                detection.x1 += x0
                detection.x2 += x0
                detection.y1 += y0
                detection.y2 += y0
                detection.keypoints[:, 0] += x0
                detection.keypoints[:, 1] += y0
                detections.append(detection)
    return nms(detections, global_iou)


def draw_overlay(image: np.ndarray, detections: Sequence[PoseDetection]) -> np.ndarray:
    canvas = image.copy()
    for detection in detections:
        color = (60, 190, 70) if detection.morphology_plausible else (0, 170, 255)
        p = detection.keypoints[:, :2].round().astype(int)
        cv2.rectangle(
            canvas,
            (round(detection.x1), round(detection.y1)),
            (round(detection.x2), round(detection.y2)),
            color,
            2,
        )
        cv2.line(canvas, tuple(p[0]), tuple(p[1]), (255, 220, 0), 2)
        cv2.line(canvas, tuple(p[2]), tuple(p[3]), (255, 0, 220), 2)
        for index, point in enumerate(p):
            cv2.circle(canvas, tuple(point), 3, color, -1)
            cv2.putText(
                canvas,
                str(index),
                (int(point[0]) + 3, int(point[1]) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def detection_row(image_name: str, detection: PoseDetection) -> dict[str, object]:
    row = asdict(detection)
    keypoints = row.pop("keypoints")
    for index, point in enumerate(np.asarray(keypoints)):
        row[f"p{index}_x"] = float(point[0])
        row[f"p{index}_y"] = float(point[1])
        row[f"p{index}_confidence"] = float(point[2])
    row["image"] = image_name
    row["status"] = "plausible" if detection.morphology_plausible else "review"
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/stomata_yolo26x_pose.pt"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--model-iou", type=float, default=0.70)
    parser.add_argument("--global-iou", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Use 0 for full-image inference; use e.g. 1280 for large target images.",
    )
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.50)
    parser.add_argument("--min-length-width-ratio", type=float, default=0.90)
    parser.add_argument("--max-length-width-ratio", type=float, default=2.10)
    parser.add_argument("--max-perpendicular-deviation", type=float, default=20.0)
    parser.add_argument("--max-center-offset", type=float, default=0.16)
    parser.add_argument("--min-keypoints-inside", type=float, default=0.75)
    parser.add_argument(
        "--preprocess",
        choices=("none", "clahe", "gray-clahe"),
        default="none",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip JPEG overlays when only quantitative evaluation is needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = MorphologyThresholds(
        min_keypoint_conf=args.min_keypoint_confidence,
        min_length_width_ratio=args.min_length_width_ratio,
        max_length_width_ratio=args.max_length_width_ratio,
        max_perpendicular_deviation_deg=args.max_perpendicular_deviation,
        max_center_offset_norm=args.max_center_offset,
        min_keypoints_inside_fraction=args.min_keypoints_inside,
    )
    paths = image_paths(args.input)
    if not paths:
        raise SystemExit(f"No images found under {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    model = YOLO(str(args.model))

    detection_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        image = read_exif_normalized(path)
        inference_image = preprocess_image(image, args.preprocess)
        if args.tile_size > 0:
            detections = infer_tiled(
                model,
                inference_image,
                tile_size=args.tile_size,
                overlap=args.overlap,
                confidence=args.confidence,
                model_iou=args.model_iou,
                global_iou=args.global_iou,
                image_size=args.image_size,
                device=args.device,
                thresholds=thresholds,
            )
        else:
            detections = infer_array(
                model,
                inference_image,
                confidence=args.confidence,
                iou=args.model_iou,
                image_size=args.image_size,
                device=args.device,
                thresholds=thresholds,
            )

        plausible = sum(detection.morphology_plausible for detection in detections)
        count_rows.append(
            {
                "image": path.name,
                "plausible_count": plausible,
                "review_count": len(detections) - plausible,
                "raw_count": len(detections),
            }
        )
        detection_rows.extend(detection_row(path.name, detection) for detection in detections)
        if not args.no_overlays:
            cv2.imwrite(
                str(overlay_dir / f"{path.stem}_pose.jpg"),
                draw_overlay(image, detections),
            )
        print(
            f"[{index}/{len(paths)}] {path.name}: "
            f"{plausible} plausible, {len(detections) - plausible} review"
        )

    count_fields = ["image", "plausible_count", "review_count", "raw_count"]
    with (args.output / "counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=count_fields)
        writer.writeheader()
        writer.writerows(count_rows)

    if detection_rows:
        with (args.output / "detections.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detection_rows[0]))
            writer.writeheader()
            writer.writerows(detection_rows)


if __name__ == "__main__":
    main()
