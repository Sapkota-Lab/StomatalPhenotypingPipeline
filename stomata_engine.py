#!/usr/bin/env python3
"""Tiled stomata detection and counting for high-resolution microscope images.

The engine runs a stomata-specific YOLO model on overlapping tiles, merges
duplicate detections across tile boundaries, writes per-image counts and full
detection records, and produces review overlays.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from ultralytics import YOLO
from ultralytics.nn.modules.block import AAttn


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def patch_yolov12_attention() -> None:
    """Support both qkv layouts used by published YOLOv12 checkpoints.

    Some YOLOv12 checkpoints contain separate ``qk`` and ``v`` layers while
    the pinned source commit expects a combined ``qkv`` layer. The model
    authors use the same compatibility strategy in their public demo.
    """
    def compatible_forward(self, x):
        batch, channels, height, width = x.shape
        tokens = height * width
        if hasattr(self, "qkv"):
            qkv = self.qkv(x)
        else:
            qkv = torch.cat((self.qk(x), self.v(x)), dim=1)
        qkv = qkv.flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(batch * self.area, tokens // self.area, channels * 3)
            batch, tokens, _ = qkv.shape
        q, k, value = qkv.view(batch, tokens, self.num_heads, self.head_dim * 3).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=3
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        value_heads = value.permute(0, 2, 1, 3)
        attended = torch_functional.scaled_dot_product_attention(q, k, value_heads)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, tokens, channels)
        value = value.reshape(batch, tokens, channels)
        if self.area > 1:
            attended = attended.reshape(batch // self.area, tokens * self.area, channels)
            value = value.reshape(batch // self.area, tokens * self.area, channels)
            batch, tokens, _ = attended.shape
        attended = attended.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        value = value.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return self.proj(attended + self.pe(value))

    AAttn.forward = compatible_forward


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_regions(width: int, height: int, tile_size: int, overlap: float):
    """Yield tiles and non-overlapping ownership cores.

    A prediction is retained only when its center lies in the tile's core.
    Core boundaries bisect overlap with neighboring tiles, preventing partial
    detections at internal tile edges from entering the final count.
    """
    xs = tile_starts(width, tile_size, overlap)
    ys = tile_starts(height, tile_size, overlap)
    for yi, y in enumerate(ys):
        top = 0.0 if yi == 0 else (ys[yi - 1] + tile_size + y) / 2
        bottom = float(height) if yi == len(ys) - 1 else (y + tile_size + ys[yi + 1]) / 2
        for xi, x in enumerate(xs):
            left = 0.0 if xi == 0 else (xs[xi - 1] + tile_size + x) / 2
            right = float(width) if xi == len(xs) - 1 else (x + tile_size + xs[xi + 1]) / 2
            yield x, y, left, top, right, bottom


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def global_nms(
    detections: list[Detection],
    threshold: float,
    center_distance_threshold: float | None = None,
) -> list[Detection]:
    """Suppress duplicate detections from overlapping inference tiles.

    IoU alone misses duplicate boxes when two tiles predict substantially
    different box sizes for the same object. ``center_distance_threshold``
    optionally suppresses boxes whose centers are nearly coincident relative
    to the larger dimension of the higher-confidence box.
    """
    if not detections:
        return []
    boxes = np.asarray([d.box for d in detections], dtype=np.float32)
    scores = np.asarray([d.confidence for d in detections], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        duplicate = iou(boxes[current], boxes[remaining]) > threshold
        if center_distance_threshold is not None:
            current_center = np.asarray([
                (boxes[current, 0] + boxes[current, 2]) / 2,
                (boxes[current, 1] + boxes[current, 3]) / 2,
            ])
            remaining_centers = np.column_stack((
                (boxes[remaining, 0] + boxes[remaining, 2]) / 2,
                (boxes[remaining, 1] + boxes[remaining, 3]) / 2,
            ))
            current_scale = max(
                boxes[current, 2] - boxes[current, 0],
                boxes[current, 3] - boxes[current, 1],
                1e-9,
            )
            remaining_scales = np.maximum(
                boxes[remaining, 2] - boxes[remaining, 0],
                boxes[remaining, 3] - boxes[remaining, 1],
            )
            center_distances = np.linalg.norm(
                remaining_centers - current_center,
                axis=1,
            ) / np.maximum(current_scale, remaining_scales)
            duplicate |= center_distances <= center_distance_threshold
        order = remaining[~duplicate]
    return [detections[index] for index in keep]


def infer_image(
    model: YOLO,
    image: np.ndarray,
    tile_size: int,
    overlap: float,
    confidence: float,
    nms_iou: float,
    device: str | None,
) -> list[Detection]:
    height, width = image.shape[:2]
    tiles: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for x, y, _left, _top, _right, _bottom in tile_regions(width, height, tile_size, overlap):
        tiles.append(image[y:min(y + tile_size, height), x:min(x + tile_size, width)])
        origins.append((x, y))

    detections: list[Detection] = []
    # Keep each prediction call small. YOLOv12's built-in NMS has a per-call
    # time budget and may silently stop early on a large list of dense tiles.
    chunk_size = 4
    for start in range(0, len(tiles), chunk_size):
        tile_chunk = tiles[start:start + chunk_size]
        origin_chunk = origins[start:start + chunk_size]
        predictions = model.predict(
            source=tile_chunk,
            imgsz=tile_size,
            conf=confidence,
            iou=nms_iou,
            device=device,
            batch=len(tile_chunk),
            verbose=False,
        )
        for result, (origin_x, origin_y), tile in zip(predictions, origin_chunk, tile_chunk):
            if result.boxes is None:
                continue
            for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().numpy(),
                                             result.boxes.conf.cpu().numpy(),
                                             result.boxes.cls.cpu().numpy()):
                if int(class_id) != 0:
                    continue
                x1, y1, x2, y2 = xyxy
                tile_height, tile_width = tile.shape[:2]
                edge_tolerance = 3
                clipped_at_internal_edge = (
                    (origin_x > 0 and x1 <= edge_tolerance)
                    or (origin_y > 0 and y1 <= edge_tolerance)
                    or (origin_x + tile_width < width and x2 >= tile_width - edge_tolerance)
                    or (origin_y + tile_height < height and y2 >= tile_height - edge_tolerance)
                )
                if clipped_at_internal_edge:
                    continue
                global_x1, global_y1 = float(x1 + origin_x), float(y1 + origin_y)
                global_x2, global_y2 = float(x2 + origin_x), float(y2 + origin_y)
                detections.append(Detection(
                    global_x1, global_y1, global_x2, global_y2, float(score)
                ))
    return global_nms(detections, nms_iou)


def image_quality(image: np.ndarray) -> dict[str, float | str]:
    """Return inexpensive acquisition-quality diagnostics.

    Thresholds are empirical for the current Canon microscope batch and are
    reported for review; they do not alter detections.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 1500 / width)
    view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    clipped = float(np.mean((gray < 5) | (gray > 250)))
    if focus < 150 or clipped > 0.05:
        flag = "low"
    elif focus < 250 or clipped > 0.03:
        flag = "medium"
    else:
        flag = "high"
    return {
        "image_quality": flag,
        "focus_score": round(focus, 2),
        "contrast_score": round(contrast, 2),
        "clipped_fraction": round(clipped, 6),
    }


def draw_overlay(
    image: np.ndarray,
    detections: list[Detection],
    output: Path,
    review_detections: list[Detection] | None = None,
    max_width: int = 2400,
) -> None:
    scale = min(1.0, max_width / image.shape[1])
    view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    thickness = max(1, round(3 * scale))
    for index, detection in enumerate(review_detections or [], start=1):
        x1, y1, x2, y2 = (round(value * scale) for value in detection.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 180, 255), thickness)
        cv2.putText(view, f"R{index}:{detection.confidence:.2f}", (x1, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 220), 1, cv2.LINE_AA)
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (round(value * scale) for value in detection.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), thickness)
        cv2.putText(view, f"A{index}:{detection.confidence:.2f}", (x1, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    review_count = len(review_detections or [])
    cv2.putText(view, f"accepted: {len(detections)}  review: {review_count}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 40, 40), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), view):
        raise OSError(f"Could not write overlay {output}")


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--model", type=Path, default=Path("models/stomaquant_yolov12.pt"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="Confidence required for the conservative accepted count")
    parser.add_argument("--review-confidence", type=float, default=0.05,
                        help="Lower threshold for candidates shown in amber for review")
    parser.add_argument("--tile-size", type=int, default=1216)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--device", default=None, help="Examples: cpu, mps, 0")
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    if not 0 <= args.review_confidence <= args.confidence <= 1:
        parser.error("Require 0 <= review-confidence <= confidence <= 1")
    paths = image_paths(args.input)
    if not paths:
        parser.error(f"No supported images found in {args.input}")

    patch_yolov12_attention()
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    count_rows = []
    detection_rows = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        candidates = infer_image(model, image, args.tile_size, args.overlap,
                                 args.review_confidence, args.nms_iou, args.device)
        accepted = [item for item in candidates if item.confidence >= args.confidence]
        review = [item for item in candidates if item.confidence < args.confidence]
        quality = image_quality(image)
        uncertainty = len(review) / max(1, len(candidates))
        count_rows.append({
            "filename": path.name,
            "accepted_count": len(accepted),
            "review_count": len(review),
            "candidate_count": len(candidates),
            "uncertainty_fraction": round(uncertainty, 4),
            **quality,
        })
        for index, detection in enumerate(candidates, start=1):
            detection_rows.append({
                "filename": path.name, "detection_id": index,
                "status": "accepted" if detection.confidence >= args.confidence else "review",
                "confidence": round(detection.confidence, 6),
                "x1": round(detection.x1, 2), "y1": round(detection.y1, 2),
                "x2": round(detection.x2, 2), "y2": round(detection.y2, 2),
            })
        draw_overlay(image, accepted, args.output / "overlays" / f"{path.stem}_overlay.jpg", review)
        print(f"{path.name}: accepted={len(accepted)} review={len(review)} quality={quality['image_quality']}")

    with (args.output / "counts.csv").open("w", newline="") as handle:
        count_fields = [
            "filename", "accepted_count", "review_count", "candidate_count",
            "uncertainty_fraction", "image_quality", "focus_score",
            "contrast_score", "clipped_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=count_fields)
        writer.writeheader()
        writer.writerows(count_rows)
    with (args.output / "detections.csv").open("w", newline="") as handle:
        fields = ["filename", "detection_id", "status", "confidence", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detection_rows)


if __name__ == "__main__":
    main()
