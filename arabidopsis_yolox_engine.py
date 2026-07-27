#!/usr/bin/env python3
"""Run the published Arabidopsis bright-field YOLOX stomata detector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from stomata_engine import Detection, draw_overlay, image_paths


DEFAULT_MODEL = Path("models/external/arabidopsis_yolox_s1920.onnx")
INPUT_SIZE = (1920, 1920)
CLASS_NAMES = ("open", "closed")


def preprocess(
    image: np.ndarray,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> tuple[np.ndarray, float]:
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    ratio = min(
        input_size[0] / image.shape[0],
        input_size[1] / image.shape[1],
    )
    resized = cv2.resize(
        image,
        (
            int(image.shape[1] * ratio),
            int(image.shape[0] * ratio),
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    padded[:resized.shape[0], :resized.shape[1]] = resized
    tensor = np.ascontiguousarray(
        padded.transpose(2, 0, 1),
        dtype=np.float32,
    )
    return tensor, ratio


def decode_yolox(
    output: np.ndarray,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    strides = (8, 16, 32)
    grids = []
    expanded_strides = []
    for stride in strides:
        height = input_size[0] // stride
        width = input_size[1] // stride
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((x_grid, y_grid), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(
            np.full((*grid.shape[:2], 1), stride)
        )
    grid = np.concatenate(grids, axis=1)
    stride_values = np.concatenate(expanded_strides, axis=1)
    decoded = output.copy()
    decoded[..., :2] = (decoded[..., :2] + grid) * stride_values
    decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * stride_values
    return decoded


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        intersection_width = np.maximum(
            0,
            np.minimum(x2[index], x2[rest])
            - np.maximum(x1[index], x1[rest]) + 1,
        )
        intersection_height = np.maximum(
            0,
            np.minimum(y2[index], y2[rest])
            - np.maximum(y1[index], y1[rest]) + 1,
        )
        intersection = intersection_width * intersection_height
        overlap = intersection / np.maximum(
            areas[index] + areas[rest] - intersection,
            1e-9,
        )
        order = rest[overlap <= threshold]
    return keep


def postprocess(
    raw_output: np.ndarray,
    ratio: float,
    image_width: int,
    image_height: int,
    confidence: float,
    nms_iou: float,
) -> list[tuple[Detection, int]]:
    predictions = decode_yolox(raw_output)[0]
    centers = predictions[:, :4]
    class_scores = predictions[:, 4:5] * predictions[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    valid = scores > confidence
    if not np.any(valid):
        return []

    centers = centers[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]
    boxes = np.empty_like(centers)
    boxes[:, 0] = centers[:, 0] - centers[:, 2] / 2
    boxes[:, 1] = centers[:, 1] - centers[:, 3] / 2
    boxes[:, 2] = centers[:, 0] + centers[:, 2] / 2
    boxes[:, 3] = centers[:, 1] + centers[:, 3] / 2
    boxes /= ratio
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, image_width)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, image_height)

    return [
        (
            Detection(
                float(boxes[index, 0]),
                float(boxes[index, 1]),
                float(boxes[index, 2]),
                float(boxes[index, 3]),
                float(scores[index]),
            ),
            int(class_ids[index]),
        )
        for index in nms(boxes, scores, nms_iou)
    ]


class ArabidopsisYoloX:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(
        self,
        image: np.ndarray,
        confidence: float,
        nms_iou: float,
    ) -> list[tuple[Detection, int]]:
        tensor, ratio = preprocess(image)
        raw_output = self.session.run(
            None,
            {self.input_name: tensor[None]},
        )[0]
        return postprocess(
            raw_output,
            ratio,
            image.shape[1],
            image.shape[0],
            confidence,
            nms_iou,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    paths = image_paths(args.input)
    if not paths:
        parser.error(f"No supported images found in {args.input}")

    model = ArabidopsisYoloX(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    count_rows = []
    detection_rows = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        predictions = model.infer(image, args.confidence, args.nms_iou)
        detections = [detection for detection, _class_id in predictions]
        open_count = sum(class_id == 0 for _detection, class_id in predictions)
        closed_count = sum(class_id == 1 for _detection, class_id in predictions)
        count_rows.append({
            "filename": path.name,
            "stomata_count": len(predictions),
            "open_count": open_count,
            "closed_count": closed_count,
        })
        for detection_id, (detection, class_id) in enumerate(predictions, 1):
            detection_rows.append({
                "filename": path.name,
                "detection_id": detection_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": round(detection.confidence, 6),
                "x1": round(detection.x1, 2),
                "y1": round(detection.y1, 2),
                "x2": round(detection.x2, 2),
                "y2": round(detection.y2, 2),
            })
        draw_overlay(
            image,
            detections,
            args.output / "overlays" / f"{path.stem}_overlay.jpg",
        )
        print(
            f"{path.name}: total={len(predictions)} "
            f"open={open_count} closed={closed_count}"
        )

    with (args.output / "counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "stomata_count",
                "open_count",
                "closed_count",
            ],
        )
        writer.writeheader()
        writer.writerows(count_rows)
    with (args.output / "detections.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "detection_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(detection_rows)


if __name__ == "__main__":
    main()
