#!/usr/bin/env python3
"""Score detector proposals for a central, elongated stomatal pore."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_MODEL = Path("models/external/arabidopsis_pore_unet.onnx")
PATCH_SIZE = 64


def proposal_patch(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    context_scale: float = 1.15,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 4.0) * context_scale
    left = max(0, int(round(center_x - side / 2)))
    top = max(0, int(round(center_y - side / 2)))
    right = min(width, int(round(center_x + side / 2)))
    bottom = min(height, int(round(center_y + side / 2)))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"Degenerate proposal crop: {box}")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(
        rgb,
        (PATCH_SIZE, PATCH_SIZE),
        interpolation=cv2.INTER_AREA
        if max(crop.shape[:2]) >= PATCH_SIZE else cv2.INTER_CUBIC,
    )


def mask_metrics(probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    binary = (probability >= threshold).astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return {
            "pore_max_probability": float(probability.max()),
            "pore_mean_probability": float(probability.mean()),
            "pore_component_count": 0,
            "pore_area_ratio": 0.0,
            "pore_aspect_ratio": 0.0,
            "pore_centroid_offset": 1.0,
            "pore_solidity": 0.0,
            "pore_score": 0.0,
        }

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1
    area = int(stats[largest_label, cv2.CC_STAT_AREA])
    component_width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    component_height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    aspect_ratio = min(component_width, component_height) / max(
        component_width,
        component_height,
        1,
    )
    centroid = centroids[largest_label]
    center = (PATCH_SIZE - 1) / 2
    centroid_offset = float(
        np.linalg.norm(centroid - center)
        / (np.sqrt(2) * PATCH_SIZE / 2)
    )
    component_mask = (labels == largest_label).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    hull_area = (
        cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
        if contours else 0.0
    )
    solidity = float(area / max(hull_area, 1.0))
    area_ratio = area / float(PATCH_SIZE * PATCH_SIZE)
    centrality = max(0.0, 1.0 - centroid_offset)
    size_weight = (
        min(1.0, area_ratio / 0.01)
        * min(1.0, 0.35 / max(area_ratio, 1e-9))
    )
    elongation_weight = max(0.0, 1.0 - aspect_ratio)
    score = (
        float(probability[labels == largest_label].mean())
        * centrality
        * size_weight
        * (0.5 + 0.5 * elongation_weight)
    )
    return {
        "pore_max_probability": float(probability.max()),
        "pore_mean_probability": float(probability.mean()),
        "pore_component_count": int(component_count - 1),
        "pore_area_ratio": float(area_ratio),
        "pore_aspect_ratio": float(aspect_ratio),
        "pore_centroid_offset": float(centroid_offset),
        "pore_solidity": solidity,
        "pore_score": float(score),
    }


class PoreVerifier:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer_batch(self, patches: list[np.ndarray]) -> list[np.ndarray]:
        if not patches:
            return []
        batch = np.asarray(patches, dtype=np.float32).transpose(0, 3, 1, 2)
        return list(self.session.run(None, {self.input_name: batch})[0][:, 0])


def score_predictions(
    images_dir: Path,
    predictions_path: Path,
    model_path: Path,
    output_path: Path,
    context_scale: float,
) -> None:
    with predictions_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = {"filename", "x1", "y1", "x2", "y2"}
    if not required.issubset(fieldnames):
        raise ValueError("Prediction CSV lacks coordinate columns")

    patches = []
    image_cache: dict[str, np.ndarray] = {}
    for row in rows:
        filename = row["filename"]
        if filename not in image_cache:
            image_cache[filename] = cv2.imread(
                str(images_dir / filename),
                cv2.IMREAD_COLOR,
            )
            if image_cache[filename] is None:
                raise ValueError(f"Could not read {images_dir / filename}")
        box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        patches.append(proposal_patch(image_cache[filename], box, context_scale))

    verifier = PoreVerifier(model_path)
    probabilities = []
    for start in range(0, len(patches), 256):
        probabilities.extend(verifier.infer_batch(patches[start:start + 256]))
    metric_rows = [mask_metrics(item) for item in probabilities]
    metric_fields = list(metric_rows[0]) if metric_rows else list(mask_metrics(
        np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + metric_fields)
        writer.writeheader()
        for row, metrics in zip(rows, metric_rows):
            writer.writerow(row | {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in metrics.items()
            })
    print(f"Wrote pore-structure scores for {len(rows)} proposals to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--context-scale", type=float, default=1.15)
    args = parser.parse_args()
    score_predictions(
        args.images,
        args.predictions,
        args.model,
        args.output,
        args.context_scale,
    )


if __name__ == "__main__":
    main()
