#!/usr/bin/env python3
"""Evaluate pose-detector CSV output against COCO boxes and keypoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class GroundTruth:
    box: np.ndarray
    keypoints: np.ndarray


@dataclass
class Prediction:
    image: str
    box: np.ndarray
    confidence: float
    keypoints: np.ndarray
    morphology_plausible: bool


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(path: Path) -> dict[str, list[GroundTruth]]:
    document = json.loads(path.read_text())
    image_names = {
        int(image["id"]): Path(image["file_name"]).name for image in document["images"]
    }
    output: dict[str, list[GroundTruth]] = defaultdict(list)
    for annotation in document["annotations"]:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        keypoints = np.asarray(annotation["keypoints"], dtype=float).reshape(4, 3)
        output[image_names[int(annotation["image_id"])]].append(
            GroundTruth(
                box=np.asarray([x, y, x + width, y + height], dtype=float),
                keypoints=keypoints,
            )
        )
    for image_name in image_names.values():
        output.setdefault(image_name, [])
    return dict(output)


def load_predictions(path: Path) -> list[Prediction]:
    output: list[Prediction] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            keypoints = np.asarray(
                [
                    [
                        float(row[f"p{index}_x"]),
                        float(row[f"p{index}_y"]),
                        float(row[f"p{index}_confidence"]),
                    ]
                    for index in range(4)
                ]
            )
            output.append(
                Prediction(
                    image=row["image"],
                    box=np.asarray(
                        [
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        ]
                    ),
                    confidence=float(row["confidence"]),
                    keypoints=keypoints,
                    morphology_plausible=row["morphology_plausible"].lower()
                    in {"true", "1", "yes"},
                )
            )
    return output


def pair_invariant_keypoint_errors(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    predicted_xy = prediction[:, :2]
    truth_xy = truth[:, :2]

    direct_length = np.linalg.norm(predicted_xy[:2] - truth_xy[:2], axis=1)
    swapped_length = np.linalg.norm(predicted_xy[:2] - truth_xy[1::-1], axis=1)
    length_errors = (
        direct_length if direct_length.sum() <= swapped_length.sum() else swapped_length
    )

    direct_width = np.linalg.norm(predicted_xy[2:] - truth_xy[2:], axis=1)
    swapped_width = np.linalg.norm(predicted_xy[2:] - truth_xy[3:1:-1], axis=1)
    width_errors = (
        direct_width if direct_width.sum() <= swapped_width.sum() else swapped_width
    )
    return np.concatenate([length_errors, width_errors])


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    levels = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= level])) if np.any(recall >= level) else 0.0
        for level in levels
    ]
    return float(np.mean(interpolated))


def evaluate(
    ground_truth: dict[str, list[GroundTruth]],
    predictions: Iterable[Prediction],
    *,
    confidence: float,
    morphology_only: bool,
    iou_threshold: float = 0.50,
) -> dict[str, float | int | bool]:
    selected = [
        prediction
        for prediction in predictions
        if prediction.image in ground_truth
        and prediction.confidence >= confidence
        and (prediction.morphology_plausible or not morphology_only)
    ]
    selected.sort(key=lambda prediction: prediction.confidence, reverse=True)

    matched: dict[str, set[int]] = defaultdict(set)
    true_positive: list[int] = []
    false_positive: list[int] = []
    normalized_keypoint_errors: list[float] = []
    image_prediction_counts: dict[str, int] = defaultdict(int)

    for prediction in selected:
        image_prediction_counts[prediction.image] += 1
        candidates = ground_truth[prediction.image]
        best_index = -1
        best_iou = 0.0
        for index, truth in enumerate(candidates):
            if index in matched[prediction.image]:
                continue
            overlap = box_iou(prediction.box, truth.box)
            if overlap > best_iou:
                best_index = index
                best_iou = overlap

        if best_index >= 0 and best_iou >= iou_threshold:
            true_positive.append(1)
            false_positive.append(0)
            matched[prediction.image].add(best_index)
            truth = candidates[best_index]
            diagonal = float(
                np.linalg.norm(
                    [truth.box[2] - truth.box[0], truth.box[3] - truth.box[1]]
                )
            )
            if diagonal > 0:
                normalized_keypoint_errors.extend(
                    pair_invariant_keypoint_errors(
                        prediction.keypoints,
                        truth.keypoints,
                    )
                    / diagonal
                )
        else:
            true_positive.append(0)
            false_positive.append(1)

    total_ground_truth = sum(len(items) for items in ground_truth.values())
    cumulative_tp = np.cumsum(true_positive, dtype=float)
    cumulative_fp = np.cumsum(false_positive, dtype=float)
    recall_curve = cumulative_tp / max(1, total_ground_truth)
    precision_curve = cumulative_tp / np.maximum(1.0, cumulative_tp + cumulative_fp)

    tp = int(sum(true_positive))
    fp = int(sum(false_positive))
    fn = total_ground_truth - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_ground_truth if total_ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    count_errors = [
        image_prediction_counts[image] - len(truths)
        for image, truths in ground_truth.items()
    ]
    keypoint_errors = np.asarray(normalized_keypoint_errors)
    return {
        "confidence": confidence,
        "morphology_only": morphology_only,
        "images": len(ground_truth),
        "ground_truth": total_ground_truth,
        "predictions": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": average_precision(recall_curve, precision_curve),
        "count_mae": float(np.mean(np.abs(count_errors))) if count_errors else 0.0,
        "count_bias": float(np.mean(count_errors)) if count_errors else 0.0,
        "keypoint_nme": float(np.mean(keypoint_errors))
        if keypoint_errors.size
        else math.nan,
        "keypoint_pck_0_10": float(np.mean(keypoint_errors <= 0.10))
        if keypoint_errors.size
        else math.nan,
        "keypoint_pck_0_20": float(np.mean(keypoint_errors <= 0.20))
        if keypoint_errors.size
        else math.nan,
    }


def best_f1_threshold(
    ground_truth: dict[str, list[GroundTruth]],
    predictions: list[Prediction],
    *,
    morphology_only: bool,
) -> dict[str, float | int | bool]:
    ranked = sorted(
        (
            prediction
            for prediction in predictions
            if prediction.image in ground_truth
            and (prediction.morphology_plausible or not morphology_only)
        ),
        key=lambda prediction: prediction.confidence,
        reverse=True,
    )
    if not ranked:
        return evaluate(
            ground_truth,
            predictions,
            confidence=math.inf,
            morphology_only=morphology_only,
        )

    total_ground_truth = sum(len(items) for items in ground_truth.values())
    matched: dict[str, set[int]] = defaultdict(set)
    cumulative_tp = 0
    cumulative_fp = 0
    best_key = (-1.0, -1.0, -1.0)
    best_threshold = ranked[0].confidence

    for index, prediction in enumerate(ranked):
        best_index = -1
        best_iou = 0.0
        for truth_index, truth in enumerate(ground_truth[prediction.image]):
            if truth_index in matched[prediction.image]:
                continue
            overlap = box_iou(prediction.box, truth.box)
            if overlap > best_iou:
                best_index = truth_index
                best_iou = overlap
        if best_index >= 0 and best_iou >= 0.50:
            matched[prediction.image].add(best_index)
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        next_score = (
            ranked[index + 1].confidence if index + 1 < len(ranked) else -math.inf
        )
        if next_score == prediction.confidence:
            continue

        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / total_ground_truth if total_ground_truth else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        key = (f1, precision, prediction.confidence)
        if key > best_key:
            best_key = key
            best_threshold = prediction.confidence

    return evaluate(
        ground_truth,
        predictions,
        confidence=best_threshold,
        morphology_only=morphology_only,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    if args.confidence is None:
        results = {
            "raw_best_f1": best_f1_threshold(
                ground_truth,
                predictions,
                morphology_only=False,
            ),
            "morphology_best_f1": best_f1_threshold(
                ground_truth,
                predictions,
                morphology_only=True,
            ),
        }
    else:
        results = {
            "raw": evaluate(
                ground_truth,
                predictions,
                confidence=args.confidence,
                morphology_only=False,
            ),
            "morphology": evaluate(
                ground_truth,
                predictions,
                confidence=args.confidence,
                morphology_only=True,
            ),
        }
    serialized = json.dumps(results, indent=2, allow_nan=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
