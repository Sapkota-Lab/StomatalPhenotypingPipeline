#!/usr/bin/env python3
"""Evaluate stomata detections against expert COCO bounding boxes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


def box_iou(first: Box, second: Box) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    return intersection / max(first_area + second_area - intersection, 1e-9)


def match_image(
    predictions: list[Box],
    ground_truth: list[Box],
    iou_threshold: float,
) -> list[tuple[Box, bool, float]]:
    unmatched = set(range(len(ground_truth)))
    matches: list[tuple[Box, bool, float]] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (index, box_iou(prediction, ground_truth[index]))
            for index in unmatched
        ]
        best_index, best_iou = (
            max(candidates, key=lambda item: item[1])
            if candidates else (-1, 0.0)
        )
        is_true_positive = best_iou >= iou_threshold
        if is_true_positive:
            unmatched.remove(best_index)
        matches.append((prediction, is_true_positive, best_iou))
    return matches


def interpolated_ap(
    ranked_true_positives: list[bool],
    ground_truth_count: int,
) -> float:
    if ground_truth_count == 0 or not ranked_true_positives:
        return 0.0
    true_positive = np.cumsum(np.asarray(ranked_true_positives, dtype=float))
    false_positive = np.cumsum(
        ~np.asarray(ranked_true_positives, dtype=bool)
    )
    recall = true_positive / ground_truth_count
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    return float(mean(
        max(
            (float(value) for value, rec in zip(precision, recall) if rec >= level),
            default=0.0,
        )
        for level in np.linspace(0.0, 1.0, 101)
    ))


def prediction_score(
    row: dict[str, str],
    score_field: str | None = None,
) -> float:
    if score_field:
        value = row.get(score_field, "").strip()
        if not value:
            raise ValueError(f"Prediction row lacks score field {score_field!r}")
        return float(value)
    values = []
    for field in (
        "confidence",
        "stomaquant_confidence",
        "yolos_confidence",
        "stomadet_confidence",
    ):
        value = row.get(field, "").strip()
        if value:
            values.append(float(value))
    return max(values, default=1.0)


def load_predictions(
    path: Path,
    status: str | None,
    source: str | None,
    confidence: float,
    min_view_support: int = 0,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = float("inf"),
    score_field: str | None = None,
) -> dict[str, list[Box]]:
    predictions: dict[str, list[Box]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "x1", "y1", "x2", "y2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} lacks prediction coordinate columns")
        for row in reader:
            if status and row.get("status") != status:
                continue
            if source and row.get("source") != source:
                continue
            score = prediction_score(row, score_field)
            if score < confidence:
                continue
            support = int(row.get("view_support") or 0)
            if support < min_view_support:
                continue
            area_ratio_value = row.get("area_ratio", "").strip()
            area_ratio = float(area_ratio_value) if area_ratio_value else 1.0
            if not min_area_ratio <= area_ratio <= max_area_ratio:
                continue
            predictions.setdefault(row["filename"], []).append(Box(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                score,
            ))
    return predictions


def load_ground_truth(
    path: Path,
) -> tuple[dict[str, list[Box]], dict[str, int]]:
    coco = json.loads(path.read_text())
    image_by_id = {
        item["id"]: Path(item["file_name"]).name for item in coco["images"]
    }
    ground_truth = {filename: [] for filename in image_by_id.values()}
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        ground_truth[image_by_id[annotation["image_id"]]].append(
            Box(x, y, x + width, y + height)
        )
    return ground_truth, image_by_id


def load_strata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        return {
            row["filename"]: row["quality_stratum"]
            for row in csv.DictReader(handle)
        }


def summarize_rows(rows: list[dict]) -> dict[str, float | int]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    errors = [int(row["count_error"]) for row in rows]
    return {
        "images": len(rows),
        "ground_truth": tp + fn,
        "predictions": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(
            2 * precision * recall / max(precision + recall, 1e-9),
            6,
        ),
        "count_mae": round(mean(abs(error) for error in errors), 6),
        "count_bias": round(mean(errors), 6),
    }


def evaluate(
    predictions: dict[str, list[Box]],
    ground_truth: dict[str, list[Box]],
    strata: dict[str, str],
    iou_threshold: float,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    ranked: list[tuple[float, bool]] = []
    for filename in sorted(ground_truth):
        predicted = predictions.get(filename, [])
        expected = ground_truth[filename]
        matches = match_image(predicted, expected, iou_threshold)
        tp = sum(is_tp for _box, is_tp, _iou in matches)
        fp = len(matches) - tp
        fn = len(expected) - tp
        rows.append({
            "filename": filename,
            "quality_stratum": strata.get(filename, "unknown"),
            "ground_truth_count": len(expected),
            "prediction_count": len(predicted),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "count_error": len(predicted) - len(expected),
        })
        ranked.extend(
            (box.score, is_tp) for box, is_tp, _iou in matches
        )

    summary = summarize_rows(rows)
    ranked.sort(key=lambda item: item[0], reverse=True)
    summary["ap50"] = round(
        interpolated_ap(
            [is_tp for _score, is_tp in ranked],
            int(summary["ground_truth"]),
        ),
        6,
    )
    by_quality = {}
    for quality in sorted({row["quality_stratum"] for row in rows}):
        by_quality[quality] = summarize_rows([
            row for row in rows if row["quality_stratum"] == quality
        ])
    summary["by_quality"] = by_quality
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status")
    parser.add_argument("--source")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--min-view-support", type=int, default=0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=float("inf"))
    parser.add_argument(
        "--score-field",
        help="Use one named CSV score column instead of the default model scores.",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    predictions = load_predictions(
        args.predictions,
        args.status,
        args.source,
        args.confidence,
        args.min_view_support,
        args.min_area_ratio,
        args.max_area_ratio,
        args.score_field,
    )
    ground_truth, _image_by_id = load_ground_truth(args.ground_truth)
    strata = load_strata(args.manifest)
    summary, rows = evaluate(
        predictions,
        ground_truth,
        strata,
        args.iou,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
