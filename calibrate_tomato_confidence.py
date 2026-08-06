#!/usr/bin/env python3
"""Calibrate tomato detector scores on validation plants and freeze for test."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from evaluate_external_predictions import (
    Box,
    box_iou,
    evaluate,
    load_ground_truth,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "confidence", "x1", "y1", "x2", "y2"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} lacks required prediction fields")
    return rows


def row_box(row: dict[str, str], score_field: str = "confidence") -> Box:
    return Box(
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
        float(row[score_field]),
    )


def label_rows(
    rows: list[dict[str, str]],
    ground_truth: dict[str, list[Box]],
    iou_threshold: float = 0.5,
) -> list[dict[str, object]]:
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["filename"] not in ground_truth:
            raise ValueError(
                f"Prediction image absent from ground truth: {row['filename']}"
            )
        by_image[row["filename"]].append(row)
    labeled = []
    for filename in sorted(ground_truth):
        unmatched = set(range(len(ground_truth[filename])))
        image_rows = sorted(
            by_image.get(filename, []),
            key=lambda row: float(row["confidence"]),
            reverse=True,
        )
        for row in image_rows:
            prediction = row_box(row)
            candidates = [
                (index, box_iou(prediction, ground_truth[filename][index]))
                for index in unmatched
            ]
            best_index, best_iou = (
                max(candidates, key=lambda item: item[1])
                if candidates else (-1, 0.0)
            )
            is_true_positive = best_iou >= iou_threshold
            if is_true_positive:
                unmatched.remove(best_index)
            labeled.append({
                **row,
                "raw_detector_score": float(row["confidence"]),
                "match_is_true_positive": int(is_true_positive),
                "match_iou": float(best_iou),
            })
    return labeled


def score_feature(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_platt(rows: list[dict[str, object]]) -> LogisticRegression:
    labels = np.asarray(
        [int(row["match_is_true_positive"]) for row in rows],
        dtype=int,
    )
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration requires both true and false candidates")
    scores = np.asarray(
        [float(row["raw_detector_score"]) for row in rows],
        dtype=float,
    )
    model = LogisticRegression(C=1.0, solver="lbfgs")
    model.fit(score_feature(scores), labels)
    return model


def predict_platt(
    model: LogisticRegression,
    scores: np.ndarray,
) -> np.ndarray:
    return model.predict_proba(score_feature(scores))[:, 1]


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    total = len(labels)
    if total == 0:
        return float("nan")
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (
            (probabilities >= lower) & (probabilities < upper)
            if index < bins - 1
            else (probabilities >= lower) & (probabilities <= upper)
        )
        if not selected.any():
            continue
        accuracy = float(labels[selected].mean())
        confidence = float(probabilities[selected].mean())
        error += selected.sum() / total * abs(accuracy - confidence)
    return float(error)


def calibration_metrics(
    rows: list[dict[str, object]],
    probability_field: str,
) -> dict[str, float | int]:
    labels = np.asarray(
        [int(row["match_is_true_positive"]) for row in rows],
        dtype=int,
    )
    probabilities = np.asarray(
        [float(row[probability_field]) for row in rows],
        dtype=float,
    )
    return {
        "candidates": len(rows),
        "true_candidates": int(labels.sum()),
        "false_candidates": int(len(labels) - labels.sum()),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(
            log_loss(labels, np.clip(probabilities, 1e-6, 1 - 1e-6))
        ),
        "ece_10_bin": expected_calibration_error(labels, probabilities),
    }


def cross_validated_probabilities(
    rows: list[dict[str, object]],
) -> np.ndarray:
    output = np.full(len(rows), np.nan, dtype=float)
    filenames = sorted({str(row["filename"]) for row in rows})
    for filename in filenames:
        train_indices = [
            index for index, row in enumerate(rows)
            if row["filename"] != filename
        ]
        test_indices = [
            index for index, row in enumerate(rows)
            if row["filename"] == filename
        ]
        training = [rows[index] for index in train_indices]
        labels = {
            int(row["match_is_true_positive"]) for row in training
        }
        if labels != {0, 1}:
            output[test_indices] = [
                float(rows[index]["raw_detector_score"])
                for index in test_indices
            ]
            continue
        model = fit_platt(training)
        scores = np.asarray([
            float(rows[index]["raw_detector_score"])
            for index in test_indices
        ])
        output[test_indices] = predict_platt(model, scores)
    if not np.isfinite(output).all():
        raise ValueError("Cross-validation did not score every candidate")
    return output


def select_threshold(
    rows: list[dict[str, object]],
    total_truth: int,
    *,
    minimum_precision: float | None = None,
) -> float:
    ranked = sorted(
        [
            (
            float(row["p_stoma"]),
            bool(row["match_is_true_positive"]),
            )
            for row in rows
        ],
        reverse=True,
    )
    true_positive = 0
    false_positive = 0
    best_key = (-1.0, -1.0, -1.0)
    best_threshold = 1.0
    for index, (score, is_true_positive) in enumerate(ranked):
        true_positive += int(is_true_positive)
        false_positive += int(not is_true_positive)
        next_score = ranked[index + 1][0] if index + 1 < len(ranked) else -1
        if next_score == score:
            continue
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(total_truth, 1)
        if minimum_precision is not None and precision < minimum_precision:
            continue
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        key = (
            recall if minimum_precision is not None else f1,
            precision,
            score,
        )
        if key > best_key:
            best_key = key
            best_threshold = score
    return best_threshold


def boxes_by_image(
    rows: list[dict[str, object]],
    threshold: float,
) -> dict[str, list[Box]]:
    result: dict[str, list[Box]] = defaultdict(list)
    for row in rows:
        if float(row["p_stoma"]) < threshold:
            continue
        result[str(row["filename"])].append(
            Box(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                float(row["p_stoma"]),
            )
        )
    return dict(result)


def apply_model(
    model: LogisticRegression,
    rows: list[dict[str, object]],
) -> None:
    scores = np.asarray(
        [float(row["raw_detector_score"]) for row in rows],
        dtype=float,
    )
    probabilities = predict_platt(model, scores)
    for row, probability in zip(rows, probabilities):
        row["p_stoma"] = float(probability)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-ground-truth", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-ground-truth", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-floor", type=float, default=0.10)
    args = parser.parse_args()

    validation_truth, _ = load_ground_truth(args.validation_ground_truth)
    test_truth, _ = load_ground_truth(args.test_ground_truth)
    validation_rows = label_rows(
        read_rows(args.validation_predictions),
        validation_truth,
    )
    test_rows = label_rows(read_rows(args.test_predictions), test_truth)

    raw_validation_rows = [
        {**row, "raw_probability": row["raw_detector_score"]}
        for row in validation_rows
    ]
    cv_probabilities = cross_validated_probabilities(validation_rows)
    for row, probability in zip(validation_rows, cv_probabilities):
        row["cv_p_stoma"] = float(probability)

    model = fit_platt(validation_rows)
    apply_model(model, validation_rows)
    apply_model(model, test_rows)
    validation_truth_count = sum(
        len(boxes) for boxes in validation_truth.values()
    )
    max_f1_threshold = select_threshold(
        validation_rows,
        validation_truth_count,
    )
    precision_threshold = select_threshold(
        validation_rows,
        validation_truth_count,
        minimum_precision=0.90,
    )
    accepted_threshold = precision_threshold
    for rows in (validation_rows, test_rows):
        for row in rows:
            probability = float(row["p_stoma"])
            row["decision_status"] = (
                "accepted"
                if probability >= accepted_threshold
                else "review"
                if probability >= args.review_floor
                else "rejected"
            )

    validation_max_f1, _ = evaluate(
        boxes_by_image(validation_rows, max_f1_threshold),
        validation_truth,
        {},
        0.50,
    )
    test_max_f1, test_per_image = evaluate(
        boxes_by_image(test_rows, max_f1_threshold),
        test_truth,
        {},
        0.50,
    )
    validation_precision, _ = evaluate(
        boxes_by_image(validation_rows, precision_threshold),
        validation_truth,
        {},
        0.50,
    )
    test_precision, _ = evaluate(
        boxes_by_image(test_rows, precision_threshold),
        test_truth,
        {},
        0.50,
    )
    validation_review, _ = evaluate(
        boxes_by_image(validation_rows, args.review_floor),
        validation_truth,
        {},
        0.50,
    )
    test_review, _ = evaluate(
        boxes_by_image(test_rows, args.review_floor),
        test_truth,
        {},
        0.50,
    )

    report = {
        "calibration_method": "Platt logistic regression on logit(raw score)",
        "calibration_scope": (
            "Conditional probability that a retained fused candidate matches "
            "an expert box at IoU >= 0.50."
        ),
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "review_floor": args.review_floor,
        "max_f1_threshold": max_f1_threshold,
        "accepted_threshold_precision_target_0_90": precision_threshold,
        "validation_raw_calibration": calibration_metrics(
            raw_validation_rows,
            "raw_probability",
        ),
        "validation_leave_one_image_out_calibration": calibration_metrics(
            validation_rows,
            "cv_p_stoma",
        ),
        "validation_apparent_calibration": calibration_metrics(
            validation_rows,
            "p_stoma",
        ),
        "test_frozen_calibration": calibration_metrics(
            test_rows,
            "p_stoma",
        ),
        "validation_at_max_f1": validation_max_f1,
        "test_at_frozen_max_f1": test_max_f1,
        "validation_at_precision_target": validation_precision,
        "test_at_frozen_precision_target": test_precision,
        "validation_at_review_floor": validation_review,
        "test_at_frozen_review_floor": test_review,
        "limitations": [
            "Calibration is conditional on candidates above the proposal floor.",
            "Only six validation images from three varieties are available.",
            "The untouched test split was not used to fit or select thresholds.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "validation_calibrated.csv", validation_rows)
    write_csv(args.output / "test_calibrated.csv", test_rows)
    write_csv(args.output / "test_per_image.csv", test_per_image)
    (args.output / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
