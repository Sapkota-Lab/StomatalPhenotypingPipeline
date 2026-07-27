#!/usr/bin/env python3
"""Calibrate on one labeled split and evaluate unchanged on another split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_external_predictions import (
    evaluate,
    load_ground_truth,
    load_predictions,
    match_image,
)


def select_f1_threshold(predictions, ground_truth) -> float:
    ranked: list[tuple[float, bool]] = []
    for filename, truth in ground_truth.items():
        ranked.extend(
            (box.score, is_true_positive)
            for box, is_true_positive, _iou in match_image(
                predictions.get(filename, []),
                truth,
                0.50,
            )
        )
    if not ranked:
        return float("inf")
    ranked.sort(key=lambda item: item[0], reverse=True)
    total_truth = sum(len(items) for items in ground_truth.values())
    true_positive = 0
    false_positive = 0
    best_key = (-1.0, -1.0, -1.0)
    best_threshold = ranked[0][0]
    for index, (score, is_true_positive) in enumerate(ranked):
        if is_true_positive:
            true_positive += 1
        else:
            false_positive += 1
        next_score = ranked[index + 1][0] if index + 1 < len(ranked) else -1.0
        if next_score == score:
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_truth if total_truth else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        key = (f1, precision, score)
        if key > best_key:
            best_key = key
            best_threshold = score
    return best_threshold


def load_all_predictions(path: Path):
    return load_predictions(
        path,
        status=None,
        source=None,
        confidence=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-ground-truth", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--test-ground-truth", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_truth, _ = load_ground_truth(args.calibration_ground_truth)
    calibration_predictions = load_all_predictions(args.calibration_predictions)
    threshold = select_f1_threshold(calibration_predictions, calibration_truth)

    calibrated_predictions = {
        filename: [box for box in boxes if box.score >= threshold]
        for filename, boxes in calibration_predictions.items()
    }
    calibration_summary, _ = evaluate(
        calibrated_predictions,
        calibration_truth,
        {},
        0.50,
    )

    test_truth, _ = load_ground_truth(args.test_ground_truth)
    test_predictions = {
        filename: [box for box in boxes if box.score >= threshold]
        for filename, boxes in load_all_predictions(args.test_predictions).items()
    }
    test_summary, test_rows = evaluate(test_predictions, test_truth, {}, 0.50)

    mean_test_count = (
        sum(len(items) for items in test_truth.values()) / max(1, len(test_truth))
    )
    gate = {
        "precision_at_least_0_90": test_summary["precision"] >= 0.90,
        "recall_at_least_0_85": test_summary["recall"] >= 0.85,
        "count_mae_at_most_10_percent": test_summary["count_mae"]
        <= max(3.0, 0.10 * mean_test_count),
        "absolute_bias_at_most_5_percent": abs(test_summary["count_bias"])
        <= max(2.0, 0.05 * mean_test_count),
    }
    gate["passed"] = all(gate.values())
    report = {
        "selected_confidence": threshold,
        "selection_rule": "maximum F1 on calibration split only",
        "calibration": calibration_summary,
        "test": test_summary,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
