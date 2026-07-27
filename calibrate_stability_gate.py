#!/usr/bin/env python3
"""Calibrate an annotation-free TTA stability gate on accurate public images.

Stability is a secondary safeguard, not evidence of accuracy. The external
accuracy summary must pass first. Thresholds are then learned only from public
images whose individual detections are already acceptably accurate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_DIRECTIONS = {
    "geometric_count_cv": "maximum",
    "geometric_max_relative_deviation": "maximum",
    "identity_repeatability": "minimum",
    "geometric_consensus_fraction": "minimum",
    "contrast_dice": "minimum",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def accuracy_gate(summary: dict) -> tuple[bool, dict[str, float]]:
    mean_count = summary["ground_truth"] / max(summary["images"], 1)
    limits = {
        "minimum_precision": 0.90,
        "minimum_recall": 0.85,
        "maximum_count_mae": max(3.0, 0.10 * mean_count),
        "maximum_absolute_count_bias": max(2.0, 0.05 * mean_count),
    }
    passed = (
        summary["precision"] >= limits["minimum_precision"]
        and summary["recall"] >= limits["minimum_recall"]
        and summary["count_mae"] <= limits["maximum_count_mae"]
        and abs(summary["count_bias"])
        <= limits["maximum_absolute_count_bias"]
    )
    return passed, limits


def accurate_filenames(rows: list[dict[str, str]]) -> set[str]:
    accepted = set()
    for row in rows:
        tp = int(row["true_positive"])
        fp = int(row["false_positive"])
        fn = int(row["false_negative"])
        ground_truth = int(row["ground_truth_count"])
        error = int(row["count_error"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        count_tolerance = max(3.0, 0.10 * ground_truth)
        if (
            precision >= 0.90
            and recall >= 0.85
            and abs(error) <= count_tolerance
        ):
            accepted.add(row["filename"])
    return accepted


def calibrate(
    stability_rows: list[dict[str, str]],
    eligible: set[str],
    tail_probability: float = 0.05,
) -> dict[str, float]:
    selected = [
        row for row in stability_rows if row["filename"] in eligible
    ]
    if len(selected) < 4:
        raise ValueError(
            "Need at least four individually accurate calibration images; "
            f"found {len(selected)}"
        )
    thresholds = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = np.asarray([float(row[metric]) for row in selected])
        quantile = (
            1.0 - tail_probability
            if direction == "maximum"
            else tail_probability
        )
        thresholds[metric] = round(
            float(np.quantile(values, quantile)), 6
        )
    return thresholds


def apply_thresholds(
    rows: list[dict[str, str]],
    thresholds: dict[str, float],
) -> list[dict[str, str]]:
    audited = []
    for row in rows:
        failures = []
        for metric, direction in METRIC_DIRECTIONS.items():
            value = float(row[metric])
            threshold = thresholds[metric]
            failed = (
                value > threshold
                if direction == "maximum"
                else value < threshold
            )
            if failed:
                failures.append(metric)
        audited.append({
            **row,
            "stability_status": "pass" if not failures else "review",
            "failed_stability_metrics": ";".join(failures),
        })
    return audited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-stability", type=Path, required=True)
    parser.add_argument("--calibration-accuracy", type=Path, required=True)
    parser.add_argument("--accuracy-summary", type=Path, required=True)
    parser.add_argument("--output-thresholds", type=Path, required=True)
    parser.add_argument("--apply-stability", type=Path)
    parser.add_argument("--output-audit", type=Path)
    parser.add_argument("--tail-probability", type=float, default=0.05)
    args = parser.parse_args()

    if not 0.0 < args.tail_probability < 0.5:
        parser.error("tail-probability must be between 0 and 0.5")
    if bool(args.apply_stability) != bool(args.output_audit):
        parser.error(
            "apply-stability and output-audit must be supplied together"
        )

    report = json.loads(args.accuracy_summary.read_text())
    summary = report.get("test", report)
    passed, limits = accuracy_gate(summary)
    if not passed:
        raise SystemExit(
            "External accuracy gate failed; stability calibration is invalid."
        )
    eligible = accurate_filenames(read_rows(args.calibration_accuracy))
    thresholds = calibrate(
        read_rows(args.calibration_stability),
        eligible,
        args.tail_probability,
    )
    payload = {
        "warning": (
            "Transformation stability is a secondary safeguard and does not "
            "establish biological correctness."
        ),
        "external_accuracy_gate_passed": True,
        "external_accuracy_limits": limits,
        "eligible_calibration_images": len(eligible),
        "tail_probability": args.tail_probability,
        "thresholds": thresholds,
    }
    args.output_thresholds.parent.mkdir(parents=True, exist_ok=True)
    args.output_thresholds.write_text(json.dumps(payload, indent=2) + "\n")

    if args.apply_stability:
        rows = apply_thresholds(
            read_rows(args.apply_stability),
            thresholds,
        )
        args.output_audit.parent.mkdir(parents=True, exist_ok=True)
        with args.output_audit.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
