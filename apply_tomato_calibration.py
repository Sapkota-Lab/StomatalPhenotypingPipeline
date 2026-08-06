#!/usr/bin/env python3
"""Apply a frozen tomato confidence mapping to fused detector candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def calibrated_probability(
    raw_score: float,
    coefficient: float,
    intercept: float,
) -> float:
    score = min(max(float(raw_score), 1e-6), 1 - 1e-6)
    logit = math.log(score / (1.0 - score))
    value = coefficient * logit + intercept
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def apply_calibration(
    rows: list[dict[str, str]],
    report: dict,
) -> list[dict[str, object]]:
    coefficient = float(report["coefficient"])
    intercept = float(report["intercept"])
    accepted_threshold = float(
        report["accepted_threshold_precision_target_0_90"]
    )
    review_floor = float(report["review_floor"])
    output = []
    for row in rows:
        raw_score = float(row["confidence"])
        probability = calibrated_probability(
            raw_score,
            coefficient,
            intercept,
        )
        decision = (
            "accepted"
            if probability >= accepted_threshold
            else "review"
            if probability >= review_floor
            else "rejected"
        )
        output.append({
            **row,
            "raw_detector_score": raw_score,
            "p_stoma": probability,
            "decision_status": decision,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Prediction table is empty")
    report = json.loads(args.calibration_report.read_text())
    calibrated = apply_calibration(rows, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(calibrated[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(calibrated)
    counts = {
        status: sum(
            row["decision_status"] == status for row in calibrated
        )
        for status in ("accepted", "review", "rejected")
    }
    print(f"Wrote {len(calibrated)} calibrated candidates: {counts}")


if __name__ == "__main__":
    main()
