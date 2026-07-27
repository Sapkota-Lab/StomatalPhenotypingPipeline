#!/usr/bin/env python3
"""Replay ensemble classification from saved detector CSV files.

This avoids repeating expensive model inference when calibrating agreement
rules against an external labelled benchmark.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ensemble_stomata_engine import (
    DETECTION_FIELDS,
    classify_ensemble,
    write_csv_atomic,
)
from stomata_engine import Detection


def load_detection_csv(
    path: Path,
    minimum_confidence: float = 0.0,
) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "confidence", "x1", "y1", "x2", "y2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} lacks standalone detection columns")
        for row in reader:
            confidence = float(row["confidence"])
            if confidence < minimum_confidence:
                continue
            grouped.setdefault(row["filename"], []).append(Detection(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                confidence,
            ))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stomaquant", type=Path, required=True)
    parser.add_argument("--yolos", type=Path, required=True)
    parser.add_argument("--stomadet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stomaquant-confidence", type=float, default=0.05)
    parser.add_argument("--yolos-confidence", type=float, default=0.05)
    parser.add_argument("--stomadet-confidence", type=float, default=0.05)
    parser.add_argument("--yolos-agreement", type=float, default=0.85)
    parser.add_argument(
        "--stomadet-yolos-agreement",
        type=float,
        default=0.92,
    )
    parser.add_argument("--yolos-only-review", type=float, default=0.95)
    parser.add_argument("--agreement-iou", type=float, default=0.05)
    parser.add_argument("--agreement-center-distance", type=float, default=0.55)
    args = parser.parse_args()

    stomaquant = load_detection_csv(
        args.stomaquant,
        args.stomaquant_confidence,
    )
    yolos = load_detection_csv(args.yolos, args.yolos_confidence)
    stomadet = load_detection_csv(args.stomadet, args.stomadet_confidence)
    filenames = sorted(set(stomaquant) | set(yolos) | set(stomadet))

    rows: list[dict] = []
    for filename in filenames:
        classified = classify_ensemble(
            stomaquant.get(filename, []),
            yolos.get(filename, []),
            stomadet.get(filename, []),
            yolos_agreement=args.yolos_agreement,
            stomadet_yolos_agreement=args.stomadet_yolos_agreement,
            yolos_only_review=args.yolos_only_review,
            agreement_iou=args.agreement_iou,
            agreement_center_distance=args.agreement_center_distance,
        )
        for detection_id, item in enumerate(classified, 1):
            rows.append({
                "filename": filename,
                "detection_id": detection_id,
                "status": item.status,
                "source": item.source,
                "stomaquant_confidence": (
                    round(item.detection.confidence, 6)
                    if item.source in {
                        "stomaquant_yolos",
                        "stomaquant_only",
                    }
                    else ""
                ),
                "yolos_confidence": (
                    round(item.yolos_confidence, 6)
                    if item.yolos_confidence is not None else ""
                ),
                "stomadet_confidence": (
                    round(item.stomadet_confidence, 6)
                    if item.stomadet_confidence is not None else ""
                ),
                "x1": round(item.detection.x1, 2),
                "y1": round(item.detection.y1, 2),
                "x2": round(item.detection.x2, 2),
                "y2": round(item.detection.y2, 2),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(args.output, DETECTION_FIELDS, rows)
    accepted = sum(row["status"] == "accepted" for row in rows)
    print(f"Wrote {accepted} accepted and {len(rows) - accepted} review rows")


if __name__ == "__main__":
    main()
