#!/usr/bin/env python3
"""Rebuild legacy ensemble outputs using the two-model consensus rule.

This migration avoids repeating model inference when an existing detections
CSV already records whether each StomaQuant proposal had YOLOS agreement.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from ensemble_stomata_engine import (
    COUNT_FIELDS,
    DETECTION_FIELDS,
    ClassifiedDetection,
    draw_ensemble_overlay,
    draw_status_overlay,
    write_csv_atomic,
)
from stomata_engine import Detection


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def migrate_source(row: dict[str, str]) -> tuple[str, str]:
    source = row["source"]
    if source == "stomaquant":
        return (
            ("accepted", "stomaquant_yolos")
            if row.get("yolos_confidence")
            else ("review", "stomaquant_only")
        )
    if source == "consensus":
        return "accepted", "stomaquant_yolos"
    if source == "stomaquant_low":
        return "review", "stomaquant_only"
    if source == "stomadet_yolos":
        return "accepted", source
    if source == "yolos_only":
        return "review", source
    raise ValueError(f"Unsupported legacy source: {source}")


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-results", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_counts = read_csv(args.legacy_results / "counts.csv")
    old_detections = read_csv(args.legacy_results / "detections.csv")
    detections_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in old_detections:
        detections_by_file[row["filename"]].append(row)

    new_counts: list[dict[str, object]] = []
    new_detections: list[dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    for count_row in old_counts:
        filename = count_row["filename"]
        image = cv2.imread(str(args.images / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {args.images / filename}")
        if count_row["analysis_status"] == "out_of_scope":
            new_counts.append({
                "filename": filename,
                "analysis_status": "out_of_scope",
                "accepted_count": 0,
                "stomaquant_yolos_count": 0,
                "stomadet_yolos_count": 0,
                "review_count": 0,
                "stomaquant_only_review_count": 0,
                "yolos_only_review_count": 0,
                "image_quality": count_row["image_quality"],
                "focus_score": count_row["focus_score"],
                "contrast_score": count_row["contrast_score"],
                "clipped_fraction": count_row["clipped_fraction"],
            })
            draw_status_overlay(
                image,
                args.output / "overlays" / f"{Path(filename).stem}_overlay.jpg",
                "out_of_scope",
            )
            continue

        classified: list[ClassifiedDetection] = []
        for detection_id, row in enumerate(detections_by_file[filename], 1):
            status, source = migrate_source(row)
            detection = Detection(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                float(
                    row["stomaquant_confidence"]
                    or row["stomadet_confidence"]
                    or row["yolos_confidence"]
                ),
            )
            item = ClassifiedDetection(
                detection=detection,
                status=status,
                source=source,
                yolos_confidence=optional_float(row["yolos_confidence"]),
                stomadet_confidence=optional_float(row["stomadet_confidence"]),
            )
            classified.append(item)
            new_detections.append({
                **row,
                "detection_id": detection_id,
                "status": status,
                "source": source,
            })

        accepted = [item for item in classified if item.status == "accepted"]
        review = [item for item in classified if item.status == "review"]
        new_counts.append({
            "filename": filename,
            "analysis_status": count_row["analysis_status"],
            "accepted_count": len(accepted),
            "stomaquant_yolos_count": sum(
                item.source == "stomaquant_yolos" for item in accepted
            ),
            "stomadet_yolos_count": sum(
                item.source == "stomadet_yolos" for item in accepted
            ),
            "review_count": len(review),
            "stomaquant_only_review_count": sum(
                item.source == "stomaquant_only" for item in review
            ),
            "yolos_only_review_count": sum(
                item.source == "yolos_only" for item in review
            ),
            "image_quality": count_row["image_quality"],
            "focus_score": count_row["focus_score"],
            "contrast_score": count_row["contrast_score"],
            "clipped_fraction": count_row["clipped_fraction"],
        })
        draw_ensemble_overlay(
            image,
            classified,
            args.output / "overlays" / f"{Path(filename).stem}_overlay.jpg",
        )

    write_csv_atomic(args.output / "counts.csv", COUNT_FIELDS, new_counts)
    write_csv_atomic(
        args.output / "detections.csv",
        DETECTION_FIELDS,
        new_detections,
    )
    print(
        f"Wrote {len(new_counts)} images and {len(new_detections)} detections "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
