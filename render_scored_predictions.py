#!/usr/bin/env python3
"""Render accepted and review boxes from a scored prediction CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from stomata_engine import Detection, draw_overlay


def load_scored(
    path: Path,
    score_field: str,
    threshold: float,
) -> dict[str, tuple[list[Detection], list[Detection]]]:
    grouped: dict[str, tuple[list[Detection], list[Detection]]] = defaultdict(
        lambda: ([], [])
    )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if score_field not in (reader.fieldnames or []):
            raise ValueError(f"{path} lacks score field {score_field!r}")
        for row in reader:
            score = float(row[score_field])
            detection = Detection(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                score,
            )
            accepted, review = grouped[row["filename"]]
            (accepted if score >= threshold else review).append(detection)
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-field", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    grouped = load_scored(
        args.predictions,
        args.score_field,
        args.threshold,
    )
    for filename, (accepted, review) in grouped.items():
        image = cv2.imread(str(args.images / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {args.images / filename}")
        draw_overlay(
            image,
            accepted,
            args.output / "overlays" / f"{Path(filename).stem}_overlay.jpg",
            review,
        )
        print(
            f"{filename}: accepted={len(accepted)} review={len(review)}"
        )


if __name__ == "__main__":
    main()
