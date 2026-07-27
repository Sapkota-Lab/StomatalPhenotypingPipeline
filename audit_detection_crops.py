#!/usr/bin/env python3
"""Create labeled contact sheets for auditing detection CSV rows."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np


def load_rows(
    detections_csv: Path,
    status: str | None,
    source: str | None,
) -> list[dict[str, str]]:
    with detections_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if status:
        rows = [row for row in rows if row["status"] == status]
    if source:
        rows = [row for row in rows if row["source"] == source]
    return rows


def crop_detection(
    image: np.ndarray,
    row: dict[str, str],
    padding: float,
) -> np.ndarray:
    x1, y1, x2, y2 = (
        float(row[name]) for name in ("x1", "y1", "x2", "y2")
    )
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = width * padding
    pad_y = height * padding
    left = max(0, math.floor(x1 - pad_x))
    top = max(0, math.floor(y1 - pad_y))
    right = min(image.shape[1], math.ceil(x2 + pad_x))
    bottom = min(image.shape[0], math.ceil(y2 + pad_y))
    return image[top:bottom, left:right]


def make_sheet(
    rows: list[dict[str, str]],
    image_dir: Path,
    output: Path,
    cell_size: int,
    columns: int,
    padding: float,
) -> None:
    label_height = 34
    rows_count = math.ceil(len(rows) / columns)
    sheet = np.full(
        (rows_count * (cell_size + label_height), columns * cell_size, 3),
        245,
        dtype=np.uint8,
    )
    image_cache: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        filename = row["filename"]
        image = image_cache.get(filename)
        if image is None:
            image = cv2.imread(str(image_dir / filename), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read {image_dir / filename}")
            image_cache[filename] = image
        crop = crop_detection(image, row, padding)
        if crop.size == 0:
            continue
        scale = min(cell_size / crop.shape[1], cell_size / crop.shape[0])
        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        row_index, column_index = divmod(index, columns)
        x = column_index * cell_size + (cell_size - resized.shape[1]) // 2
        y = (
            row_index * (cell_size + label_height)
            + (cell_size - resized.shape[0]) // 2
        )
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        confidence = next(
            (
                row[name]
                for name in (
                    "stomaquant_confidence",
                    "yolos_confidence",
                    "stomadet_confidence",
                )
                if row.get(name)
            ),
            "",
        )
        label = f"{filename.removesuffix('.JPG')} #{row['detection_id']} {confidence}"
        cv2.putText(
            sheet,
            label,
            (
                column_index * cell_size + 5,
                (row_index + 1) * (cell_size + label_height) - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", choices=("accepted", "review"))
    parser.add_argument("--source")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=220)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--padding", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_rows(args.detections, args.status, args.source)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    if not rows:
        parser.error("No detections matched the requested filters")
    make_sheet(
        rows,
        args.images,
        args.output,
        args.cell_size,
        args.columns,
        args.padding,
    )
    print(f"Wrote {len(rows)} crops to {args.output}")


if __name__ == "__main__":
    main()
