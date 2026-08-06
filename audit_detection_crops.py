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
from PIL import Image, ImageOps


def load_rows(
    detections_csv: Path,
    status: str | None,
    source: str | None,
    water_status: str | None = None,
) -> list[dict[str, str]]:
    with detections_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if status:
        rows = [
            row for row in rows
            if row.get("status", row.get("decision_status")) == status
        ]
    if source:
        rows = [
            row for row in rows
            if row.get("source", row.get("proposal_source")) == source
        ]
    if water_status:
        rows = [
            row for row in rows
            if row.get("water_status") == water_status
        ]
    return rows


def crop_detection(
    image: np.ndarray,
    row: dict[str, str],
    padding: float,
) -> np.ndarray:
    if {"x1", "y1", "x2", "y2"}.issubset(row):
        coordinate_fields = ("x1", "y1", "x2", "y2")
    else:
        coordinate_fields = (
            "bbox_x1_px",
            "bbox_y1_px",
            "bbox_x2_px",
            "bbox_y2_px",
        )
    x1, y1, x2, y2 = (
        float(row[name]) for name in coordinate_fields
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


def detection_coordinates(
    row: dict[str, str],
) -> tuple[float, float, float, float]:
    if {"x1", "y1", "x2", "y2"}.issubset(row):
        coordinate_fields = ("x1", "y1", "x2", "y2")
    else:
        coordinate_fields = (
            "bbox_x1_px",
            "bbox_y1_px",
            "bbox_x2_px",
            "bbox_y2_px",
        )
    return tuple(float(row[name]) for name in coordinate_fields)


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
        filename = row.get("filename", row.get("stomata_image", ""))
        image = image_cache.get(filename)
        if image is None:
            matches = [
                path for path in image_dir.rglob("*")
                if path.is_file() and path.name.lower() == filename.lower()
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one image named {filename}, found {len(matches)}"
                )
            with Image.open(matches[0]) as source_image:
                rgb = np.asarray(
                    ImageOps.exif_transpose(source_image).convert("RGB")
                )
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            image_cache[filename] = image
        annotated = image.copy()
        x1, y1, x2, y2 = detection_coordinates(row)
        status = row.get("status", row.get("decision_status", ""))
        color = (40, 190, 40) if status == "accepted" else (0, 165, 255)
        thickness = max(2, round(min(image.shape[:2]) / 1500))
        cv2.rectangle(
            annotated,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            color,
            thickness,
        )
        crop = crop_detection(annotated, row, padding)
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
                    "p_stoma",
                    "stomaquant_confidence",
                    "yolos_confidence",
                    "stomadet_confidence",
                )
                if row.get(name)
            ),
            "",
        )
        label = (
            f"{Path(filename).stem} #{row['detection_id']} "
            f"{confidence}".rstrip()
        )
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
    parser.add_argument(
        "--water-status",
        choices=("watered", "drought_stressed"),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=220)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--padding", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_rows(
        args.detections,
        args.status,
        args.source,
        args.water_status,
    )
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
