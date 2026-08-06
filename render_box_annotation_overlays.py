#!/usr/bin/env python3
"""Render full-field expert-box overlays and a contact sheet for QC."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from build_stomata_ledger import find_image, load_oriented_rgb


def render(
    image: np.ndarray,
    boxes: list[dict[str, float]],
    label: str,
) -> np.ndarray:
    view = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    thickness = max(2, round(max(view.shape[:2]) / 1800))
    font_scale = max(0.7, max(view.shape[:2]) / 5000)
    for index, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = (
            int(round(float(box[field])))
            for field in ("x1", "y1", "x2", "y2")
        )
        cv2.rectangle(view, (x1, y1), (x2, y2), (30, 230, 30), thickness)
        cv2.putText(
            view,
            str(index),
            (x1, max(20, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (30, 230, 30),
            thickness,
            cv2.LINE_AA,
        )
    banner_height = max(70, round(view.shape[0] * 0.025))
    cv2.rectangle(
        view,
        (0, 0),
        (view.shape[1], banner_height),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        view,
        label,
        (20, round(banner_height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.8, banner_height / 55),
        (255, 255, 255),
        max(2, round(banner_height / 35)),
        cv2.LINE_AA,
    )
    return view


def make_contact_sheet(
    previews: list[np.ndarray],
    output: Path,
    *,
    cell_width: int = 900,
    columns: int = 4,
) -> None:
    if not previews:
        raise ValueError("No annotation previews were rendered")
    resized = []
    for image in previews:
        scale = cell_width / image.shape[1]
        resized.append(
            cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        )
    cell_height = max(image.shape[0] for image in resized)
    rows = math.ceil(len(resized) / columns)
    sheet = np.full(
        (rows * cell_height, columns * cell_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, image in enumerate(resized):
        row, column = divmod(index, columns)
        y = row * cell_height
        x = column * cell_width
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("box_annotations"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("Stomata_Images"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/combined/annotation_overlays"),
    )
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = {
            row["stomata_image"]: row for row in csv.DictReader(handle)
        }
    previews = []
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, metadata in sorted(manifest.items()):
        annotation_path = (
            args.annotations / f"{Path(filename).stem}.json"
        )
        annotation = json.loads(annotation_path.read_text())
        image = load_oriented_rgb(find_image(args.images, filename))
        boxes = annotation.get("boxes", [])
        label = (
            f"{filename} | {metadata['variety']} | "
            f"{metadata['water_status']} | n={len(boxes)}"
        )
        overlay = render(image, boxes, label)
        output_path = args.output / f"{Path(filename).stem}_overlay.jpg"
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f"Could not write {output_path}")
        previews.append(overlay)
    make_contact_sheet(previews, args.output / "contact_sheet.jpg")
    print(
        f"Rendered {len(previews)} full-field annotation overlays to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
