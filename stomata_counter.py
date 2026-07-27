#!/usr/bin/env python3
"""Exploratory baseline for counting stomata-like objects in microscope images.

This is a candidate generator, not a validated biological measurement method.
It uses local contrast enhancement, dark-feature extraction, and contour filters.
Review the exported overlays before trusting any count.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def count_image(path: Path, overlay_path: Path, max_width: int = 1800) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")

    scale = min(1.0, max_width / image.shape[1])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(gray)

    # Stomatal pores/complexes often contain a locally dark feature. Blackhat
    # emphasizes dark structures while suppressing slow background variation.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    dark = cv2.GaussianBlur(dark, (5, 5), 0)
    threshold = max(12, int(np.percentile(dark, 93)))
    mask = cv2.threshold(dark, threshold, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30 or area > 12000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = max(w, h) / max(1, min(w, h))
        if ratio > 5.0:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1e-6)
        if circularity < 0.12:
            continue
        accepted.append((contour, (x, y, w, h)))

    overlay = image.copy()
    for index, (contour, box) in enumerate(accepted, start=1):
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
        x, y, w, h = box
        cv2.putText(overlay, str(index), (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    return len(accepted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--output", type=Path, default=Path("pilot_output"))
    args = parser.parse_args()

    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in images:
        count = count_image(image, args.output / f"{image.stem}_overlay.jpg")
        rows.append({"filename": image.name, "candidate_count": count})
    with (args.output / "candidate_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["filename", "candidate_count"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
