#!/usr/bin/env python3
"""Render StomataHub Pascal-VOC expert boxes on source images."""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def load_objects(xml_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    root = ET.parse(xml_path).getroot()
    objects = []
    for node in root.findall("object"):
        name = (node.findtext("class") or node.findtext("name") or "").strip().lower()
        if name == "stoma":
            name = "stomata"
        if name not in {"stomata", "pore"}:
            continue
        box = node.find("bndbox")
        values = tuple(
            int(round(float(box.findtext(field, "0"))))
            for field in ("xmin", "ymin", "xmax", "ymax")
        )
        objects.append((name, values))
    return objects


def render_image(image_path: Path, output_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {image_path}")
    objects = load_objects(image_path.with_suffix(".xml"))
    colors = {"stomata": (0, 220, 0), "pore": (220, 0, 220)}
    for name, (x1, y1, x2, y2) in objects:
        color = colors[name]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image,
            name,
            (x1, max(24, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write {output_path}")
    return image


def make_contact_sheet(images: list[np.ndarray], output: Path, cell_size: int, columns: int) -> None:
    if not images:
        return
    rows = math.ceil(len(images) / columns)
    sheet = np.full((rows * cell_size, columns * cell_size, 3), 245, dtype=np.uint8)
    for index, image in enumerate(images):
        scale = min(cell_size / image.shape[1], cell_size / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x = column * cell_size + (cell_size - resized.shape[1]) // 2
        y = row * cell_size + (cell_size - resized.shape[0]) // 2
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--cell-size", type=int, default=700)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()
    images = sorted(args.source.glob("*.jpg"), key=natural_key)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        parser.error(f"No JPG files found in {args.source}")
    rendered = []
    for image_path in images:
        rendered.append(render_image(image_path, args.output / image_path.name))
    make_contact_sheet(
        rendered,
        args.output / "contact_sheet.jpg",
        args.cell_size,
        args.columns,
    )
    print(f"Rendered {len(images)} expert-labeled images to {args.output}")


if __name__ == "__main__":
    main()
