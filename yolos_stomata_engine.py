#!/usr/bin/env python3
"""Alternative tiled stomata detector using a fine-tuned YOLOS transformer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from stomata_engine import Detection, draw_overlay, global_nms, image_paths, tile_starts


DEFAULT_MODEL = "DunnBC22/yolos-small-Stomata_Cells"


def open_oriented_image(path: Path) -> Image.Image:
    """Load a JPEG with its EXIF orientation applied.

    OpenCV applies EXIF orientation by default, while Pillow does not. Applying
    it explicitly keeps transformer detections, OpenCV detections, and saved
    overlays in the same coordinate system.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def infer_yolos(
    processor,
    model,
    image: Image.Image,
    confidence: float,
    overlap: float,
    tile_width: int,
    tile_height: int,
    device: str,
) -> list[Detection]:
    width, height = image.size
    tiles: list[Image.Image] = []
    origins: list[tuple[int, int]] = []
    for y in tile_starts(height, tile_height, overlap):
        for x in tile_starts(width, tile_width, overlap):
            tiles.append(image.crop((x, y, min(x + tile_width, width), min(y + tile_height, height))))
            origins.append((x, y))

    detections: list[Detection] = []
    for start in range(0, len(tiles), 4):
        batch = tiles[start:start + 4]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        sizes = torch.tensor([tile.size[::-1] for tile in batch], device=device)
        results = processor.post_process_object_detection(outputs, threshold=confidence, target_sizes=sizes)
        for result, (origin_x, origin_y) in zip(results, origins[start:start + 4]):
            for score, box in zip(result["scores"], result["boxes"]):
                x1, y1, x2, y2 = box.detach().cpu().tolist()
                detections.append(Detection(x1 + origin_x, y1 + origin_y,
                                            x2 + origin_x, y2 + origin_y, float(score)))
    # The same stoma can receive boxes of substantially different heights on
    # overlapping tiles, leaving too little IoU for conventional NMS.
    return global_nms(detections, 0.30, center_distance_threshold=0.55)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--output", type=Path, default=Path("results_yolos"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--tile-width", type=int, default=864)
    parser.add_argument("--tile-height", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = image_paths(args.input)
    if not paths:
        parser.error(f"No supported images found in {args.input}")
    processor = AutoImageProcessor.from_pretrained(args.model, use_fast=False)
    model = AutoModelForObjectDetection.from_pretrained(args.model).eval().to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    counts, rows = [], []
    for path in paths:
        image = open_oriented_image(path)
        detections = infer_yolos(processor, model, image, args.confidence, args.overlap,
                                 args.tile_width, args.tile_height, args.device)
        counts.append({"filename": path.name, "stomata_count": len(detections)})
        for index, detection in enumerate(detections, 1):
            rows.append({"filename": path.name, "detection_id": index,
                         "confidence": round(detection.confidence, 6),
                         "x1": round(detection.x1, 2), "y1": round(detection.y1, 2),
                         "x2": round(detection.x2, 2), "y2": round(detection.y2, 2)})
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        draw_overlay(bgr, detections, args.output / "overlays" / f"{path.stem}_overlay.jpg")
        print(f"{path.name}: {len(detections)}")
    with (args.output / "counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "stomata_count"])
        writer.writeheader(); writer.writerows(counts)
    fields = ["filename", "detection_id", "confidence", "x1", "y1", "x2", "y2"]
    with (args.output / "detections.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
