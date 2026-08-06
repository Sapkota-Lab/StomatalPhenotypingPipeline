#!/usr/bin/env python3
"""Run the published StomaDet Faster R-CNN on microscope images.

StomaDet was trained on bright-field maize leaf imprints. It uses Detectron2,
so this evaluator remains separate from the production ensemble until visual
validation demonstrates that it transfers to the current plant material.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from stomata_engine import (
    Detection,
    draw_overlay,
    global_nms,
    image_paths,
    tile_regions,
)


DEFAULT_MODEL = Path("models/external/stomadet/models/stoma_detector.pth")
DEFAULT_CONFIG = Path("models/external/stomadet/models/stoma_detector.yaml")
PUBLISHED_ARCHIVE_SHA256 = (
    "88ef26439ac190f30679883863430da232596e4f49242a8a789e49416c6c4a69"
)


def build_stomadet(
    config_path: Path,
    checkpoint_path: Path,
    confidence: float,
    device: str,
) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = str(checkpoint_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    return DefaultPredictor(cfg)


def infer_stomadet(
    predictor: DefaultPredictor,
    image,
) -> list[Detection]:
    instances = predictor(image[:, :, :3])["instances"].to("cpu")
    detections: list[Detection] = []
    for box, score, class_id in zip(
        instances.pred_boxes.tensor,
        instances.scores,
        instances.pred_classes,
    ):
        if int(class_id) != 0:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            Detection(x1, y1, x2, y2, float(score))
        )
    return detections


def infer_stomadet_tiled(
    predictor: DefaultPredictor,
    image,
    tile_size: int,
    overlap: float,
    nms_iou: float,
) -> list[Detection]:
    """Run StomaDet on overlapping tiles while preserving native object scale."""
    if tile_size <= 0:
        return infer_stomadet(predictor, image)
    height, width = image.shape[:2]
    detections: list[Detection] = []
    for x, y, left, top, right, bottom in tile_regions(
        width,
        height,
        tile_size,
        overlap,
    ):
        tile = image[
            y:min(y + tile_size, height),
            x:min(x + tile_size, width),
        ]
        for detection in infer_stomadet(predictor, tile):
            global_detection = Detection(
                detection.x1 + x,
                detection.y1 + y,
                detection.x2 + x,
                detection.y2 + y,
                detection.confidence,
            )
            center_x = (global_detection.x1 + global_detection.x2) / 2
            center_y = (global_detection.y1 + global_detection.y2) / 2
            if not (
                left <= center_x < right
                and top <= center_y < bottom
            ):
                continue
            detections.append(global_detection)
    return global_nms(
        detections,
        nms_iou,
        center_distance_threshold=0.30,
    )


def load_scope_manifest(path: Path | None) -> dict[str, str]:
    """Load the explicit analysis scope used to protect proper runs."""
    if path is None:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "analysis_scope"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest {path} must contain filename and analysis_scope"
            )
        scopes = {}
        for row in reader:
            filename = row["filename"].strip()
            scope = row["analysis_scope"].strip()
            if scope not in {"eligible", "out_of_scope"}:
                raise ValueError(f"Invalid analysis_scope {scope!r}")
            if filename in scopes:
                raise ValueError(f"Duplicate manifest row for {filename}")
            scopes[filename] = scope
    return scopes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_stomadet"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional scope manifest; only rows marked eligible are processed.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Native-resolution tile size; 0 preserves full-field inference.",
    )
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip JPEG overlays when only quantitative evaluation is needed.",
    )
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    if not args.config.exists():
        parser.error(f"Config not found: {args.config}")
    paths = image_paths(args.input)
    scopes = load_scope_manifest(args.manifest)
    if scopes:
        paths = [path for path in paths if scopes.get(path.name) == "eligible"]
    if not paths:
        parser.error(f"No supported images found in {args.input}")

    predictor = build_stomadet(
        args.config,
        args.model,
        args.confidence,
        args.device,
    )
    count_rows: list[dict] = []
    detection_rows: list[dict] = []
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        detections = infer_stomadet_tiled(
            predictor,
            image,
            args.tile_size,
            args.overlap,
            args.nms_iou,
        )
        count_rows.append({
            "filename": path.name,
            "stomata_count": len(detections),
        })
        for detection_id, detection in enumerate(detections, 1):
            detection_rows.append({
                "filename": path.name,
                "detection_id": detection_id,
                "confidence": round(detection.confidence, 6),
                "x1": round(detection.x1, 2),
                "y1": round(detection.y1, 2),
                "x2": round(detection.x2, 2),
                "y2": round(detection.y2, 2),
            })
        if not args.no_overlays:
            draw_overlay(
                image,
                detections,
                args.output / "overlays" / f"{path.stem}_overlay.jpg",
            )
        print(f"{path.name}: {len(detections)}")

    with (args.output / "counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "stomata_count"],
        )
        writer.writeheader()
        writer.writerows(count_rows)
    with (args.output / "detections.csv").open("w", newline="") as handle:
        fields = [
            "filename", "detection_id", "confidence",
            "x1", "y1", "x2", "y2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detection_rows)


if __name__ == "__main__":
    main()
