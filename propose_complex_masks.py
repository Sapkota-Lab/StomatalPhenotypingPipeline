#!/usr/bin/env python3
"""Propose stomatal-complex masks inside expert boxes with YOLO segmentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import SAM, YOLO

from complex_mask_annotator import balanced_queue, polygon_area


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def candidate_rank(
    predicted_box: np.ndarray,
    expert_box: np.ndarray,
    confidence: float,
) -> tuple[float, float, float]:
    iou = box_iou(predicted_box, expert_box)
    predicted_center = np.asarray([
        (predicted_box[0] + predicted_box[2]) / 2,
        (predicted_box[1] + predicted_box[3]) / 2,
    ])
    expert_center = np.asarray([
        (expert_box[0] + expert_box[2]) / 2,
        (expert_box[1] + expert_box[3]) / 2,
    ])
    diagonal = max(
        math.hypot(
            float(expert_box[2] - expert_box[0]),
            float(expert_box[3] - expert_box[1]),
        ),
        1.0,
    )
    center_similarity = max(
        0.0,
        1.0 - float(np.linalg.norm(predicted_center - expert_center)) / diagonal,
    )
    rank = 0.50 * iou + 0.30 * float(confidence) + 0.20 * center_similarity
    return rank, iou, center_similarity


def select_complex_mask(
    result,
    expert_box: np.ndarray,
) -> dict | None:
    if result.masks is None or result.boxes is None:
        return None
    polygons = result.masks.xy
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    candidates = []
    for index, (box, class_id, confidence) in enumerate(
        zip(boxes, classes, confidences)
    ):
        if int(class_id) != 0 or index >= len(polygons):
            continue
        polygon = np.asarray(polygons[index], dtype=float)
        if polygon.ndim != 2 or len(polygon) < 3:
            continue
        rank, iou, center_similarity = candidate_rank(
            box,
            expert_box,
            float(confidence),
        )
        candidates.append({
            "polygon": polygon,
            "model_confidence": float(confidence),
            "box_iou_with_expert": iou,
            "center_similarity": center_similarity,
            "selection_rank": rank,
        })
    return max(candidates, key=lambda row: row["selection_rank"]) if candidates else None


def select_prompted_mask(result, expert_box: np.ndarray) -> dict | None:
    if result.masks is None or len(result.masks.data) == 0:
        return None
    mask = result.masks.data[0].detach().cpu().numpy() > 0.5
    height, width = mask.shape
    box_width = expert_box[2] - expert_box[0]
    box_height = expert_box[3] - expert_box[1]
    left = max(0, math.floor(expert_box[0] - 0.1 * box_width))
    top = max(0, math.floor(expert_box[1] - 0.1 * box_height))
    right = min(width, math.ceil(expert_box[2] + 0.1 * box_width))
    bottom = min(height, math.ceil(expert_box[3] + 0.1 * box_height))
    clipped = np.zeros_like(mask)
    clipped[top:bottom, left:right] = mask[top:bottom, left:right]
    y_coordinates, x_coordinates = np.nonzero(clipped)
    if len(x_coordinates) < 3:
        return None
    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float32)
    polygon = cv2.convexHull(points).reshape(-1, 2).astype(float)
    if len(polygon) < 3:
        return None
    predicted_box = np.asarray([
        polygon[:, 0].min(),
        polygon[:, 1].min(),
        polygon[:, 0].max(),
        polygon[:, 1].max(),
    ])
    confidence = (
        float(result.boxes.conf[0].detach().cpu())
        if result.boxes is not None and len(result.boxes) else 0.0
    )
    rank, iou, center_similarity = candidate_rank(
        predicted_box,
        expert_box,
        confidence,
    )
    return {
        "polygon": polygon,
        "model_confidence": confidence,
        "box_iou_with_expert": iou,
        "center_similarity": center_similarity,
        "selection_rank": rank,
    }


def crop_bounds(
    box: dict[str, float],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    box_width = box["x2"] - box["x1"]
    box_height = box["y2"] - box["y1"]
    return (
        max(0, math.floor(box["x1"] - padding * box_width)),
        max(0, math.floor(box["y1"] - padding * box_height)),
        min(width, math.ceil(box["x2"] + padding * box_width)),
        min(height, math.ceil(box["y2"] + padding * box_height)),
    )


def oriented_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


def find_image(root: Path, filename: str) -> Path:
    matches = [
        path for path in root.rglob(filename)
        if path.is_file() and path.name.lower() == filename.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one image named {filename}, found {len(matches)}")
    return matches[0]


def write_overlay(
    crop: np.ndarray,
    expert_box: np.ndarray,
    polygon: np.ndarray | None,
    output: Path,
) -> None:
    canvas = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        canvas,
        tuple(np.round(expert_box[:2]).astype(int)),
        tuple(np.round(expert_box[2:]).astype(int)),
        (0, 165, 255),
        2,
    )
    if polygon is not None:
        points = np.round(polygon).astype(np.int32)
        layer = canvas.copy()
        cv2.fillPoly(layer, [points], (40, 190, 40))
        canvas = cv2.addWeighted(layer, 0.25, canvas, 0.75, 0)
        cv2.polylines(canvas, [points], True, (40, 220, 40), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("Stomata_Images"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    parser.add_argument("--boxes", type=Path, default=Path("box_annotations"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/sam2.1_t.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_artifacts/complex_mask_proposals_v1"),
    )
    parser.add_argument("--max-per-image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--padding", type=float, default=0.75)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overlays", action="store_true")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    queue = balanced_queue(
        manifest,
        args.boxes,
        args.max_per_image,
        args.seed,
    )
    if args.limit is not None:
        queue = queue[:args.limit]
    by_image = defaultdict(list)
    for item in queue:
        by_image[item["image"]].append(item)
    is_prompted_sam = "sam" in args.model.stem.lower()
    model = SAM(str(args.model)) if is_prompted_sam else YOLO(args.model)
    records = []
    proposal_dir = args.output / "json"
    overlay_dir = args.output / "overlays"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    model_hash = file_sha256(args.model)
    for filename in sorted(by_image):
        image = oriented_rgb(find_image(args.images, filename))
        height, width = image.shape[:2]
        crops = []
        crop_metadata = []
        for item in by_image[filename]:
            left, top, right, bottom = crop_bounds(
                item["box"],
                width,
                height,
                args.padding,
            )
            crop = image[top:bottom, left:right]
            crops.append(crop)
            crop_metadata.append((item, left, top, right, bottom))
        if is_prompted_sam:
            results = []
            for crop, metadata in zip(crops, crop_metadata):
                item, left, top, _right, _bottom = metadata
                box = item["box"]
                prompt = [[
                    box["x1"] - left,
                    box["y1"] - top,
                    box["x2"] - left,
                    box["y2"] - top,
                ]]
                result = model.predict(
                    crop,
                    bboxes=prompt,
                    device=args.device,
                    verbose=False,
                )
                results.append(result[0])
        else:
            results = model.predict(
                crops,
                imgsz=640,
                conf=args.confidence,
                iou=0.5,
                retina_masks=True,
                device=args.device,
                verbose=False,
            )
        for crop, result, metadata in zip(crops, results, crop_metadata):
            item, left, top, right, bottom = metadata
            box = item["box"]
            expert_crop_box = np.asarray([
                box["x1"] - left,
                box["y1"] - top,
                box["x2"] - left,
                box["y2"] - top,
            ])
            selected = (
                select_prompted_mask(result, expert_crop_box)
                if is_prompted_sam
                else select_complex_mask(result, expert_crop_box)
            )
            polygon = None
            if selected is not None:
                polygon = selected.pop("polygon")
                full_polygon = [
                    [round(float(x + left), 2), round(float(y + top), 2)]
                    for x, y in polygon
                ]
                area = polygon_area(full_polygon)
                status = "proposed"
            else:
                full_polygon = []
                area = None
                status = "no_proposal"
                selected = {
                    "model_confidence": None,
                    "box_iou_with_expert": None,
                    "center_similarity": None,
                    "selection_rank": None,
                }
            record = {
                "schema_version": 2,
                **item,
                "status": status,
                "polygon": full_polygon,
                "complex_area_px2": area,
                **selected,
                "crop": {
                    "x1": left,
                    "y1": top,
                    "x2": right,
                    "y2": bottom,
                },
                "proposal_method": (
                    "sam2.1_t_expert_box_prompt"
                    if is_prompted_sam
                    else "stomaquant_yolov12seg_class_0"
                ),
                "geometry_method": (
                    "convex_hull_of_prompted_mask_clipped_to_110pct_expert_box"
                    if is_prompted_sam
                    else "native_segmentation_polygon"
                ),
                "model_path": str(args.model),
                "model_sha256": model_hash,
                "validated_against_manual_polygon": False,
            }
            destination = proposal_dir / f"{item['object_id'].replace(':', '_')}.json"
            destination.write_text(json.dumps(record, indent=2) + "\n")
            if args.overlays:
                write_overlay(
                    crop,
                    expert_crop_box,
                    polygon,
                    overlay_dir / f"{item['object_id'].replace(':', '_')}.jpg",
                )
            records.append(record)
        print(
            filename,
            sum(row["status"] == "proposed" for row in records),
            "cumulative proposals",
        )
    summary = {
        "objects": len(records),
        "proposals": sum(row["status"] == "proposed" for row in records),
        "no_proposal": sum(row["status"] == "no_proposal" for row in records),
        "model_sha256": model_hash,
        "validated": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
