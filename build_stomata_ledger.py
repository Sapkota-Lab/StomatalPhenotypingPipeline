#!/usr/bin/env python3
"""Build a conservative per-stoma ledger from expert box annotations.

Bounding boxes provide localization and coarse geometry only. This importer
intentionally leaves precise stomatal-complex and pore measurements empty
until validated instance masks are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


ACCEPTED_ANNOTATION_STATUSES = {"complete", "completed", "approved"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def find_image(images_dir: Path, filename: str) -> Path:
    direct = images_dir / filename
    if direct.exists():
        return direct
    matches = [
        path for path in images_dir.rglob("*")
        if path.is_file() and path.name.lower() == filename.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source image for {filename}, found {len(matches)}"
        )
    return matches[0]


def load_oriented_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(
            ImageOps.exif_transpose(source).convert("RGB")
        ).copy()


def clipped_box(
    box: dict[str, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1 = min(max(float(box["x1"]), 0.0), float(width))
    y1 = min(max(float(box["y1"]), 0.0), float(height))
    x2 = min(max(float(box["x2"]), 0.0), float(width))
    y2 = min(max(float(box["y2"]), 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate annotation box: {box}")
    return x1, y1, x2, y2


def local_image_metrics(
    rgb: np.ndarray,
    box: tuple[float, float, float, float],
    context_scale: float = 1.25,
) -> dict[str, float]:
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = max(4.0, (x2 - x1) * context_scale)
    crop_height = max(4.0, (y2 - y1) * context_scale)
    left = max(0, int(math.floor(center_x - crop_width / 2)))
    top = max(0, int(math.floor(center_y - crop_height / 2)))
    right = min(width, int(math.ceil(center_x + crop_width / 2)))
    bottom = min(height, int(math.ceil(center_y + crop_height / 2)))
    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"Empty local crop for box {box}")
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    return {
        "local_focus_score": float(
            cv2.Laplacian(gray, cv2.CV_32F).var()
        ),
        "local_contrast_score": float(gray.std()),
        "local_brightness_mean": float(gray.mean()),
        "local_brightness_sd": float(gray.std()),
        "local_edge_density": float(np.count_nonzero(edges) / edges.size),
        "local_saturation_fraction": float(
            np.count_nonzero((gray <= 2) | (gray >= 253)) / gray.size
        ),
    }


def axis_aligned_edge_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def add_spatial_metrics(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    diagonals = [
        math.hypot(float(row["bbox_width_px"]), float(row["bbox_height_px"]))
        for row in rows
    ]
    radius = 2.0 * float(np.median(diagonals))
    for index, row in enumerate(rows):
        candidates = []
        center_x = float(row["center_x_px"])
        center_y = float(row["center_y_px"])
        first_box = tuple(
            float(row[field])
            for field in (
                "bbox_x1_px",
                "bbox_y1_px",
                "bbox_x2_px",
                "bbox_y2_px",
            )
        )
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            dx = float(other["center_x_px"]) - center_x
            dy = float(other["center_y_px"]) - center_y
            center_distance = math.hypot(dx, dy)
            other_box = tuple(
                float(other[field])
                for field in (
                    "bbox_x1_px",
                    "bbox_y1_px",
                    "bbox_x2_px",
                    "bbox_y2_px",
                )
            )
            candidates.append(
                (
                    center_distance,
                    math.degrees(math.atan2(dy, dx)) % 360.0,
                    axis_aligned_edge_distance(first_box, other_box),
                )
            )
        row["neighbor_radius_px"] = radius
        row["neighbors_within_radius"] = sum(
            distance <= radius for distance, _bearing, _edge in candidates
        )
        row["local_density_per_px2"] = (
            float(row["neighbors_within_radius"]) / (math.pi * radius**2)
            if radius > 0 else ""
        )
        if candidates:
            nearest = min(candidates, key=lambda item: item[0])
            row["nearest_neighbor_center_distance_px"] = nearest[0]
            row["nearest_neighbor_bearing_deg"] = nearest[1]
            row["nearest_neighbor_edge_distance_px"] = nearest[2]
        else:
            row["nearest_neighbor_center_distance_px"] = ""
            row["nearest_neighbor_bearing_deg"] = ""
            row["nearest_neighbor_edge_distance_px"] = ""


def build_ledger(
    manifest_rows: list[dict[str, str]],
    annotation_paths: list[Path],
    images_dir: Path,
    *,
    run_id: str,
) -> list[dict[str, object]]:
    manifest_by_image = {row["stomata_image"]: row for row in manifest_rows}
    output = []
    seen_images = set()
    for annotation_path in sorted(annotation_paths):
        annotation = json.loads(annotation_path.read_text())
        filename = Path(annotation["image"]).name
        if filename in seen_images:
            raise ValueError(f"Duplicate annotation image: {filename}")
        seen_images.add(filename)
        manifest = manifest_by_image.get(filename)
        if manifest is None:
            continue
        source_path = find_image(images_dir, filename)
        rgb = load_oriented_rgb(source_path)
        height, width = rgb.shape[:2]
        if (width, height) != (
            int(annotation["width"]),
            int(annotation["height"]),
        ):
            raise ValueError(
                f"Annotation dimensions do not match {filename}: "
                f"{annotation['width']}x{annotation['height']} versus "
                f"{width}x{height}"
            )
        status = str(annotation.get("status", "unknown")).strip().lower()
        accepted = status in ACCEPTED_ANNOTATION_STATUSES
        image_rows = []
        for index, raw_box in enumerate(annotation.get("boxes", []), 1):
            box = clipped_box(raw_box, width, height)
            x1, y1, x2, y2 = box
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance_to_edge = min(
                center_x,
                center_y,
                width - center_x,
                height - center_y,
            )
            touches_edge = (
                x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
            )
            reasons = (
                "expert_box;annotation_complete"
                if accepted else
                f"expert_box;annotation_{status};requires_review"
            )
            image_rows.append({
                "stoma_id": f"{run_id}:{Path(filename).stem}:{index:04d}",
                "run_id": run_id,
                "study_id": manifest["study_id"],
                "genotype_id": manifest["genotype_id"],
                "plant_unit_id": manifest["plant_unit_id"],
                "variety": manifest["variety"],
                "water_status": manifest["water_status"],
                "water_ml": manifest["water_ml"],
                "image_id": Path(filename).stem,
                "stomata_image": filename,
                "detection_id": index,
                "proposal_source": "expert_box_annotation",
                "annotation_status": status,
                "decision_status": "accepted" if accepted else "review",
                "decision_reasons": reasons,
                "p_stoma": "",
                "calibration_id": "",
                "center_x_px": center_x,
                "center_y_px": center_y,
                "bbox_x1_px": x1,
                "bbox_y1_px": y1,
                "bbox_x2_px": x2,
                "bbox_y2_px": y2,
                "bbox_width_px": bbox_width,
                "bbox_height_px": bbox_height,
                "bbox_long_axis_px": max(bbox_width, bbox_height),
                "bbox_short_axis_px": min(bbox_width, bbox_height),
                "bbox_area_px2": bbox_width * bbox_height,
                "bbox_aspect_ratio": (
                    max(bbox_width, bbox_height)
                    / min(bbox_width, bbox_height)
                ),
                "coarse_geometry_only": True,
                "complex_length_px": "",
                "complex_width_px": "",
                "complex_area_px2": "",
                "complex_perimeter_px": "",
                "complex_segmentation_quality": "",
                "pore_area_px2": "",
                "pore_length_px": "",
                "pore_width_px": "",
                "pore_segmentation_quality": "",
                "measurement_quality_passed": False,
                "measurement_exclusion_reason": (
                    "validated_stomatal_complex_mask_unavailable"
                ),
                "touches_image_edge": touches_edge,
                "edge_censored": touches_edge,
                "distance_to_image_edge_px": distance_to_edge,
                **local_image_metrics(rgb, box),
            })
        add_spatial_metrics(image_rows)
        output.extend(image_rows)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No mapped stomata annotations were found")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


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
        default=Path("results/combined/stomata.csv"),
    )
    parser.add_argument("--run-id", default="expert_box_import_v1")
    args = parser.parse_args()

    rows = build_ledger(
        read_csv(args.manifest),
        list(args.annotations.glob("*.json")),
        args.images,
        run_id=args.run_id,
    )
    write_csv(args.output, rows)
    accepted = sum(row["decision_status"] == "accepted" for row in rows)
    print(
        f"Wrote {len(rows)} per-stoma records to {args.output}: "
        f"{accepted} accepted, {len(rows) - accepted} pending review; "
        "precise complex and pore areas remain unavailable."
    )


if __name__ == "__main__":
    main()
