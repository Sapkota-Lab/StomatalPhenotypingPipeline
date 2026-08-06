#!/usr/bin/env python3
"""Build a balanced measurement table from quality-selected manual masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from apply_validated_complex_masks import polygon_traits
from complex_mask_annotator import quality_queue


def additional_polygon_traits(
    polygon: list[list[float]],
) -> dict[str, float]:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    moments = cv2.moments(contour)
    return {
        "complex_convex_area_px2": hull_area,
        "complex_equivalent_diameter_px": math.sqrt(4 * area / math.pi),
        "complex_circularity": (
            4 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        ),
        "complex_centroid_x_px": (
            float(moments["m10"] / moments["m00"])
            if moments["m00"] else float(contour[:, 0, 0].mean())
        ),
        "complex_centroid_y_px": (
            float(moments["m01"] / moments["m00"])
            if moments["m00"] else float(contour[:, 0, 1].mean())
        ),
    }


def select_measurements(
    manifest_rows: list[dict[str, str]],
    candidates: list[dict],
    annotations: dict[str, tuple[dict, Path]],
    *,
    max_per_image: int,
    min_per_image: int,
) -> tuple[list[dict], dict]:
    manifest_by_image = {
        row["stomata_image"]: row for row in manifest_rows
    }
    candidates_by_image: dict[str, list[dict]] = {}
    for item in candidates:
        candidates_by_image.setdefault(item["image"], []).append(item)

    rows = []
    coverage = {}
    for image in sorted(manifest_by_image):
        completed = []
        for item in candidates_by_image.get(image, []):
            pair = annotations.get(item["object_id"])
            if pair is None or pair[0].get("status") != "complete":
                continue
            polygon = pair[0].get("polygon", [])
            if len(polygon) < 3:
                continue
            completed.append((item, pair[0], pair[1]))
        selected = completed[:max_per_image]
        if len(selected) < min_per_image:
            raise ValueError(
                f"{image} has {len(selected)} usable masks; "
                f"{min_per_image} required"
            )
        coverage[image] = {
            "usable_completed_masks": len(completed),
            "selected_masks": len(selected),
        }
        for selected_rank, (item, annotation, path) in enumerate(selected, 1):
            polygon = annotation["polygon"]
            traits = {
                **polygon_traits(polygon),
                **additional_polygon_traits(polygon),
            }
            rows.append({
                "object_id": item["object_id"],
                "stomata_image": image,
                "filename": image,
                "study_id": manifest_by_image[image]["study_id"],
                "plant_unit_id": item["plant_unit_id"],
                "genotype_id": item["genotype_id"],
                "variety": item["variety"],
                "water_status": item["water_status"],
                "water_ml": manifest_by_image[image]["water_ml"],
                "expert_box_index": item["expert_box_index"],
                "quality_rank_among_all_expert_boxes": (
                    item["quality_rank_within_image"]
                ),
                "selected_rank_among_completed_masks": selected_rank,
                "local_quality_score": item["local_quality_score"],
                "local_focus_score": item["local_focus_score"],
                "local_contrast_5_95": item["local_contrast_5_95"],
                "local_clipped_fraction": item["local_clipped_fraction"],
                "edge_margin_px": item["edge_margin_px"],
                **traits,
                "area_unit": "px2",
                "length_unit": "px",
                "physical_scale_available": False,
                "decision_status": "accepted",
                "measurement_quality_passed": True,
                "measurement_source": "manual_outer_stomatal_complex_polygon",
                "sample_selection_method": (
                    "top_local_image_quality_among_completed_masks_v1"
                ),
                "mask_annotation_path": str(path),
                "reviewer": annotation.get("reviewer", ""),
                "reviewed_at_utc": annotation.get("reviewed_at_utc", ""),
                "polygon_json": json.dumps(polygon, separators=(",", ":")),
            })

    treatments = Counter(row["water_status"] for row in rows)
    report = {
        "schema_version": 1,
        "objects_selected": len(rows),
        "images": len(coverage),
        "plants": len({row["plant_unit_id"] for row in rows}),
        "varieties": len({row["genotype_id"] for row in rows}),
        "objects_by_treatment": dict(treatments),
        "max_per_image": max_per_image,
        "min_per_image": min_per_image,
        "coverage_by_image": coverage,
        "selection_method": (
            "top_local_image_quality_among_completed_masks_v1"
        ),
        "selection_variables": [
            "local_focus_score",
            "local_contrast_5_95",
            "local_clipped_fraction",
            "edge_margin_px",
        ],
        "selection_excludes": [
            "treatment",
            "variety",
            "object_area",
            "mask_area",
            "apparent_stomatal_shape",
        ],
        "measurement_scope": (
            "Manual outer stomatal-complex polygons; pixel units only."
        ),
        "interpretation": (
            "Quality-selected descriptive sample, not an unbiased estimate "
            "of the full within-image stomatal population."
        ),
    }
    return rows, report


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    parser.add_argument("--images", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--boxes", type=Path, default=Path("box_annotations"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("complex_mask_annotations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quality_mask_analysis_v1"),
    )
    parser.add_argument("--max-per-image", type=int, default=3)
    parser.add_argument("--min-per-image", type=int, default=2)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    candidates = quality_queue(
        manifest,
        args.boxes,
        args.images,
        max_per_image=10**9,
    )
    annotations = {}
    for path in args.annotations.glob("*.json"):
        annotation = json.loads(path.read_text())
        object_id = annotation["object_id"]
        if object_id in annotations:
            raise ValueError(f"Duplicate annotation {object_id}")
        annotations[object_id] = (annotation, path)
    rows, report = select_measurements(
        manifest,
        candidates,
        annotations,
        max_per_image=args.max_per_image,
        min_per_image=args.min_per_image,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "stomatal_complex_measurements.csv", rows)
    (args.output / "measurement_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Wrote {len(rows)} measurements across {report['images']} images "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
