#!/usr/bin/env python3
"""Validate proposed stomatal-complex masks against manual polygons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def polygon_mask(
    polygon: list[list[float]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = bounds
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    points = np.asarray(polygon, dtype=float)
    points[:, 0] -= left
    points[:, 1] -= top
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


def common_bounds(
    first: list[list[float]],
    second: list[list[float]],
) -> tuple[int, int, int, int]:
    points = np.asarray(first + second, dtype=float)
    return (
        math.floor(points[:, 0].min()) - 1,
        math.floor(points[:, 1].min()) - 1,
        math.ceil(points[:, 0].max()) + 2,
        math.ceil(points[:, 1].max()) + 2,
    )


def shape_metrics(polygon: list[list[float]]) -> dict[str, float]:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    (_center, dimensions, angle) = cv2.minAreaRect(contour)
    length = max(float(dimensions[0]), float(dimensions[1]))
    width = min(float(dimensions[0]), float(dimensions[1]))
    if dimensions[0] < dimensions[1]:
        angle += 90
    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "length_px": length,
        "width_px": width,
        "orientation_deg": angle % 180,
    }


def compare_polygons(
    manual: list[list[float]],
    proposed: list[list[float]],
) -> dict[str, float]:
    bounds = common_bounds(manual, proposed)
    manual_mask = polygon_mask(manual, bounds)
    proposed_mask = polygon_mask(proposed, bounds)
    intersection = int(np.logical_and(manual_mask, proposed_mask).sum())
    union = int(np.logical_or(manual_mask, proposed_mask).sum())
    manual_pixels = int(manual_mask.sum())
    proposed_pixels = int(proposed_mask.sum())
    manual_shape = shape_metrics(manual)
    proposed_shape = shape_metrics(proposed)
    relative_error = (
        (proposed_shape["area_px2"] - manual_shape["area_px2"])
        / manual_shape["area_px2"]
    )
    return {
        "mask_iou": intersection / union if union else 0.0,
        "mask_dice": (
            2 * intersection / (manual_pixels + proposed_pixels)
            if manual_pixels + proposed_pixels else 0.0
        ),
        "manual_area_px2": manual_shape["area_px2"],
        "proposed_area_px2": proposed_shape["area_px2"],
        "area_error_px2": (
            proposed_shape["area_px2"] - manual_shape["area_px2"]
        ),
        "relative_area_error": relative_error,
        "absolute_percentage_area_error": abs(relative_error) * 100,
        "manual_length_px": manual_shape["length_px"],
        "proposed_length_px": proposed_shape["length_px"],
        "manual_width_px": manual_shape["width_px"],
        "proposed_width_px": proposed_shape["width_px"],
        "manual_orientation_deg": manual_shape["orientation_deg"],
        "proposed_orientation_deg": proposed_shape["orientation_deg"],
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    values = lambda field: np.asarray(
        [float(row[field]) for row in rows],
        dtype=float,
    )
    return {
        "objects": len(rows),
        "median_iou": float(np.median(values("mask_iou"))),
        "median_dice": float(np.median(values("mask_dice"))),
        "mean_relative_area_bias_percent": float(
            values("relative_area_error").mean() * 100
        ),
        "median_absolute_percentage_area_error": float(
            np.median(values("absolute_percentage_area_error"))
        ),
        "p90_absolute_percentage_area_error": float(
            np.quantile(values("absolute_percentage_area_error"), 0.90)
        ),
    }


def gate(rows: list[dict], missing_proposals: int = 0) -> dict:
    plants = Counter(row["plant_unit_id"] for row in rows)
    treatments = Counter(row["water_status"] for row in rows)
    summary = summarize(rows)
    checks = {
        "no_missing_proposals": missing_proposals == 0,
        "all_validation_masks_blinded_to_proposals": bool(rows) and all(
            str(row.get("annotation_blinded_to_proposal", "")).lower()
            in {"true", "1"}
            for row in rows
        ),
        "at_least_128_masks": len(rows) >= 128,
        "all_32_plants_represented": len(plants) == 32,
        "at_least_3_masks_per_plant": bool(plants) and min(plants.values()) >= 3,
        "both_treatments_at_least_60": (
            treatments.get("watered", 0) >= 60
            and treatments.get("drought_stressed", 0) >= 60
        ),
        "median_iou_at_least_0_75": summary.get("median_iou", 0) >= 0.75,
        "median_dice_at_least_0_85": summary.get("median_dice", 0) >= 0.85,
        "median_absolute_area_error_at_most_10_percent": (
            summary.get("median_absolute_percentage_area_error", math.inf)
            <= 10
        ),
        "p90_absolute_area_error_at_most_25_percent": (
            summary.get("p90_absolute_percentage_area_error", math.inf) <= 25
        ),
        "absolute_mean_area_bias_at_most_5_percent": (
            abs(summary.get("mean_relative_area_bias_percent", math.inf)) <= 5
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual",
        type=Path,
        default=Path("complex_mask_annotations"),
    )
    parser.add_argument(
        "--proposals",
        type=Path,
        default=Path(
            "validation_artifacts/complex_mask_proposals_v1/json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "validation_artifacts/complex_mask_validation_v1"
        ),
    )
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()
    rows = []
    missing_proposals = []
    status_counts = Counter()
    proposal_model_hashes = set()
    proposal_methods = set()
    proposal_geometry_methods = set()
    proposal_schema_versions = set()
    for manual_path in sorted(args.manual.glob("*.json")):
        manual = json.loads(manual_path.read_text())
        status = manual.get("status", "unknown")
        status_counts[status] += 1
        if status != "complete":
            continue
        proposal_path = args.proposals / manual_path.name
        if not proposal_path.is_file():
            missing_proposals.append(manual["object_id"])
            continue
        proposal = json.loads(proposal_path.read_text())
        proposal_model_hashes.add(proposal.get("model_sha256"))
        proposal_methods.add(proposal.get("proposal_method"))
        proposal_geometry_methods.add(proposal.get("geometry_method"))
        proposal_schema_versions.add(proposal.get("schema_version"))
        if proposal.get("status") != "proposed" or not proposal.get("polygon"):
            missing_proposals.append(manual["object_id"])
            continue
        rows.append({
            "object_id": manual["object_id"],
            "image": manual["image"],
            "plant_unit_id": manual["plant_unit_id"],
            "genotype_id": manual["genotype_id"],
            "variety": manual["variety"],
            "water_status": manual["water_status"],
            "annotation_blinded_to_proposal": manual.get(
                "annotation_blinded_to_proposal",
                False,
            ),
            "model_confidence": proposal.get("model_confidence"),
            "box_iou_with_expert": proposal.get("box_iou_with_expert"),
            **compare_polygons(
                manual["polygon"],
                proposal["polygon"],
            ),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output / "per_object.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    by_treatment = {
        status: summarize([
            row for row in rows if row["water_status"] == status
        ])
        for status in ("watered", "drought_stressed")
    }
    by_variety_rows = defaultdict(list)
    for row in rows:
        by_variety_rows[row["variety"]].append(row)
    report = {
        "manual_status_counts": dict(sorted(status_counts.items())),
        "matched_complete_masks": len(rows),
        "missing_proposals": missing_proposals,
        "proposal_model_hashes": sorted(
            value for value in proposal_model_hashes if value
        ),
        "proposal_methods": sorted(
            value for value in proposal_methods if value
        ),
        "proposal_geometry_methods": sorted(
            value for value in proposal_geometry_methods if value
        ),
        "proposal_schema_versions": sorted(
            value for value in proposal_schema_versions if value is not None
        ),
        "overall": summarize(rows),
        "by_treatment": by_treatment,
        "by_variety": {
            variety: summarize(selected)
            for variety, selected in sorted(by_variety_rows.items())
        },
        "measurement_gate": gate(rows, len(missing_proposals)),
        "area_definition": (
            "Filled outer stomatal-complex contour, including the pore "
            "inside the outer boundary."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if args.require_gate and not report["measurement_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
