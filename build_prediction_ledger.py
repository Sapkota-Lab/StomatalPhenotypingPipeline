#!/usr/bin/env python3
"""Join calibrated model candidates to plants and measurement metadata."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from build_stomata_ledger import (
    add_spatial_metrics,
    find_image,
    load_oriented_rgb,
    local_image_metrics,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_prediction_ledger(
    manifest_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    images_dir: Path,
    *,
    run_id: str,
    calibration_id: str,
) -> list[dict[str, object]]:
    manifest_by_image = {
        row["stomata_image"]: row for row in manifest_rows
    }
    predictions_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        filename = Path(row["filename"]).name
        if filename not in manifest_by_image:
            raise ValueError(f"No plant mapping for {filename}")
        predictions_by_image[filename].append(row)
    output = []
    for filename in sorted(predictions_by_image):
        manifest = manifest_by_image[filename]
        image = load_oriented_rgb(find_image(images_dir, filename))
        height, width = image.shape[:2]
        image_rows = []
        ordered = sorted(
            predictions_by_image[filename],
            key=lambda row: float(row["p_stoma"]),
            reverse=True,
        )
        for index, prediction in enumerate(ordered, 1):
            x1 = min(max(float(prediction["x1"]), 0.0), float(width))
            y1 = min(max(float(prediction["y1"]), 0.0), float(height))
            x2 = min(max(float(prediction["x2"]), 0.0), float(width))
            y2 = min(max(float(prediction["y2"]), 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            touches_edge = (
                x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
            )
            decision_status = prediction["decision_status"]
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
                "proposal_source": "tomato_yolov12_tiled_fused",
                "decision_status": decision_status,
                "decision_reasons": (
                    f"calibrated_{decision_status};full_image_tiled_fusion"
                ),
                "detector_score_tomato_yolov12": float(
                    prediction["raw_detector_score"]
                ),
                "p_stoma": float(prediction["p_stoma"]),
                "calibration_id": calibration_id,
                "center_x_px": center_x,
                "center_y_px": center_y,
                "bbox_x1_px": x1,
                "bbox_y1_px": y1,
                "bbox_x2_px": x2,
                "bbox_y2_px": y2,
                "bbox_width_px": box_width,
                "bbox_height_px": box_height,
                "bbox_long_axis_px": max(box_width, box_height),
                "bbox_short_axis_px": min(box_width, box_height),
                "bbox_area_px2": box_width * box_height,
                "bbox_aspect_ratio": (
                    max(box_width, box_height)
                    / min(box_width, box_height)
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
                "distance_to_image_edge_px": min(
                    center_x,
                    center_y,
                    width - center_x,
                    height - center_y,
                ),
                "neighbor_radius_px": "",
                "neighbors_within_radius": "",
                "local_density_per_px2": "",
                "nearest_neighbor_center_distance_px": "",
                "nearest_neighbor_bearing_deg": "",
                "nearest_neighbor_edge_distance_px": "",
                **local_image_metrics(image, (x1, y1, x2, y2)),
            })
        # Rejected proposals are retained for auditability, but they are not
        # biological objects and therefore must not influence spatial traits.
        analysis_rows = [
            row for row in image_rows
            if row["decision_status"] in {"accepted", "review"}
        ]
        add_spatial_metrics(analysis_rows)
        output.extend(image_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("Stomata_Images"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="tomato_yolov12_full_v1")
    parser.add_argument(
        "--calibration-id",
        default="tomato_platt_validation_v1",
    )
    args = parser.parse_args()
    rows = build_prediction_ledger(
        read_csv(args.manifest),
        read_csv(args.predictions),
        args.images,
        run_id=args.run_id,
        calibration_id=args.calibration_id,
    )
    if not rows:
        raise ValueError("No prediction ledger rows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} calibrated ledger records to {args.output}")


if __name__ == "__main__":
    main()
