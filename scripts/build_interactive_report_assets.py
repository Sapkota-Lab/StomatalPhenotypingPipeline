#!/usr/bin/env python3
"""Prepare data and image assets for the integrated interactive report."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "interactive_report"
PUBLIC = SITE / "public"
IMAGES = PUBLIC / "images"
THERMAL = Path("/Users/lenquanhone/SapkotaLab/SeedThermal")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def copy_asset(source: Path, name: str) -> None:
    shutil.copy2(source, IMAGES / name)


def render_mask_crop(row: dict[str, str], index: int) -> str:
    annotation = json.loads(Path(row["mask_annotation_path"]).read_text())
    polygon = annotation["polygon"]
    image_path = ROOT / "Stomata_Images" / row["stomata_image"]
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    pad = max(45, span * 0.55)
    crop_box = (
        max(0, int(min(xs) - pad)),
        max(0, int(min(ys) - pad)),
        min(image.width, int(max(xs) + pad)),
        min(image.height, int(max(ys) + pad)),
    )
    crop = image.crop(crop_box)
    shifted = [
        (point[0] - crop_box[0], point[1] - crop_box[1])
        for point in polygon
    ]
    draw = ImageDraw.Draw(crop, "RGBA")
    draw.polygon(shifted, fill=(212, 239, 192, 58))
    draw.line(shifted + [shifted[0]], fill=(218, 255, 178, 245), width=6)
    scale = 760 / max(crop.width, crop.height)
    crop = crop.resize(
        (
            max(1, round(crop.width * scale)),
            max(1, round(crop.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (900, 900), "#101713")
    canvas.paste(
        crop,
        ((900 - crop.width) // 2, (900 - crop.height) // 2),
    )
    filename = f"stomata-mask-{index}.jpg"
    canvas.save(IMAGES / filename, quality=94, optimize=True)
    return filename


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "downloads").mkdir(parents=True, exist_ok=True)
    integrated_root = ROOT / "results/integrated_exact_v1"
    shutil.copy2(
        integrated_root / "combined_product_dataset.json",
        PUBLIC / "downloads/combined-product-dataset.json",
    )
    shutil.copy2(
        integrated_root / "combined_plant_image_dataset.csv",
        PUBLIC / "downloads/combined-plant-image-dataset.csv",
    )
    shutil.copy2(
        integrated_root / "combined_detection_dataset.csv",
        PUBLIC / "downloads/combined-detection-dataset.csv",
    )
    shutil.copy2(
        integrated_root / "combined_measurement_dataset.csv",
        PUBLIC / "downloads/combined-measurement-dataset.csv",
    )
    shutil.copy2(
        integrated_root / "exact_join_audit.csv",
        PUBLIC / "downloads/exact-join-audit.csv",
    )

    thermal_assets = {
        "thermal-control.png": (
            "dashboard/src/assets/images/"
            "S20260715_P15_CherokeePurple_Control_thermal.png"
        ),
        "thermal-watered.png": (
            "dashboard/src/assets/images/"
            "S20260715_P16_CherokeePurple_Regular_thermal.png"
        ),
        "thermal-control-msx.jpg": (
            "dashboard/src/assets/images/"
            "S20260715_P15_CherokeePurple_Control_msx.jpg"
        ),
        "thermal-watered-msx.jpg": (
            "dashboard/src/assets/images/"
            "S20260715_P16_CherokeePurple_Regular_msx.jpg"
        ),
    }
    for name, relative in thermal_assets.items():
        copy_asset(THERMAL / relative, name)

    comparison_root = (
        ROOT / "results/quality_mask_analysis_v1/area_comparisons"
    )
    copy_asset(
        comparison_root / "paired_treatment_area.png",
        "paired-stomatal-area.png",
    )
    copy_asset(
        comparison_root / "variety_treatment_area.png",
        "variety-stomatal-area.png",
    )

    measurements = read_csv(
        ROOT
        / "results/quality_mask_analysis_v1/"
        "stomatal_complex_measurements.csv"
    )
    representative = [
        measurements[0],
        measurements[len(measurements) // 3],
        measurements[2 * len(measurements) // 3],
    ]
    mask_images = [
        render_mask_crop(row, index + 1)
        for index, row in enumerate(representative)
    ]

    plants = read_csv(
        ROOT
        / "results/combined_quality_mask_analysis_v1/"
        "plant_phenotypes.csv"
    )
    varieties = read_csv(
        ROOT
        / "results/combined_quality_mask_analysis_v1/"
        "variety_phenotypes.csv"
    )
    detections = read_csv(
        ROOT
        / "results/integrated_exact_v1/"
        "combined_detection_dataset.csv"
    )

    plant_records = []
    for row in plants:
        plant_records.append({
            "plantId": row["plant_unit_id"],
            "joinMethod": row["join_method"],
            "joinKeyVariety": row["join_key_variety"],
            "joinKeyTreatment": row["join_key_treatment"],
            "stomataSourceVariety": row["stomata_source_variety"],
            "stomataSourceTreatment": row["stomata_source_treatment"],
            "thermalSourceVariety": row["thermal_source_variety"],
            "thermalSourceTreatment": row["thermal_source_treatment"],
            "genotype": row["genotype_id"],
            "variety": row["variety"],
            "treatment": row["water_status"],
            "stomataImage": row["stomata_image"],
            "stomataPhotoSerial": row["stomata_photo_serial"],
            "thermalSession": row["thermal_session_id"],
            "thermalPlantId": row["thermal_session_plant_id"],
            "thermalSourceFile": row["thermal_source_file"],
            "stomatalArea": number(row["stomatal_complex_area_mean"]),
            "stomatalMedian": number(row["stomatal_complex_area_median"]),
            "stomatalSd": number(row["stomatal_complex_area_sd"]),
            "stomataN": int(row["stomata_objects_measured"]),
            "candidateTotal": int(row["stomata_candidates_total"]),
            "candidateAccepted": int(row["stomata_candidates_accepted"]),
            "candidateReview": int(row["stomata_candidates_review"]),
            "candidateRejected": int(row["stomata_candidates_rejected"]),
            "expectedTrue": number(row["stomata_expected_true_candidates"]),
            "canopyTemp": number(row["canopy_mean_c"]),
            "canopyMedian": number(row["canopy_median_c"]),
            "canopyMin": number(row["canopy_min_c"]),
            "canopyMax": number(row["canopy_max_c"]),
            "canopySd": number(row["canopy_std_c"]),
            "canopyP10": number(row["canopy_p10_c"]),
            "canopyP90": number(row["canopy_p90_c"]),
            "canopyDeltaAir": number(row["canopy_delta_t_air_c"]),
            "projectedArea": number(row["projected_plant_area_px2"]),
            "projectedAreaFraction": number(
                row["projected_plant_area_fraction"]
            ),
            "thermalQc": row["thermal_mask_qc_passed"].lower() == "true",
            "thermalQcFlag": row["thermal_mask_qc_flag"],
        })

    variety_records = []
    for row in varieties:
        variety_records.append({
            "genotype": row["genotype_id"],
            "variety": row["variety"],
            "stomatal": {
                "watered": number(row["watered_mean_area"]),
                "drought": number(row["drought_mean_area"]),
                "delta": number(row["difference_drought_minus_watered"]),
                "wateredN": int(row["watered_n_stomata"]),
                "droughtN": int(row["drought_n_stomata"]),
            },
            "thermal": {
                "watered": number(row["watered_canopy_mean_c"]),
                "drought": number(row["drought_canopy_mean_c"]),
                "delta": number(
                    row["difference_drought_minus_watered_canopy_mean_c"]
                ),
                "wateredQc": row[
                    "watered_thermal_mask_qc_passed"
                ].lower() == "true",
                "droughtQc": row[
                    "drought_thermal_mask_qc_passed"
                ].lower() == "true",
            },
            "projectedArea": {
                "watered": number(row["watered_projected_plant_area_px2"]),
                "drought": number(row["drought_projected_plant_area_px2"]),
                "delta": number(
                    row[
                        "difference_drought_minus_watered_"
                        "projected_plant_area_px2"
                    ]
                ),
            },
        })

    object_records = [{
        "objectId": row["object_id"],
        "stomataImage": row["stomata_image"],
        "variety": row["variety"],
        "treatment": row["water_status"],
        "area": number(row["complex_area_px2"]),
        "length": number(row["complex_length_px"]),
        "width": number(row["complex_width_px"]),
        "aspectRatio": number(row["complex_aspect_ratio"]),
        "solidity": number(row["complex_solidity"]),
        "circularity": number(row["complex_circularity"]),
        "focus": number(row["local_focus_score"]),
    } for row in measurements]

    detections_by_image_status = defaultdict(list)
    for row in detections:
        detections_by_image_status[
            (row["stomata_image"], row["decision_status"])
        ].append(row)
    detection_records = []
    for rows in detections_by_image_status.values():
        selected = sorted(
            rows,
            key=lambda row: number(row["calibrated_p_stoma"]) or 0,
            reverse=True,
        )[:12]
        detection_records.extend({
            "stomataImage": row["stomata_image"],
            "detectionId": row["detection_id"],
            "confidence": number(row["detector_confidence"]),
            "pStoma": number(row["calibrated_p_stoma"]),
            "decision": row["decision_status"],
            "x1": number(row["bbox_x1_px"]),
            "y1": number(row["bbox_y1_px"]),
            "x2": number(row["bbox_x2_px"]),
            "y2": number(row["bbox_y2_px"]),
        } for row in selected)

    report = {
        "generated": "2026-07-29",
        "assets": {"maskImages": mask_images},
        "summary": {
            "plants": 32,
            "varieties": 16,
            "manualMasks": 85,
            "thermalQcPlants": 30,
            "thermalQcPairs": 14,
        },
        "findings": {
            "stomatal": {
                "wateredMean": 10968.522349239933,
                "droughtMean": 10211.455252737855,
                "difference": -757.0670965020786,
                "relativePercent": -6.9,
                "ciLow": -2766.3328304773877,
                "ciHigh": 1252.1986374732303,
                "p": 0.43445621134608337,
                "wilcoxonP": 0.40374755859375,
            },
            "thermal": {
                "wateredMean": 26.052,
                "droughtMean": 25.811,
                "difference": -0.241,
                "relativePercent": -0.92,
                "ciLow": -0.528,
                "ciHigh": 0.046,
                "p": 0.0929,
                "wilcoxonP": 0.2166,
            },
            "projectedArea": {
                "wateredMean": 6034.8,
                "droughtMean": 8408.6,
                "difference": 2373.8,
                "relativePercent": 39.3,
                "ciLow": 968.4,
                "ciHigh": 3779.2,
                "p": 0.00294,
                "wilcoxonP": 0.00525,
            },
            "crossModal": {
                "temperaturePearson": 0.022,
                "temperatureP": 0.908,
                "projectedAreaPearson": 0.038,
                "projectedAreaP": 0.842,
            },
        },
        "plants": plant_records,
        "varieties": variety_records,
        "objects": object_records,
        "detections": detection_records,
    }
    destination = SITE / "app/study-data.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
