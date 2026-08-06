#!/usr/bin/env python3
"""Build strict plant, detection, and measurement products for the combined study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


PLANT_CONTEXT_FIELDS = (
    "study_id",
    "join_method",
    "join_key_variety",
    "join_key_treatment",
    "stomata_source_variety",
    "stomata_source_treatment",
    "thermal_source_variety",
    "thermal_source_treatment",
    "genotype_id",
    "plant_unit_id",
    "variety",
    "variety_class",
    "treatment_code",
    "water_status",
    "water_ml",
    "stomata_image",
    "stomata_photo_serial",
    "thermal_session_id",
    "thermal_session_plant_id",
    "thermal_source_file",
)

THERMAL_FIELDS = (
    "thermal_capture_time",
    "canopy_mean_c",
    "canopy_median_c",
    "canopy_min_c",
    "canopy_max_c",
    "canopy_std_c",
    "canopy_p10_c",
    "canopy_p90_c",
    "canopy_delta_t_air_c",
    "projected_plant_area_px2",
    "projected_plant_area_fraction",
    "thermal_mask_qc_passed",
    "thermal_mask_qc_flag",
)

STOMATA_AGGREGATE_FIELDS = (
    "stomata_detection_run_status",
    "stomata_candidates_total",
    "stomata_candidates_accepted",
    "stomata_candidates_review",
    "stomata_candidates_rejected",
    "stomata_expected_true_candidates",
    "expert_stomata_objects_total",
    "stomata_objects_measured",
    "stomata_measurement_coverage_fraction",
    "stomatal_complex_area_unit",
    "stomatal_complex_area_mean",
    "stomatal_complex_area_median",
    "stomatal_complex_area_sd",
    "stomatal_complex_area_q25",
    "stomatal_complex_area_q75",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty dataset: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def exact_index(
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
    label: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in index:
            raise ValueError(f"Duplicate {label} key: {key}")
        index[key] = row
    return index


def build_products(
    plants: list[dict[str, str]],
    detections: list[dict[str, str]],
    measurements: list[dict[str, str]],
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    plant_by_image_key = exact_index(
        plants,
        ("stomata_image",),
        "plant image",
    )
    plant_by_image = {
        key[0]: row for key, row in plant_by_image_key.items()
    }
    plant_by_join_key = exact_index(
        plants,
        ("join_key_variety", "join_key_treatment"),
        "canonical variety+treatment",
    )
    if len(plants) != 32 or len(plant_by_join_key) != 32:
        raise ValueError(
            "Expected exactly 32 unique plant records keyed by "
            "canonical variety+treatment"
        )

    detections_by_image: dict[str, list[dict]] = defaultdict(list)
    detection_rows: list[dict] = []
    for detection in detections:
        filename = detection["filename"]
        if filename not in plant_by_image:
            raise ValueError(f"Unmatched detection image: {filename}")
        plant = plant_by_image[filename]
        record = {
            **{field: plant.get(field, "") for field in PLANT_CONTEXT_FIELDS},
            **{field: plant.get(field, "") for field in THERMAL_FIELDS},
            "detection_id": detection["detection_id"],
            "detector_status": detection["status"],
            "detector_confidence": numeric(detection["confidence"]),
            "raw_detector_score": numeric(detection["raw_detector_score"]),
            "calibrated_p_stoma": numeric(detection["p_stoma"]),
            "decision_status": detection["decision_status"],
            "bbox_x1_px": numeric(detection["x1"]),
            "bbox_y1_px": numeric(detection["y1"]),
            "bbox_x2_px": numeric(detection["x2"]),
            "bbox_y2_px": numeric(detection["y2"]),
        }
        detection_rows.append(record)
        detections_by_image[filename].append(record)

    measurements_by_image: dict[str, list[dict]] = defaultdict(list)
    measurement_rows: list[dict] = []
    for measurement in measurements:
        filename = measurement["stomata_image"]
        if filename not in plant_by_image:
            raise ValueError(f"Unmatched measurement image: {filename}")
        plant = plant_by_image[filename]
        record = {
            **{field: plant.get(field, "") for field in PLANT_CONTEXT_FIELDS},
            **{field: plant.get(field, "") for field in THERMAL_FIELDS},
            **{
                field: value
                for field, value in measurement.items()
                if field not in {
                    "study_id",
                    "plant_unit_id",
                    "genotype_id",
                    "variety",
                    "water_status",
                    "water_ml",
                    "stomata_image",
                    "filename",
                }
            },
        }
        measurement_rows.append(record)
        measurements_by_image[filename].append(record)

    audit_rows: list[dict] = []
    hierarchical_plants: list[dict] = []
    for plant in plants:
        filename = plant["stomata_image"]
        image_detections = detections_by_image[filename]
        image_measurements = measurements_by_image[filename]
        decisions = Counter(row["decision_status"] for row in image_detections)
        p_values = [
            row["calibrated_p_stoma"]
            for row in image_detections
            if row["calibrated_p_stoma"] is not None
        ]
        audit_rows.append({
            "join_key_variety": plant["join_key_variety"],
            "join_key_treatment": plant["join_key_treatment"],
            "join_method": plant["join_method"],
            "stomata_image": filename,
            "stomata_photo_serial": plant["stomata_photo_serial"],
            "thermal_session_plant_id": plant["thermal_session_plant_id"],
            "thermal_source_file": plant["thermal_source_file"],
            "plant_match_count": 1,
            "detection_rows_matched": len(image_detections),
            "measurement_rows_matched": len(image_measurements),
            "join_status": "exact_match",
        })
        hierarchical_plants.append({
            "identity": {
                field: plant.get(field, "") for field in PLANT_CONTEXT_FIELDS
            },
            "thermal": {
                field: plant.get(field, "") for field in THERMAL_FIELDS
            },
            "stomata": {
                "aggregate": {
                    **{
                        field: plant.get(field, "")
                        for field in STOMATA_AGGREGATE_FIELDS
                    },
                    "detector_confidence_mean": (
                        mean(
                            row["detector_confidence"]
                            for row in image_detections
                            if row["detector_confidence"] is not None
                        )
                        if image_detections else None
                    ),
                    "calibrated_p_stoma_mean": (
                        mean(p_values) if p_values else None
                    ),
                    "decision_counts_recomputed": dict(decisions),
                },
                "detections": image_detections,
                "measurements": image_measurements,
            },
        })

    if len(detection_rows) != len(detections):
        raise ValueError("Detection join changed the detection row count")
    if len(measurement_rows) != len(measurements):
        raise ValueError("Measurement join changed the measurement row count")
    if any(row["join_status"] != "exact_match" for row in audit_rows):
        raise ValueError("At least one plant failed the exact join audit")

    product = {
        "schema_version": "integrated_exact_v1",
        "join_contract": {
            "fields": ["canonical variety", "canonical treatment"],
            "logic": "both fields must match exactly",
            "fallback_join": None,
            "fuzzy_matching": False,
            "duplicate_keys_allowed": False,
        },
        "summary": {
            "plants": len(plants),
            "detections": len(detection_rows),
            "manual_measurements": len(measurement_rows),
            "exact_matches": len(audit_rows),
            "unmatched": 0,
        },
        "plants": hierarchical_plants,
    }
    return plants, detection_rows, measurement_rows, audit_rows, product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plants",
        type=Path,
        default=Path(
            "results/combined_quality_mask_analysis_v1/"
            "plant_phenotypes.csv"
        ),
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path(
            "validation_artifacts/tomato_production_v1/"
            "calibrated_detections.csv"
        ),
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path(
            "results/quality_mask_analysis_v1/"
            "stomatal_complex_measurements.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/integrated_exact_v1"),
    )
    args = parser.parse_args()

    outputs = build_products(
        read_csv(args.plants),
        read_csv(args.detections),
        read_csv(args.measurements),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "combined_plant_image_dataset.csv", outputs[0])
    write_csv(args.output / "combined_detection_dataset.csv", outputs[1])
    write_csv(args.output / "combined_measurement_dataset.csv", outputs[2])
    write_csv(args.output / "exact_join_audit.csv", outputs[3])
    (args.output / "combined_product_dataset.json").write_text(
        json.dumps(outputs[4], indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(outputs[0])} plants, {len(outputs[1])} detections, "
        f"and {len(outputs[2])} manual measurements to {args.output}"
    )


if __name__ == "__main__":
    main()
