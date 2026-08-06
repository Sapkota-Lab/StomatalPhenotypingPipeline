#!/usr/bin/env python3
"""Build plant, variety, and treatment tables for the combined experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


TRUE_VALUES = {"true", "1", "yes", "passed", "pass"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(
    manifest_rows: list[dict[str, str]],
    area_plant_rows: list[dict[str, str]],
    area_variety_rows: list[dict[str, str]],
    area_treatment_rows: list[dict[str, str]],
    validated_stomata_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    area_by_plant = {
        row["plant_unit_id"]: row for row in area_plant_rows
    }
    stomata_by_image = defaultdict(list)
    for row in validated_stomata_rows:
        stomata_by_image[row["stomata_image"]].append(row)
    predictions_by_image = defaultdict(list)
    for row in prediction_rows:
        predictions_by_image[row["stomata_image"]].append(row)
    plant_rows = []
    for manifest in sorted(
        manifest_rows,
        key=lambda row: (row["genotype_id"], row["water_status"]),
    ):
        area = area_by_plant.get(manifest["plant_unit_id"], {})
        objects = stomata_by_image[manifest["stomata_image"]]
        measured = sum(
            row.get("measurement_quality_passed", "").lower()
            in TRUE_VALUES
            for row in objects
        )
        predictions = predictions_by_image[manifest["stomata_image"]]
        decisions = Counter(
            row.get("decision_status", "unknown") for row in predictions
        )
        expected_true = sum(
            float(row["p_stoma"])
            for row in predictions
            if row.get("p_stoma")
        )
        plant_rows.append({
            **manifest,
            "stomata_detection_run_status": (
                "calibrated_detection_complete"
                if predictions else "not_available"
            ),
            "stomata_candidates_total": len(predictions),
            "stomata_candidates_accepted": decisions.get("accepted", 0),
            "stomata_candidates_review": decisions.get("review", 0),
            "stomata_candidates_rejected": decisions.get("rejected", 0),
            "stomata_expected_true_candidates": expected_true,
            "expert_stomata_objects_total": len(objects),
            "stomata_objects_measured": measured,
            "stomata_measurement_coverage_fraction": (
                measured / len(objects) if objects else None
            ),
            "stomatal_complex_area_unit": area.get("area_unit", ""),
            "stomatal_complex_area_mean": number(area.get("mean_area")),
            "stomatal_complex_area_median": number(area.get("median_area")),
            "stomatal_complex_area_sd": number(area.get("area_sd")),
            "stomatal_complex_area_q25": number(area.get("area_q25")),
            "stomatal_complex_area_q75": number(area.get("area_q75")),
        })

    manifest_by_genotype_status = {
        (row["genotype_id"], row["water_status"]): row
        for row in manifest_rows
    }
    variety_rows = []
    for area in area_variety_rows:
        genotype = area["genotype_id"]
        watered = manifest_by_genotype_status[(genotype, "watered")]
        drought = manifest_by_genotype_status[(genotype, "drought_stressed")]
        thermal_fields = {}
        for field in (
            "canopy_mean_c",
            "canopy_delta_t_air_c",
            "projected_plant_area_px2",
            "projected_plant_area_fraction",
        ):
            watered_value = number(watered.get(field))
            drought_value = number(drought.get(field))
            thermal_fields[f"watered_{field}"] = watered_value
            thermal_fields[f"drought_{field}"] = drought_value
            thermal_fields[f"difference_drought_minus_watered_{field}"] = (
                drought_value - watered_value
                if watered_value is not None and drought_value is not None
                else None
            )
        variety_rows.append({
            **area,
            **thermal_fields,
            "watered_thermal_mask_qc_passed": watered[
                "thermal_mask_qc_passed"
            ],
            "drought_thermal_mask_qc_passed": drought[
                "thermal_mask_qc_passed"
            ],
        })

    area_treatment_by_status = {
        row["water_status"]: row for row in area_treatment_rows
    }
    treatment_rows = []
    for status in ("watered", "drought_stressed"):
        selected = [
            row for row in plant_rows if row["water_status"] == status
        ]
        thermal_qc = [
            row for row in selected
            if row["thermal_mask_qc_passed"].lower() in TRUE_VALUES
        ]
        record = dict(area_treatment_by_status.get(status, {}))
        record.update({
            "water_status": status,
            "n_thermal_plants_qc_passed": len(thermal_qc),
        })
        for field in (
            "canopy_mean_c",
            "canopy_delta_t_air_c",
            "projected_plant_area_px2",
            "projected_plant_area_fraction",
        ):
            values = [
                number(row[field]) for row in thermal_qc
                if number(row[field]) is not None
            ]
            record[f"mean_{field}_qc_passed"] = (
                mean(values) if values else None
            )
        treatment_rows.append(record)
    return plant_rows, variety_rows, treatment_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    parser.add_argument("--area-results", type=Path, required=True)
    parser.add_argument("--stomata", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "validation_artifacts/tomato_production_v1/stomata.csv"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = build_tables(
        read_csv(args.manifest),
        read_csv(args.area_results / "plant_stomatal_area.csv"),
        read_csv(args.area_results / "variety_stomatal_area.csv"),
        read_csv(args.area_results / "treatment_stomatal_area.csv"),
        read_csv(args.stomata),
        read_csv(args.predictions),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "plant_phenotypes.csv", tables[0])
    write_csv(args.output / "variety_phenotypes.csv", tables[1])
    write_csv(args.output / "treatment_comparison.csv", tables[2])
    metadata = {
        "plants": len(tables[0]),
        "varieties": len(tables[1]),
        "treatments": len(tables[2]),
        "treatment_mapping": {
            "Regular": "watered",
            "Control": "drought_stressed",
        },
        "statistical_unit": "plant",
    }
    (args.output / "combined_analysis_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote combined phenotype tables to {args.output}")


if __name__ == "__main__":
    main()
