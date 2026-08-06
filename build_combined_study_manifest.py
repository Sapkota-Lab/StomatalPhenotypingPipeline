#!/usr/bin/env python3
"""Join stomatal and SeedThermal observations into stable plant units."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STUDY_ID = "tomato_2026_combined"
THERMAL_SESSION = "S20260715"
TREATMENT_MAP = {
    "regular": ("watered", 100),
    "control": ("drought_stressed", 0),
}
VARIETIES = (
    "NC264G",
    "25203G",
    "25317G",
    "26-016",
    "26-018",
    "25-014",
    "25-013",
    "Cherokee Purple",
    "25-12",
    "Mountain Gem",
    "26-037",
    "26-033",
    "Mountain Girl",
    "Ailsa Craig",
    "NC261G",
    "Mountain Heritage",
)

CANONICAL_VARIETY_BY_SOURCE_LABEL = {
    "NC264G": "NC264G",
    "25203 G": "25203G",
    "25203G": "25203G",
    "25317G": "25317G",
    "26-016": "26-016",
    "26-018": "26-018",
    "25-014": "25-014",
    "25-013": "25-013",
    "Cherokee Purple": "Cherokee Purple",
    "CherokeePurple": "Cherokee Purple",
    "25-12": "25-12",
    "Mountain Gem": "Mountain Gem",
    "MountainGem": "Mountain Gem",
    "26-037": "26-037",
    "26-033": "26-033",
    "Mountain Girl": "Mountain Girl",
    "MountainGirl": "Mountain Girl",
    "Ailsa Craig": "Ailsa Craig",
    "AilsaCraig": "Ailsa Craig",
    "AlisaCraig": "Ailsa Craig",
    "NC261G": "NC261G",
    "Mountain Heritage": "Mountain Heritage",
    "MountainHeritage": "Mountain Heritage",
}
GENOTYPE_ID_BY_VARIETY = {
    name: f"V{index:02d}" for index, name in enumerate(VARIETIES, 1)
}
CANONICAL_TREATMENT_BY_SOURCE_LABEL = {
    "regular": "regular",
    "Regular": "regular",
    "control": "control",
    "Control": "control",
}


def canonical_variety(value: str) -> str:
    source_label = value.strip()
    if source_label not in CANONICAL_VARIETY_BY_SOURCE_LABEL:
        raise ValueError(f"Unknown variety {value!r}")
    return CANONICAL_VARIETY_BY_SOURCE_LABEL[source_label]


def canonical_treatment(value: str) -> str:
    source_label = value.strip()
    if source_label not in CANONICAL_TREATMENT_BY_SOURCE_LABEL:
        raise ValueError(f"Unknown treatment {value!r}")
    return CANONICAL_TREATMENT_BY_SOURCE_LABEL[source_label]


def unique_by_key(rows: list[dict], key_fields: tuple[str, ...], label: str):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError(f"Duplicate {label} key: {key}")
        result[key] = row
    return result


def build_rows(
    stomata_rows: list[dict[str, str]],
    thermal_rows: list[dict[str, str]],
    *,
    require_complete: bool = True,
) -> list[dict[str, object]]:
    mapped_stomata = [
        row for row in stomata_rows if row.get("mapping_status") == "mapped"
    ]
    july15 = [
        row for row in thermal_rows
        if row.get("session_id") == THERMAL_SESSION
    ]

    thermal_index = unique_by_key(
        [
            {
                **row,
                "_variety_key": canonical_variety(row["variety"]),
                "_treatment_key": canonical_treatment(row["treatment"]),
            }
            for row in july15
        ],
        ("_variety_key", "_treatment_key"),
        "thermal",
    )

    output = []
    for stomata in mapped_stomata:
        variety_key = canonical_variety(stomata["plant_name"])
        treatment_code = canonical_treatment(stomata["treatment"])
        thermal = thermal_index.get((variety_key, treatment_code))
        if thermal is None:
            raise ValueError(
                f"No July 15 thermal record for "
                f"{stomata['plant_name']} / {treatment_code}"
            )

        water_status, water_ml = TREATMENT_MAP[treatment_code]
        genotype_id = GENOTYPE_ID_BY_VARIETY[variety_key]
        treatment_suffix = "W" if water_status == "watered" else "D"
        pixel_count = int(float(thermal["pixel_count"]))
        frame_area = 640 * 480
        area_fraction = pixel_count / frame_area
        mask_qc_passed = area_fraction <= 0.10
        output.append({
            "study_id": STUDY_ID,
            "join_method": "exact_canonical_variety_and_treatment",
            "join_key_variety": variety_key,
            "join_key_treatment": treatment_code,
            "stomata_source_variety": stomata["plant_name"],
            "stomata_source_treatment": stomata["treatment"],
            "thermal_source_variety": thermal["variety"],
            "thermal_source_treatment": thermal["treatment"],
            "genotype_id": genotype_id,
            "plant_unit_id": f"TOM26-{genotype_id}-{treatment_suffix}",
            "variety": variety_key,
            "variety_class": thermal["group_type"],
            "treatment_code": treatment_code,
            "water_status": water_status,
            "water_ml": water_ml,
            "stomata_image": stomata["filename"],
            "stomata_photo_serial": stomata["photo_serial"],
            "stomata_analysis_status": "pending_validated_object_ledger",
            "thermal_session_id": thermal["session_id"],
            "thermal_session_plant_id": thermal["plant_id"],
            "thermal_source_file": thermal["source_file"],
            "thermal_capture_time": thermal["capture_time"],
            "canopy_mean_c": float(thermal["mean_c"]),
            "canopy_median_c": float(thermal["median_c"]),
            "canopy_min_c": float(thermal["min_c"]),
            "canopy_max_c": float(thermal["max_c"]),
            "canopy_std_c": float(thermal["std_c"]),
            "canopy_p10_c": float(thermal["p10_c"]),
            "canopy_p90_c": float(thermal["p90_c"]),
            "canopy_delta_t_air_c": float(thermal["delta_t_air"]),
            "projected_plant_area_px2": pixel_count,
            "projected_plant_area_fraction": area_fraction,
            "thermal_mask_qc_passed": mask_qc_passed,
            "thermal_mask_qc_flag": (
                "" if mask_qc_passed
                else "auto_mask_selected_more_than_10_percent_of_frame"
            ),
        })

    output.sort(
        key=lambda row: (
            row["genotype_id"],
            0 if row["water_status"] == "drought_stressed" else 1,
        )
    )
    if require_complete:
        expected = len(VARIETIES) * len(TREATMENT_MAP)
        if len(output) != expected:
            raise ValueError(
                f"Expected {expected} joined plant units, got {len(output)}"
            )
        if len(thermal_index) != expected:
            raise ValueError(
                f"Expected {expected} July 15 thermal records, got "
                f"{len(thermal_index)}"
            )
    unique_by_key(output, ("plant_unit_id",), "combined plant-unit")
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_thermal_data = (
        Path(__file__).resolve().parent.parent
        / "SeedThermal"
        / "processed"
        / "thermal_data_all_sessions.csv"
    )
    parser.add_argument(
        "--stomata-metadata",
        type=Path,
        default=Path("image_metadata.csv"),
    )
    parser.add_argument(
        "--thermal-data",
        type=Path,
        default=default_thermal_data,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    args = parser.parse_args()

    rows = build_rows(
        read_csv(args.stomata_metadata),
        read_csv(args.thermal_data),
    )
    write_csv(args.output, rows)
    failed = sum(not row["thermal_mask_qc_passed"] for row in rows)
    print(
        f"Wrote {len(rows)} plant units across {len(VARIETIES)} varieties "
        f"to {args.output}; {failed} thermal masks require review."
    )


if __name__ == "__main__":
    main()
