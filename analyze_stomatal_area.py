#!/usr/bin/env python3
"""Compare validated per-stoma area by water treatment and variety."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy import stats


TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def image_field(fieldnames: set[str]) -> str:
    for candidate in ("stomata_image", "filename", "image"):
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "Stomata table needs one of: stomata_image, filename, image"
    )


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n_stomata": 0,
            "mean_area": None,
            "median_area": None,
            "area_sd": None,
            "area_q25": None,
            "area_q75": None,
        }
    return {
        "n_stomata": len(values),
        "mean_area": mean(values),
        "median_area": median(values),
        "area_sd": stdev(values) if len(values) > 1 else None,
        "area_q25": quantile(values, 0.25),
        "area_q75": quantile(values, 0.75),
    }


def keep_measurement(
    row: dict[str, str],
    *,
    area_field: str,
    status_field: str,
    accepted_status: str,
    quality_field: str | None,
) -> bool:
    raw_area = row.get(area_field, "").strip()
    if not raw_area:
        return False
    try:
        area = float(raw_area)
    except ValueError:
        return False
    if not math.isfinite(area) or area <= 0:
        return False
    if status_field and row.get(status_field, "").strip() != accepted_status:
        return False
    if quality_field and quality_field in row:
        if row[quality_field].strip().lower() not in TRUE_VALUES:
            return False
    return True


def analyze(
    manifest_rows: list[dict[str, str]],
    stoma_rows: list[dict[str, str]],
    *,
    area_field: str,
    area_unit: str,
    status_field: str,
    accepted_status: str,
    quality_field: str | None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    if not manifest_rows:
        raise ValueError("Combined manifest is empty")
    if not stoma_rows:
        raise ValueError("Stomata table is empty")
    manifest_by_image = {}
    for row in manifest_rows:
        filename = row["stomata_image"]
        if filename in manifest_by_image:
            raise ValueError(f"Duplicate manifest image: {filename}")
        manifest_by_image[filename] = row

    object_fields = set(stoma_rows[0])
    if area_field not in object_fields:
        raise ValueError(f"Stomata table lacks area field {area_field!r}")
    if status_field and status_field not in object_fields:
        raise ValueError(f"Stomata table lacks status field {status_field!r}")
    filename_field = image_field(object_fields)

    areas_by_plant: dict[str, list[float]] = defaultdict(list)
    unmatched_images = set()
    excluded_objects = 0
    for row in stoma_rows:
        filename = Path(row[filename_field]).name
        manifest = manifest_by_image.get(filename)
        if manifest is None:
            unmatched_images.add(filename)
            continue
        if not keep_measurement(
            row,
            area_field=area_field,
            status_field=status_field,
            accepted_status=accepted_status,
            quality_field=quality_field,
        ):
            excluded_objects += 1
            continue
        areas_by_plant[manifest["plant_unit_id"]].append(
            float(row[area_field])
        )
    if unmatched_images:
        examples = ", ".join(sorted(unmatched_images)[:5])
        raise ValueError(f"Stomata rows lack manifest matches: {examples}")

    plant_rows = []
    for manifest in sorted(
        manifest_rows,
        key=lambda row: (row["genotype_id"], row["water_status"]),
    ):
        values = areas_by_plant.get(manifest["plant_unit_id"], [])
        plant_rows.append({
            "study_id": manifest["study_id"],
            "genotype_id": manifest["genotype_id"],
            "plant_unit_id": manifest["plant_unit_id"],
            "variety": manifest["variety"],
            "water_status": manifest["water_status"],
            "water_ml": int(manifest["water_ml"]),
            "stomata_image": manifest["stomata_image"],
            "area_field": area_field,
            "area_unit": area_unit,
            **summarize(values),
        })

    plant_by_genotype_status = {
        (row["genotype_id"], row["water_status"]): row
        for row in plant_rows
    }
    variety_rows = []
    paired_differences = []
    paired_watered = []
    paired_drought = []
    for genotype_id in sorted({row["genotype_id"] for row in plant_rows}):
        watered = plant_by_genotype_status.get((genotype_id, "watered"))
        drought = plant_by_genotype_status.get(
            (genotype_id, "drought_stressed")
        )
        if watered is None or drought is None:
            raise ValueError(f"Incomplete manifest pair for {genotype_id}")
        w_mean = watered["mean_area"]
        d_mean = drought["mean_area"]
        complete = w_mean is not None and d_mean is not None
        difference = d_mean - w_mean if complete else None
        descriptive_mean = mean([w_mean, d_mean]) if complete else None
        if complete:
            paired_watered.append(float(w_mean))
            paired_drought.append(float(d_mean))
            paired_differences.append(float(difference))
        variety_rows.append({
            "genotype_id": genotype_id,
            "variety": watered["variety"],
            "watered_plant_unit_id": watered["plant_unit_id"],
            "drought_plant_unit_id": drought["plant_unit_id"],
            "watered_mean_area": w_mean,
            "drought_mean_area": d_mean,
            "descriptive_mean_area": descriptive_mean,
            "difference_drought_minus_watered": difference,
            "watered_n_stomata": watered["n_stomata"],
            "drought_n_stomata": drought["n_stomata"],
            "complete_pair": complete,
            "area_unit": area_unit,
        })

    treatment_rows = []
    for status in ("watered", "drought_stressed"):
        selected = [
            row for row in plant_rows
            if row["water_status"] == status and row["mean_area"] is not None
        ]
        plant_means = [float(row["mean_area"]) for row in selected]
        treatment_rows.append({
            "water_status": status,
            "n_plants": len(selected),
            "n_stomata": sum(int(row["n_stomata"]) for row in selected),
            "mean_of_plant_mean_area": (
                mean(plant_means) if plant_means else None
            ),
            "median_of_plant_mean_area": (
                median(plant_means) if plant_means else None
            ),
            "sd_of_plant_mean_area": (
                stdev(plant_means) if len(plant_means) > 1 else None
            ),
            "area_unit": area_unit,
        })

    paired = {"n_complete_variety_pairs": len(paired_differences)}
    if len(paired_differences) >= 2:
        paired_array = np.asarray(paired_differences, dtype=float)
        difference_mean = float(paired_array.mean())
        sem = stats.sem(paired_array)
        ci_low, ci_high = stats.t.interval(
            0.95,
            len(paired_array) - 1,
            loc=difference_mean,
            scale=sem,
        )
        t_result = stats.ttest_rel(paired_drought, paired_watered)
        try:
            w_result = stats.wilcoxon(paired_drought, paired_watered)
            wilcoxon_statistic = float(w_result.statistic)
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_statistic = None
            wilcoxon_p = None
        paired.update({
            "difference_definition": "drought_stressed_minus_watered",
            "mean_paired_difference": difference_mean,
            "median_paired_difference": float(np.median(paired_array)),
            "difference_95_ci_low": float(ci_low),
            "difference_95_ci_high": float(ci_high),
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_p_value": float(t_result.pvalue),
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_p_value": wilcoxon_p,
            "cohen_dz": (
                difference_mean / float(paired_array.std(ddof=1))
                if paired_array.std(ddof=1) > 0 else None
            ),
        })

    report = {
        "area_field": area_field,
        "area_unit": area_unit,
        "object_inclusion": {
            "status_field": status_field,
            "accepted_status": accepted_status,
            "quality_field": quality_field,
            "excluded_object_rows": excluded_objects,
        },
        "paired_treatment_comparison": paired,
        "variety_comparison_interpretation": (
            "Descriptive only: this batch has one plant per "
            "variety-by-treatment cell."
        ),
        "statistical_unit": (
            "Plant-level mean stomatal area; individual stomata are subsamples."
        ),
    }
    return plant_rows, variety_rows, treatment_rows, report


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(output: Path, variety_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [row for row in variety_rows if row["complete_pair"]]
    if not complete:
        return
    colors = {"watered": "#277da1", "drought_stressed": "#d1495b"}
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for row in complete:
        values = [row["watered_mean_area"], row["drought_mean_area"]]
        axis.plot([0, 1], values, color="#aeb6bf", linewidth=1, zorder=1)
        axis.scatter(
            [0, 1],
            values,
            c=[colors["watered"], colors["drought_stressed"]],
            s=42,
            zorder=2,
        )
    axis.set_xticks([0, 1], ["Watered", "Drought-stressed"])
    axis.set_ylabel(f"Plant mean stomatal-complex area ({complete[0]['area_unit']})")
    axis.set_title("Paired stomatal-complex area by treatment")
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "paired_treatment_area.png", dpi=220)
    plt.close(figure)

    ordered = sorted(
        complete,
        key=lambda row: row["difference_drought_minus_watered"],
    )
    figure, axis = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    positions = np.arange(len(ordered))
    differences = [
        row["difference_drought_minus_watered"] for row in ordered
    ]
    axis.axvline(0, color="#50555c", linewidth=1)
    axis.hlines(
        positions,
        0,
        differences,
        color=[
            colors["drought_stressed"] if value >= 0 else colors["watered"]
            for value in differences
        ],
        linewidth=2,
    )
    axis.scatter(
        differences,
        positions,
        color=[
            colors["drought_stressed"] if value >= 0 else colors["watered"]
            for value in differences
        ],
        s=45,
    )
    axis.set_yticks(positions, [row["variety"] for row in ordered])
    axis.set_xlabel(
        f"Drought-stressed minus watered area ({complete[0]['area_unit']})"
    )
    axis.set_title("Within-variety stomatal-complex area differences")
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "paired_area_differences.png", dpi=220)
    plt.close(figure)

    ordered = sorted(complete, key=lambda row: row["variety"].lower())
    figure, axis = plt.subplots(figsize=(10.5, 7.4), constrained_layout=True)
    positions = np.arange(len(ordered))
    watered = [row["watered_mean_area"] for row in ordered]
    drought = [row["drought_mean_area"] for row in ordered]
    for position, first, second in zip(positions, watered, drought):
        axis.plot(
            [first, second],
            [position, position],
            color="#b9c0c8",
            linewidth=1.5,
            zorder=1,
        )
    axis.scatter(
        watered,
        positions,
        label="Watered",
        color=colors["watered"],
        s=48,
        zorder=2,
    )
    axis.scatter(
        drought,
        positions,
        label="Drought-stressed",
        color=colors["drought_stressed"],
        s=48,
        zorder=2,
    )
    axis.set_yticks(positions, [row["variety"] for row in ordered])
    axis.set_xlabel(f"Plant mean stomatal-complex area ({complete[0]['area_unit']})")
    axis.set_title("Stomatal-complex area among 16 tomato varieties")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "variety_treatment_area.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    parser.add_argument("--stomata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area-field", default="complex_area_px2")
    parser.add_argument("--area-unit", default="px2")
    parser.add_argument("--status-field", default="decision_status")
    parser.add_argument("--accepted-status", default="accepted")
    parser.add_argument(
        "--quality-field",
        default="measurement_quality_passed",
        help="Set to an empty string when the table has no quality column.",
    )
    args = parser.parse_args()

    outputs = analyze(
        read_csv(args.manifest),
        read_csv(args.stomata),
        area_field=args.area_field,
        area_unit=args.area_unit,
        status_field=args.status_field,
        accepted_status=args.accepted_status,
        quality_field=args.quality_field or None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "plant_stomatal_area.csv", outputs[0])
    write_csv(args.output / "variety_stomatal_area.csv", outputs[1])
    write_csv(args.output / "treatment_stomatal_area.csv", outputs[2])
    write_figures(args.output, outputs[1])
    (args.output / "stomatal_area_analysis.json").write_text(
        json.dumps(outputs[3], indent=2) + "\n"
    )
    print(
        f"Wrote plant, variety, and treatment stomatal-area outputs to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
