from __future__ import annotations

import base64
import json
import math
import mimetypes
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "combined_findings_html"
ASSETS = OUT / "assets"
PLANTS_CSV = ROOT / "results" / "combined_quality_mask_analysis_v1" / "plant_phenotypes.csv"
VARIETIES_CSV = ROOT / "results" / "combined_quality_mask_analysis_v1" / "variety_phenotypes.csv"
DETECTIONS_CSV = ROOT / "results" / "integrated_exact_v1" / "combined_detection_dataset.csv"
MEASUREMENTS_CSV = ROOT / "results" / "integrated_exact_v1" / "combined_measurement_dataset.csv"
MASK_EXAMPLE = ROOT / "deliverables" / "_report_assets" / "manual_mask_example.png"
THERMAL_ROOT = ROOT.parent / "SeedThermal"
THERMAL_ASSETS = THERMAL_ROOT / "dashboard" / "src" / "assets" / "images"


def clean(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def ci95(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    return tuple(
        float(v)
        for v in stats.t.interval(
            0.95,
            len(values) - 1,
            loc=values.mean(),
            scale=stats.sem(values),
        )
    )


def paired_summary(varieties: pd.DataFrame) -> dict:
    stomatal_delta = varieties["difference_drought_minus_watered"].dropna()
    thermal_qc = varieties[
        varieties["watered_thermal_mask_qc_passed"].astype(bool)
        & varieties["drought_thermal_mask_qc_passed"].astype(bool)
    ].copy()
    temp_delta = thermal_qc[
        "difference_drought_minus_watered_canopy_mean_c"
    ].dropna()
    area_delta = thermal_qc[
        "difference_drought_minus_watered_projected_plant_area_px2"
    ].dropna()

    def tests(delta: pd.Series) -> dict:
        return {
            "n_pairs": int(len(delta)),
            "difference": float(delta.mean()),
            "ci_low": ci95(delta)[0],
            "ci_high": ci95(delta)[1],
            "paired_t_p": float(stats.ttest_1samp(delta, 0).pvalue),
            "wilcoxon_p": float(stats.wilcoxon(delta).pvalue),
        }

    result = {
        "stomatal": {
            **tests(stomatal_delta),
            "regular_mean": float(varieties["watered_mean_area"].mean()),
            "control_mean": float(varieties["drought_mean_area"].mean()),
            "percent_difference": float(
                100
                * stomatal_delta.mean()
                / varieties["watered_mean_area"].mean()
            ),
            "control_greater": int((stomatal_delta > 0).sum()),
        },
        "temperature": {
            **tests(temp_delta),
            "regular_mean": float(thermal_qc["watered_canopy_mean_c"].mean()),
            "control_mean": float(thermal_qc["drought_canopy_mean_c"].mean()),
            "percent_difference": float(
                100
                * temp_delta.mean()
                / thermal_qc["watered_canopy_mean_c"].mean()
            ),
            "control_greater": int((temp_delta > 0).sum()),
        },
        "projected_area": {
            **tests(area_delta),
            "regular_mean": float(
                thermal_qc["watered_projected_plant_area_px2"].mean()
            ),
            "control_mean": float(
                thermal_qc["drought_projected_plant_area_px2"].mean()
            ),
            "percent_difference": float(
                100
                * area_delta.mean()
                / thermal_qc["watered_projected_plant_area_px2"].mean()
            ),
            "control_greater": int((area_delta > 0).sum()),
        },
    }
    return result


def build_dataset() -> dict:
    plants = pd.read_csv(PLANTS_CSV)
    varieties = pd.read_csv(VARIETIES_CSV)
    detections = pd.read_csv(DETECTIONS_CSV)
    measurements = pd.read_csv(MEASUREMENTS_CSV)

    detection_summary = (
        detections.groupby(["plant_unit_id", "decision_status"])
        .agg(
            n=("detection_id", "count"),
            mean_detector_confidence=("detector_confidence", "mean"),
            mean_calibrated_probability=("calibrated_p_stoma", "mean"),
        )
        .reset_index()
    )
    measurement_summary = (
        measurements.groupby("plant_unit_id")
        .agg(
            mean_length_px=("complex_length_px", "mean"),
            mean_width_px=("complex_width_px", "mean"),
            mean_aspect_ratio=("complex_aspect_ratio", "mean"),
            mean_eccentricity=("complex_eccentricity", "mean"),
            mean_solidity=("complex_solidity", "mean"),
            mean_circularity=("complex_circularity", "mean"),
            mean_orientation_deg=("orientation_deg", "mean"),
        )
        .reset_index()
    )
    plants = plants.merge(measurement_summary, on="plant_unit_id", how="left")

    detection_lookup: dict[str, dict] = {}
    for row in detection_summary.to_dict("records"):
        detection_lookup.setdefault(row["plant_unit_id"], {})[
            row["decision_status"]
        ] = {
            "n": int(row["n"]),
            "mean_detector_confidence": clean(row["mean_detector_confidence"]),
            "mean_calibrated_probability": clean(
                row["mean_calibrated_probability"]
            ),
        }

    plant_records = {}
    for row in plants.to_dict("records"):
        row = {key: clean(value) for key, value in row.items()}
        row["detection_confidence_by_status"] = detection_lookup.get(
            row["plant_unit_id"], {}
        )
        plant_records[row["plant_unit_id"]] = row

    pair_records = []
    for row in varieties.to_dict("records"):
        row = {key: clean(value) for key, value in row.items()}
        regular = plant_records[row["watered_plant_unit_id"]]
        control = plant_records[row["drought_plant_unit_id"]]
        pair_records.append(
            {
                "genotype_id": row["genotype_id"],
                "variety": row["variety"],
                "regular": regular,
                "control": control,
                "stomatal_area_average": row["descriptive_mean_area"],
                "stomatal_area_difference": row[
                    "difference_drought_minus_watered"
                ],
                "temperature_difference": row[
                    "difference_drought_minus_watered_canopy_mean_c"
                ],
                "projected_area_difference": row[
                    "difference_drought_minus_watered_projected_plant_area_px2"
                ],
                "thermal_pair_qc": bool(
                    row["watered_thermal_mask_qc_passed"]
                    and row["drought_thermal_mask_qc_passed"]
                ),
            }
        )

    return {
        "generated_from": {
            "plants": str(PLANTS_CSV.relative_to(ROOT)),
            "detections": str(DETECTIONS_CSV.relative_to(ROOT)),
            "measurements": str(MEASUREMENTS_CSV.relative_to(ROOT)),
        },
        "counts": {
            "plants": int(len(plants)),
            "varieties": int(plants["variety"].nunique()),
            "manual_stomata": int(plants["stomata_objects_measured"].sum()),
            "detector_candidates": int(plants["stomata_candidates_total"].sum()),
            "accepted_candidates": int(
                plants["stomata_candidates_accepted"].sum()
            ),
            "review_candidates": int(plants["stomata_candidates_review"].sum()),
            "rejected_candidates": int(
                plants["stomata_candidates_rejected"].sum()
            ),
            "thermal_qc_passed": int(plants["thermal_mask_qc_passed"].sum()),
            "exact_joins": int(
                (plants["join_method"] == "exact_canonical_variety_and_treatment").sum()
            ),
        },
        "paired": paired_summary(varieties),
        "pairs": pair_records,
    }


def copy_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MASK_EXAMPLE, ASSETS / "manual_mask_example.png")
    shutil.copy2(THERMAL_ASSETS / "uk_logo.png", ASSETS / "uk_logo.png")
    thermal_example = (
        THERMAL_ASSETS / "S20260715_P25_MountainGirl_Control_thermal.png"
    )
    shutil.copy2(thermal_example, ASSETS / "thermal_example.png")


def write_report(data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <meta name="description" content="Combined SeedThermal and stomatal phenotyping findings for 16 tomato varieties under Regular and Control treatments.">
  <title>Tomato Drought Study - Sapkota Horticultural Breeding and Genetics Lab</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <a class="skip-link" href="#main">Skip to findings</a>
  <header class="topbar">
    <nav aria-label="Report sections">
      <a href="#answers">Answers</a>
      <a href="#plants">Plants</a>
      <a href="#varieties">Varieties</a>
    </nav>
  </header>

  <main id="main">
    <section class="hero" id="top">
      <div class="hero-copy">
        <div class="lab-lockup">
          <img src="assets/uk_logo.png" alt="University of Kentucky logo">
        </div>
        <h1>Tomato Drought Study - Sapkota Horticultural Breeding and Genetics Lab</h1>
        <p class="hero-deck">Integrated thermal and stomatal findings from 32 plants and 16 tomato varieties.</p>
        <a class="primary-action" href="#plants">Inspect every plant</a>
      </div>
    </section>

    <section class="provenance-strip" aria-label="Dataset scope">
      <div><strong id="count-plants"></strong><span>plant units</span></div>
      <div><strong id="count-varieties"></strong><span>paired varieties</span></div>
      <div><strong id="count-masks"></strong><span>manual stomatal masks</span></div>
      <div><strong id="count-joins"></strong><span>exact composite joins</span></div>
    </section>

    <section class="section answers" id="answers">
      <div class="section-heading">
        <h2>The answers, with uncertainty intact</h2>
        <p>Control means 0 mL and drought-stressed. Regular means 100 mL and watered.</p>
      </div>
      <div class="answer-ledger">
        <article class="answer answer-primary">
          <p class="answer-number" id="stomata-delta"></p>
          <h3>No clear treatment difference in mean stomatal-complex area</h3>
          <p id="stomata-summary"></p>
          <dl id="stomata-stats"></dl>
        </article>
        <article class="answer">
          <p class="answer-number" id="temp-delta"></p>
          <h3>No clear treatment difference in canopy temperature</h3>
          <p id="temp-summary"></p>
          <dl id="temp-stats"></dl>
        </article>
        <article class="answer answer-caution">
          <p class="answer-number" id="projected-delta"></p>
          <h3>A difference in selected thermal-mask area</h3>
          <p id="projected-summary"></p>
          <dl id="projected-stats"></dl>
        </article>
      </div>
    </section>

    <section class="section plant-explorer" id="plants">
      <div class="section-heading">
        <h2>Both projects, plant by plant</h2>
        <p>Select a variety to compare its exact Regular and Control records. No serial-number inference or fuzzy join is used.</p>
      </div>
      <div class="variety-selector" id="variety-selector" role="list" aria-label="Choose a variety"></div>
      <div class="pair-title">
        <div>
          <p id="pair-genotype"></p>
          <h3 id="pair-variety"></h3>
        </div>
        <p id="pair-qc"></p>
      </div>
      <div class="paired-plants">
        <article class="plant-column regular-column" id="regular-card"></article>
        <article class="plant-column control-column" id="control-card"></article>
      </div>
      <div class="pair-interpretation" id="pair-interpretation"></div>
    </section>

    <section class="section variety-analysis" id="varieties">
      <div class="section-heading">
        <h2>Average area among all 16 varieties</h2>
        <p>Use the metric switch to separate stomatal-complex area from SeedThermal projected selected area.</p>
      </div>
      <div class="metric-switch" role="group" aria-label="Choose area metric">
        <button type="button" class="active" data-metric="stomatal">Stomatal-complex area</button>
        <button type="button" data-metric="projected">Projected selected area</button>
      </div>
      <div class="chart-intro" id="chart-intro"></div>
      <div class="variety-chart" id="variety-chart" aria-label="Area comparison across 16 tomato varieties"></div>
      <p class="chart-note" id="chart-note"></p>
    </section>

    <section class="section treatment-comparison">
      <div class="section-heading">
        <h2>Watered compared with drought-stressed</h2>
        <p>Paired differences are Control minus Regular. Confidence intervals and tests use variety as the paired unit.</p>
      </div>
      <div class="treatment-grid" id="treatment-grid"></div>
    </section>

    <section class="section evidence-table-section">
      <div class="section-heading">
        <h2>Complete paired evidence table</h2>
        <p>Every variety appears once. Failed thermal masks remain visible and are marked rather than silently removed.</p>
      </div>
      <div class="table-wrap">
        <table>
          <colgroup>
            <col class="variety-col">
            <col span="3" class="stomata-col">
            <col span="3" class="temperature-col">
            <col span="3" class="projected-col">
          </colgroup>
          <thead>
            <tr>
              <th rowspan="2">Variety</th>
              <th colspan="3" class="group-heading group-stomata group-start">Stomatal-complex area, px²</th>
              <th colspan="3" class="group-heading group-temperature group-start">Canopy mean, °C</th>
              <th colspan="3" class="group-heading group-projected group-start">Projected selected area, px²</th>
            </tr>
            <tr>
              <th class="group-stomata group-start">Regular</th><th class="group-stomata">Control</th><th class="group-stomata">Δ</th>
              <th class="group-temperature group-start">Regular</th><th class="group-temperature">Control</th><th class="group-temperature">Δ</th>
              <th class="group-projected group-start">Regular</th><th class="group-projected">Control</th><th class="group-projected">Δ</th>
            </tr>
          </thead>
          <tbody id="evidence-body"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script src="report.js"></script>
</body>
</html>
"""
    html = html.replace(
        "__REPORT_DATA__",
        json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
            "</", "<\\/"
        ),
    )
    (OUT / "index.html").write_text(html, encoding="utf-8")


def write_standalone_attachment() -> Path:
    html = (OUT / "index.html").read_text(encoding="utf-8")
    css = (OUT / "styles.css").read_text(encoding="utf-8")
    javascript = (OUT / "report.js").read_text(encoding="utf-8")

    html = html.replace(
        '  <link rel="stylesheet" href="styles.css">',
        f"  <style>\n{css}\n  </style>",
    )
    html = html.replace(
        '  <script src="report.js"></script>',
        f"  <script>\n{javascript}\n  </script>",
    )

    for asset in ASSETS.iterdir():
        if not asset.is_file():
            continue
        mime_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
        html = html.replace(
            f"assets/{asset.name}",
            f"data:{mime_type};base64,{encoded}",
        )

    attachment = ROOT / "deliverables" / "Combined_Tomato_Phenotyping_Findings.html"
    attachment.write_text(html, encoding="utf-8")
    return attachment


def main() -> None:
    data = build_dataset()
    copy_assets()
    write_report(data)
    attachment = write_standalone_attachment()
    print(
        json.dumps(
            {
                "output": str(OUT / "index.html"),
                "attachment": str(attachment),
                "plants": data["counts"]["plants"],
                "varieties": data["counts"]["varieties"],
                "exact_joins": data["counts"]["exact_joins"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
