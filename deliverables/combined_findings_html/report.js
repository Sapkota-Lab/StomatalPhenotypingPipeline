const report = JSON.parse(document.getElementById("report-data").textContent);

const number = (value, digits = 0) =>
  value == null
    ? "Not available"
    : Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

const signed = (value, digits = 1, suffix = "") => {
  if (value == null) return "Not available";
  const sign = value > 0 ? "+" : "";
  return `${sign}${number(value, digits)}${suffix}`;
};

const probability = (value) =>
  value == null ? "Not available" : `${number(value * 100, 1)}%`;

const pvalue = (value) => {
  if (value == null) return "Not available";
  if (value < 0.001) return "<0.001";
  return number(value, 3);
};

const setText = (id, text) => {
  document.getElementById(id).textContent = text;
};

const definitionList = (items) =>
  items
    .map(
      ([term, value]) =>
        `<div><dt>${term}</dt><dd>${value}</dd></div>`
    )
    .join("");

const { counts, paired } = report;
setText("count-plants", number(counts.plants));
setText("count-varieties", number(counts.varieties));
setText("count-masks", number(counts.manual_stomata));
setText("count-joins", `${number(counts.exact_joins)}/${number(counts.plants)}`);

setText("stomata-delta", signed(paired.stomatal.percent_difference, 1, "%"));
setText(
  "stomata-summary",
  `Control averaged ${number(
    paired.stomatal.control_mean,
    0
  )} px² versus ${number(
    paired.stomatal.regular_mean,
    0
  )} px² for Regular. The 95% confidence interval includes zero.`
);
document.getElementById("stomata-stats").innerHTML = definitionList([
  ["Control - Regular", `${signed(paired.stomatal.difference, 0)} px²`],
  [
    "95% CI",
    `${signed(paired.stomatal.ci_low, 0)} to ${signed(
      paired.stomatal.ci_high,
      0
    )}`,
  ],
  ["Paired t-test", `p = ${pvalue(paired.stomatal.paired_t_p)}`],
  ["Pairs", number(paired.stomatal.n_pairs)],
]);

setText("temp-delta", `${signed(paired.temperature.difference, 2)}°C`);
setText(
  "temp-summary",
  `QC-filtered Control plants averaged ${number(
    paired.temperature.control_mean,
    2
  )}°C versus ${number(
    paired.temperature.regular_mean,
    2
  )}°C for Regular. Control was slightly cooler, but the interval includes zero.`
);
document.getElementById("temp-stats").innerHTML = definitionList([
  [
    "95% CI",
    `${signed(paired.temperature.ci_low, 2)} to ${signed(
      paired.temperature.ci_high,
      2
    )}°C`,
  ],
  ["Paired t-test", `p = ${pvalue(paired.temperature.paired_t_p)}`],
  ["Wilcoxon", `p = ${pvalue(paired.temperature.wilcoxon_p)}`],
  ["QC pairs", number(paired.temperature.n_pairs)],
]);

setText(
  "projected-delta",
  signed(paired.projected_area.percent_difference, 1, "%")
);
setText(
  "projected-summary",
  `QC-filtered Control masks averaged ${number(
    paired.projected_area.control_mean,
    0
  )} px² versus ${number(
    paired.projected_area.regular_mean,
    0
  )} px² for Regular. The paired signal is statistically detectable.`
);
document.getElementById("projected-stats").innerHTML = definitionList([
  [
    "Control - Regular",
    `${signed(paired.projected_area.difference, 0)} px²`,
  ],
  [
    "95% CI",
    `${signed(paired.projected_area.ci_low, 0)} to ${signed(
      paired.projected_area.ci_high,
      0
    )}`,
  ],
  ["Paired t-test", `p = ${pvalue(paired.projected_area.paired_t_p)}`],
  ["QC pairs", number(paired.projected_area.n_pairs)],
]);

const selector = document.getElementById("variety-selector");
report.pairs.forEach((pair, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = pair.variety;
  button.dataset.index = index;
  button.setAttribute("role", "listitem");
  if (index === 0) button.classList.add("active");
  button.addEventListener("click", () => selectPair(index));
  selector.appendChild(button);
});

const confidenceLine = (plant, status) => {
  const item = plant.detection_confidence_by_status[status];
  if (!item) return "Not available";
  return `${number(item.n)} at ${probability(
    item.mean_calibrated_probability
  )} mean calibrated probability`;
};

const plantCard = (plant, treatment) => {
  const qcClass = plant.thermal_mask_qc_passed ? "qc-pass" : "qc-fail";
  const qcText = plant.thermal_mask_qc_passed
    ? "Thermal mask QC passed"
    : "Thermal mask QC failed";
  return `
    <header class="plant-head">
      <div>
        <h4>${treatment}</h4>
        <p>${plant.water_ml} mL, ${plant.water_status.replaceAll("_", " ")}</p>
      </div>
      <p>${plant.plant_unit_id}</p>
    </header>
    <div class="plant-meta">
      <div><span>Stomata image</span><strong>${plant.stomata_image}</strong></div>
      <div><span>Photo serial</span><strong>${plant.stomata_photo_serial}</strong></div>
      <div class="span-2"><span>Thermal source</span><strong>${plant.thermal_source_file}</strong></div>
    </div>
    <section class="metric-group">
      <h5>Expert stomatal-complex measurements</h5>
      <dl>
        <div><dt>Measured objects</dt><dd>${number(
          plant.stomata_objects_measured
        )}</dd></div>
        <div><dt>Mean area</dt><dd>${number(
          plant.stomatal_complex_area_mean,
          0
        )} px²</dd></div>
        <div><dt>Median area</dt><dd>${number(
          plant.stomatal_complex_area_median,
          0
        )} px²</dd></div>
        <div><dt>Mean length</dt><dd>${number(
          plant.mean_length_px,
          1
        )} px</dd></div>
        <div><dt>Mean width</dt><dd>${number(
          plant.mean_width_px,
          1
        )} px</dd></div>
        <div><dt>Mean aspect ratio</dt><dd>${number(
          plant.mean_aspect_ratio,
          2
        )}</dd></div>
      </dl>
    </section>
    <section class="metric-group">
      <h5>Detector candidate evidence</h5>
      <dl>
        <div><dt>Total candidates</dt><dd>${number(
          plant.stomata_candidates_total
        )}</dd></div>
        <div><dt>Accepted</dt><dd>${number(
          plant.stomata_candidates_accepted
        )}</dd></div>
        <div><dt>Needs review</dt><dd>${number(
          plant.stomata_candidates_review
        )}</dd></div>
        <div class="span-2"><dt>Accepted confidence</dt><dd>${confidenceLine(
          plant,
          "accepted"
        )}</dd></div>
        <div><dt>Expected true candidates</dt><dd>${number(
          plant.stomata_expected_true_candidates,
          1
        )}</dd></div>
      </dl>
    </section>
    <section class="metric-group">
      <h5>SeedThermal findings</h5>
      <dl>
        <div><dt>Canopy mean</dt><dd>${number(
          plant.canopy_mean_c,
          2
        )}°C</dd></div>
        <div><dt>Canopy median</dt><dd>${number(
          plant.canopy_median_c,
          2
        )}°C</dd></div>
        <div><dt>Canopy range</dt><dd>${number(
          plant.canopy_min_c,
          2
        )}-${number(plant.canopy_max_c, 2)}°C</dd></div>
        <div><dt>ΔT from 25°C</dt><dd>${signed(
          plant.canopy_delta_t_air_c,
          2
        )}°C</dd></div>
        <div><dt>Projected selected area</dt><dd>${number(
          plant.projected_plant_area_px2,
          0
        )} px²</dd></div>
        <div><dt>Mask status</dt><dd><span class="${qcClass}">${qcText}</span></dd></div>
      </dl>
    </section>
  `;
};

function selectPair(index) {
  const pair = report.pairs[index];
  document
    .querySelectorAll("#variety-selector button")
    .forEach((button, i) => button.classList.toggle("active", i === index));
  setText("pair-genotype", pair.genotype_id);
  setText("pair-variety", pair.variety);
  setText(
    "pair-qc",
    pair.thermal_pair_qc
      ? "Complete thermal pair passes QC"
      : "Thermal pair shown, but excluded from paired thermal inference"
  );
  document.getElementById("regular-card").innerHTML = plantCard(
    pair.regular,
    "Regular"
  );
  document.getElementById("control-card").innerHTML = plantCard(
    pair.control,
    "Control"
  );
  document.getElementById("pair-interpretation").innerHTML = `
    <div><span>Stomatal area, Control - Regular</span><strong>${signed(
      pair.stomatal_area_difference,
      0
    )} px²</strong></div>
    <div><span>Canopy temperature, Control - Regular</span><strong>${signed(
      pair.temperature_difference,
      2
    )}°C</strong></div>
    <div><span>Projected selected area, Control - Regular</span><strong>${
      pair.thermal_pair_qc
        ? `${signed(pair.projected_area_difference, 0)} px²`
        : "Excluded by thermal QC"
    }</strong></div>
  `;
}

selectPair(0);

const chart = document.getElementById("variety-chart");
const chartIntro = document.getElementById("chart-intro");
const chartNote = document.getElementById("chart-note");

function renderChart(metric) {
  const isStomatal = metric === "stomatal";
  const items = [...report.pairs].sort((a, b) => {
    const aValue = isStomatal
      ? a.stomatal_area_average
      : (a.regular.projected_plant_area_px2 +
          a.control.projected_plant_area_px2) /
        2;
    const bValue = isStomatal
      ? b.stomatal_area_average
      : (b.regular.projected_plant_area_px2 +
          b.control.projected_plant_area_px2) /
        2;
    return bValue - aValue;
  });
  const values = items.flatMap((pair) =>
    isStomatal
      ? [
          pair.regular.stomatal_complex_area_mean,
          pair.control.stomatal_complex_area_mean,
        ]
      : [
          pair.regular.thermal_mask_qc_passed
            ? pair.regular.projected_plant_area_px2
            : null,
          pair.control.thermal_mask_qc_passed
            ? pair.control.projected_plant_area_px2
            : null,
        ]
  );
  const max = Math.max(...values.filter((value) => value != null));
  chart.innerHTML = items
    .map((pair) => {
      const regular = isStomatal
        ? pair.regular.stomatal_complex_area_mean
        : pair.regular.thermal_mask_qc_passed
        ? pair.regular.projected_plant_area_px2
        : null;
      const control = isStomatal
        ? pair.control.stomatal_complex_area_mean
        : pair.control.thermal_mask_qc_passed
        ? pair.control.projected_plant_area_px2
        : null;
      const regularWidth = regular == null ? 0 : (regular / max) * 100;
      const controlWidth = control == null ? 0 : (control / max) * 100;
      return `
        <div class="chart-row">
          <div class="chart-row-label" title="${pair.variety}">${pair.variety}</div>
          <div class="bar-zone">
            <span class="bar regular" style="width:${regularWidth}%"></span>
            <span class="bar control" style="width:${controlWidth}%"></span>
          </div>
          <div class="chart-values">
            <div class="regular-text">R ${number(regular, 0)}</div>
            <div class="control-text">C ${number(control, 0)}</div>
          </div>
        </div>`;
    })
    .join("");
  chartIntro.textContent = isStomatal
    ? "Descriptively, mean stomatal-complex area across treatments ranges from 6,461 px² for NC264G to 13,860 px² for 26-037. Each treatment mean is based on only two or three selected expert masks."
    : "Projected selected area is shown only where the thermal mask passed quality control.";
  chartNote.textContent = isStomatal
    ? "Regular is green and Control is orange. Ranking reflects this pilot sample only and should not be generalized as a genotype effect."
    : "Regular is green and Control is orange. Missing bars mark failed masks. Do not compare the two failed projected-area values with QC-passing masks.";
}

document.querySelectorAll(".metric-switch button").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".metric-switch button")
      .forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderChart(button.dataset.metric);
  });
});
renderChart("stomatal");

const treatmentGrid = document.getElementById("treatment-grid");
treatmentGrid.innerHTML = [
  {
    title: "Stomatal-complex area",
    item: paired.stomatal,
    digits: 0,
    unit: "px²",
    note: "No clear paired difference. Manual masks, 16 complete pairs.",
  },
  {
    title: "Canopy mean temperature",
    item: paired.temperature,
    digits: 2,
    unit: "°C",
    note: "No clear paired difference. Thermal-QC subset, 14 complete pairs.",
  },
  {
    title: "Projected selected area",
    item: paired.projected_area,
    digits: 0,
    unit: "px²",
    note: "Paired difference. Thermal-QC subset, 14 complete pairs.",
  },
]
  .map(
    ({ title, item, digits, unit, note }) => `
      <article class="treatment-item">
        <h3>${title}</h3>
        <div class="comparison-numbers">
          <div class="regular-value"><span>Regular, watered</span><strong>${number(
            item.regular_mean,
            digits
          )} ${unit}</strong></div>
          <div class="control-value"><span>Control, drought</span><strong>${number(
            item.control_mean,
            digits
          )} ${unit}</strong></div>
        </div>
        <p>${note} Control - Regular: ${signed(
      item.difference,
      digits
    )} ${unit}; 95% CI ${signed(item.ci_low, digits)} to ${signed(
      item.ci_high,
      digits
    )}; paired p = ${pvalue(item.paired_t_p)}.</p>
      </article>`
  )
  .join("");

document.getElementById("evidence-body").innerHTML = report.pairs
  .map((pair) => {
    const regularFail = pair.regular.thermal_mask_qc_passed
      ? ""
      : "qc-cell-fail";
    const controlFail = pair.control.thermal_mask_qc_passed
      ? ""
      : "qc-cell-fail";
    const projectedDelta = pair.thermal_pair_qc
      ? signed(pair.projected_area_difference, 0)
      : "Excluded";
    return `<tr>
      <td>${pair.variety}</td>
      <td class="cell-stomata group-start">${number(pair.regular.stomatal_complex_area_mean, 0)}</td>
      <td class="cell-stomata">${number(pair.control.stomatal_complex_area_mean, 0)}</td>
      <td class="cell-stomata">${signed(pair.stomatal_area_difference, 0)}</td>
      <td class="cell-temperature group-start">${number(pair.regular.canopy_mean_c, 2)}</td>
      <td class="cell-temperature">${number(pair.control.canopy_mean_c, 2)}</td>
      <td class="cell-temperature">${signed(pair.temperature_difference, 2)}</td>
      <td class="cell-projected group-start ${regularFail}" title="${
      regularFail ? "Failed thermal mask QC" : "Passed thermal mask QC"
    }">${number(pair.regular.projected_plant_area_px2, 0)}</td>
      <td class="cell-projected ${controlFail}" title="${
      controlFail ? "Failed thermal mask QC" : "Passed thermal mask QC"
    }">${number(pair.control.projected_plant_area_px2, 0)}</td>
      <td class="cell-projected">${projectedDelta}</td>
    </tr>`;
  })
  .join("");
