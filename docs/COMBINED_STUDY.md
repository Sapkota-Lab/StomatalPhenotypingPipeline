# Combined Tomato Thermal and Stomatal Phenotyping Study

## Scope

`SeedThermal` and `StomatalPhenotypingPipeline` are two measurement pipelines
for one tomato drought experiment. They must be analyzed through one shared
plant manifest:

- SeedThermal contributes canopy-temperature distributions, temperature
  relative to ambient, and projected plant-area measurements.
- StomatalPhenotypingPipeline contributes per-stoma detections, calibrated
  correctness probabilities, stomatal-complex measurements, pore traits, and
  spatial measurements.

The combined study contains 16 varieties and two treatment plants per variety,
for 32 plant units.

## Treatment definition

| Source code | Biological interpretation | Water amount |
|---|---|---:|
| `Regular` / `regular` | Watered | 100 mL |
| `Control` / `control` | Not watered / drought-stressed | 0 mL |

The source treatment code must remain in the data for provenance. Analyses and
figures should use the biological labels `watered` and `drought_stressed`.

## Stable identifiers

SeedThermal session-local IDs are not globally stable. `P01` through `P16`
represent genotype-pair positions in the July 13/14 paired sessions, while
`P01` through `P32` identify individual treatment images on July 15.

The combined project therefore uses:

- `genotype_id`: `V01` through `V16`;
- `plant_unit_id`: `TOM26-{genotype_id}-{W|D}`;
- `thermal_session_id` and `thermal_session_plant_id`: retained only as source
  provenance;
- `stomata_image`: the one stomatal micrograph linked to that plant unit.

Canonical variety names are shared between projects. Spacing and the
`AlisaCraig`/`Ailsa Craig` spelling difference are normalized only for joins;
source spellings remain traceable.

## Required stomatal-area outputs

“Area” must be named explicitly in every output. The primary requested trait
is `stomatal_complex_area`, not thermal projected plant area.

### 1. Watered versus drought-stressed

For each plant:

1. retain stomata that satisfy the predeclared measurement-quality rule;
2. compute the plant-level mean and median stomatal-complex area;
3. record the number of measured stomata and the excluded/review fraction.

Then compare the 16 within-variety pairs:

- mean plant-level stomatal area under watered conditions;
- mean plant-level stomatal area under drought stress;
- paired difference, drought-stressed minus watered;
- 95% confidence interval across variety pairs;
- paired t-test and Wilcoxon signed-rank sensitivity analysis;
- paired dot plot and difference plot.

Individual stomata are subsamples, not independent biological replicates.
Statistical tests must use plant-level summaries.

### 2. Comparison among all 16 varieties

Report, for every variety:

- watered plant mean and median stomatal area;
- drought-stressed plant mean and median stomatal area;
- descriptive average of the two plant means;
- within-variety treatment difference;
- measured-stomata sample size and measurement-quality coverage.

There is only one plant per variety × treatment cell in this batch. Therefore,
the 16-variety comparison is descriptive. Per-stoma rows must not be used as
replicates for an ANOVA claiming inferential variety differences.

## Thermal and plant-area integration

The combined plant table should include:

- canopy mean, median, minimum, maximum, standard deviation, P10, and P90
  temperature;
- canopy temperature minus ambient temperature;
- projected plant-mask area in pixels and fraction of the thermal frame;
- mask quality status and exclusion reason;
- thermal session, source filename, and capture time.

Projected plant area and stomatal-complex area are distinct traits and must
never share a generic `area` column.

Once stomatal traits are validated, exploratory cross-modal analyses may
include:

- canopy temperature versus mean stomatal-complex area;
- canopy temperature versus mean pore aperture;
- projected plant area versus stomatal traits;
- treatment differences in thermal and stomatal traits;
- variety-level thermal–stomatal response profiles.

With 32 plants, these relationships are exploratory and should report points,
effect sizes, and uncertainty rather than overfit multivariable models.

## Current SeedThermal findings and limitations

July 15 provides one Control and one Regular thermal image per variety. Across
all 16 pairs, the existing auto-mask extraction reports:

- mean Control temperature: approximately 25.69 °C;
- mean Regular temperature: approximately 25.84 °C;
- mean paired difference, Control minus Regular: approximately -0.15 °C;
- paired t-test: approximately `p=0.45`.

These values do not show an overall treatment-temperature difference in the
current batch.

Two auto-generated masks are invalid for plant-area analysis:

- `P02`, NC264G Regular: 51.6% of the frame selected;
- `P23`, 26-033 Control: 76.3% of the frame selected.

Most other masks select 1.4% to 3.9% of the frame. The two failed masks also
affect their extracted temperature statistics and must be corrected or
excluded before combined inference.

After excluding the two affected varieties to preserve complete pairs, the
remaining 14 auto masks provisionally show larger projected area for
drought-stressed plants. This result must remain provisional until all masks,
camera geometry, and physical area calibration are reviewed.

## Combined data products

1. `combined_study_manifest.csv`: 32 stable plant units linking both projects.
2. `stomata.csv`: per-stoma evidence and measurements.
3. `plant_phenotypes.csv`: one row per plant with thermal, plant-area, and
   stomatal summaries.
4. `variety_phenotypes.csv`: paired treatment results for each variety.
5. `treatment_comparison.csv`: overall paired treatment effects.
6. A results workbook/dashboard that reads these outputs without redefining
   treatment codes or experimental units.

## Analysis gates

Stomatal-area comparisons are not released until:

- detection correctness is calibrated on held-out tomato plants;
- complex-area measurement error is validated against expert masks or
  contours;
- the physical-scale status is explicit;
- the object-to-image-to-plant join is complete;
- thermal-mask QC is passed or exclusions are recorded; and
- every summary reports biological and object sample sizes separately.

