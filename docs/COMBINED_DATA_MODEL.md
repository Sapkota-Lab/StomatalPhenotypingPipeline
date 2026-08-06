# Combined SeedThermal + stomatal data model

## Join contract

Records are joined only when both canonical fields match:

1. `join_key_variety`
2. `join_key_treatment`

The merge is an inner, one-to-one plant join. Duplicate composite keys,
unrecognized source labels, unmatched detection images, and unmatched manual
measurement images are fatal errors. There is no fuzzy matching, substring
matching, similarity scoring, or fallback join.

Treatments use the study definitions:

- `Regular` / `regular` → `regular` → watered, 100 mL
- `Control` / `control` → `control` → drought-stressed, 0 mL

Variety spellings are converted through the explicit source-label table in
`build_combined_study_manifest.py`. The table contains only labels observed in
the source files. Unknown spellings fail rather than being guessed.

## Canonical products

`results/integrated_exact_v1/combined_product_dataset.json` is the complete
hierarchical product. It contains one record per plant with:

- exact source identifiers and join keys;
- thermal temperature and selected-area measurements;
- plant-level stomatal detection aggregates;
- every calibrated candidate with detector confidence and calibrated
  probability;
- every quality-selected manual stomatal-complex measurement.

Flat audit-friendly companion tables are:

- `combined_plant_image_dataset.csv`
- `combined_detection_dataset.csv`
- `combined_measurement_dataset.csv`
- `exact_join_audit.csv`

## Current audit

- 32 plant/image records
- 32 unique canonical variety + treatment keys
- 5,633 calibrated detection rows
- 85 manual measurement rows
- 32 exact matches
- 0 unmatched plant, detection, or measurement records
