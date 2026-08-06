# Stomatal Phenotyping Pipeline

Automated per-stoma detection, confidence assessment, morphometry, and
phenotyping in high-resolution microscope images.

## Project direction

The project is now organized as a scientific analysis engine rather than a
counter. Its primary output will be a traceable record for every stomatal
candidate: location, calibrated correctness probability, model evidence,
quality flags, morphology and pore measurements, spatial context, specimen
metadata, and review history. Per-image counts are derived summaries.

The target architecture, data schema, measurement boundaries, and validation
requirements are defined in
[`docs/ANALYSIS_ENGINE.md`](docs/ANALYSIS_ENGINE.md).

SeedThermal and this repository are two modalities of one tomato phenotyping
study. The shared experimental units, treatment semantics, requested
stomatal-area comparisons, thermal join, and replication limits are defined in
[`docs/COMBINED_STUDY.md`](docs/COMBINED_STUDY.md). In all biological outputs,
`Regular` means watered (100 mL) and `Control` means not watered /
drought-stressed (0 mL).

## Current detection baseline

`ensemble_stomata_engine.py` combines three stomata-specific detectors:

- StomaQuant YOLOv12 supplies conservative proposals.
- A fine-tuned YOLOS transformer independently verifies lower-confidence
  proposals.
- StomaDet Faster R-CNN supplies proposals from a closer bright-field
  leaf-imprint domain, but contributes a count only when YOLOS independently
  agrees.

No detector may add a count alone. A StomaQuant proposal is counted only when
high-confidence YOLOS agrees spatially. YOLOS and StomaDet may also jointly add
a detection that StomaQuant missed. Unmatched single-model detections remain
review candidates because batch-wide auditing found image-dependent cell-wall,
debris, and bubble false positives.

StomaQuant and YOLOS use overlapping tiled inference to preserve small stomata
in the high-resolution source images; StomaDet evaluates the resized full
field. EXIF orientation is normalized so all models and saved overlays use the
same coordinate system.

The StomaQuant checkpoint comes from the open-source
[StomaQuant](https://github.com/kjxlau/StomaQuant) project and was trained on
manually annotated barley, rice, and sugarcane microscopy. The second checkpoint
is [DunnBC22/yolos-small-Stomata_Cells](https://huggingface.co/DunnBC22/yolos-small-Stomata_Cells).
The third is the published
[StomaDet](https://github.com/grimmlab/StomaDet) Faster R-CNN trained on more
than 10,000 annotated stomata from bright-field maize leaf imprints.
The approach follows the supervised object-detection direction used by
[StomaDet](https://github.com/grimmlab/StomaDet),
[StomaAI](https://github.com/xdynames/sai-app), and
[LabelStoma](https://github.com/ancasag/labelStoma), rather than generic contour
or circle detection.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install --no-build-isolation -r requirements.txt
.venv/bin/python download_model.py
```

## Run the current detection baseline

```bash
.venv/bin/python ensemble_stomata_engine.py \
  --input Stomata_Images \
  --output results \
  --device auto
```

`auto` selects CUDA, then Apple Metal, then CPU. CPU works but transformer
inference is substantially slower. These current count-oriented outputs are
transitional and will be replaced by the unified object ledger:

- `results/counts.csv`: two-model consensus counts, review counts, analysis
  status, and acquisition-quality diagnostics
- `results/detections.csv`: source, status, model confidences, and coordinates
  for every retained detection
- `results/overlays/`: green StomaQuant/YOLOS accepted boxes, magenta
  StomaDet/YOLOS accepted boxes, amber single-model StomaQuant review
  candidates, and blue YOLOS-only review candidates

`image_manifest.csv` records the contact-sheet scope review for the current
mixed microscopy batch. Images without a leaf epidermal field (fungal
structures, conidia, or blank fields) are written as `out_of_scope` with no
count instead of being passed through a stomata detector. New images absent
from the manifest are processed and marked `counted_unreviewed`.

The current batch also has an explicit user-defined scope decision: 42 files
are marked `out_of_scope` with an `extra_sample` note, and the remaining 32
files are the proper-analysis set. Do not interpret diagnostic counts from the
extra samples as part of the biological analysis.

`image_metadata.csv` links the supplied photo serials to plant names and
regular/control treatments. It contains all 74 image filenames: 32 are mapped
to 16 paired plants and 42 are explicitly marked `not_in_workbook`. Rebuild it
from the original, read-only workbook with:

```bash
.venv/bin/python import_stomata_metadata.py \
  --workbook /Users/lenquanhone/Downloads/Stomata.xlsx
```

Use `image_metadata.csv` as the metadata input to the annotation dataset
builder. The shared `specimen_id` keeps each regular/control pair in one data
split and prevents plant-level leakage.

Build the shared 32-plant manifest from the stomatal metadata and SeedThermal's
July 15 measurements with:

```bash
.venv/bin/python build_combined_study_manifest.py
```

After the detector produces an expert-validated per-stoma ledger with measured
stomatal-complex areas, generate the requested treatment and 16-variety area
comparisons with:

```bash
.venv/bin/python analyze_stomatal_area.py \
  --stomata path/to/validated_stomata.csv \
  --output results/combined_stomatal_area
```

The analysis first summarizes measurements within each plant. It then performs
the watered-versus-drought comparison across the 16 matched variety pairs.
The 16-variety output is descriptive because this batch contains only one
plant per variety × treatment cell. Do not substitute bounding-box area for a
validated stomatal-complex mask area.

Long runs checkpoint both CSV files after every image and resume by default.
Use `--no-resume` to deliberately recompute a compatible output directory.

For a faster single-model baseline, run `stomata_engine.py`. For UKY cluster
execution, see `cluster/README.md` and `cluster/lcc_ensemble.sbatch`.

## Create reference annotations

The earlier color-flood development selector is not suitable for biological
ground truth because cell interiors and pavement-cell boundaries often share
colors with stomata. Use the zoomable bounding-box annotator instead:

```bash
.venv/bin/python box_annotator.py
```

It opens `http://127.0.0.1:8765`, loads the morphology-diverse priority queue
first, and autosaves tight stomatal-complex boxes under `box_annotations/`.
Mouse-wheel or `+`/`-` zoom changes the microscope image itself; Space-drag or
middle-drag pans. Draw one box around the paired guard cells and central pore,
not around a pore alone or a whole pavement cell. Coordinates are stored
against EXIF-normalized full-resolution images.

Once completed annotations have specimen IDs, create deterministic 1024-pixel
training tiles with:

```bash
.venv/bin/python build_annotated_dataset.py
```

The builder keeps each specimen in exactly one train/validation/test split,
adds background-negative tiles, and writes both `dataset.yaml` and a split
manifest under `training_data/annotated_v1/`.

The completed 32-image tomato annotation dataset, split composition, treatment
coverage, and remaining full-field QC warnings are documented in
[`docs/ANNOTATION_DATASET.md`](docs/ANNOTATION_DATASET.md).

Build the first conservative per-stoma ledger from the box annotations with:

```bash
.venv/bin/python build_stomata_ledger.py
```

This writes `results/combined/stomata.csv`. It includes stable plant linkage,
box localization, coarse box geometry, local image-quality measurements, and
nearest-neighbor measurements. Provisional or in-progress annotations remain
in the `review` tier. Precise stomatal-complex and pore fields remain empty
until validated masks exist; `bbox_area_px2` must not be analyzed as
`complex_area_px2`.

Create a randomized crop audit of ledger objects with:

```bash
.venv/bin/python audit_detection_crops.py \
  --detections results/combined/stomata.csv \
  --images Stomata_Images \
  --output results/combined/visual_audit/review_crops_100.jpg \
  --status review
```

The calibrated tomato production run is submitted with:

```bash
sbatch cluster/lcc_production_inference.sbatch
```

It analyzes exactly the 32 mapped plant images and writes raw candidates,
calibrated detections, and the joined object ledger under
`validation_artifacts/tomato_production_v1/`. The first completed run
(`5380067`) retained 5,633 low-floor candidates: 325 `accepted`, 1,495
`review`, and 3,813 `rejected`. Rejected proposals remain auditable but do not
contribute to spatial measurements.

Two mask-review designs are available. For a reduced descriptive analysis,
trace the three clearest expert-boxed objects in every image with:

```bash
.venv/bin/python complex_mask_annotator.py \
  --reviewer YOUR_NAME \
  --selection-mode quality \
  --max-per-image 3 \
  --queue-manifest validation_artifacts/complex_mask_quality_sample_v1.json
```

This current batch contains 96 selected objects: three from each of 32 plants,
split evenly between watered and drought-stressed plants. Objects are ranked
within each image using local focus, contrast, pixel clipping, and edge
containment. Selection does not use treatment, variety, object area, mask
area, or apparent stomatal shape. This design supports rough, descriptive
per-image means; it is not an unbiased estimate of all stomata in an image,
and three objects per plant do not support strong within-plant distributional
claims.

After review, build a balanced table from the clearest maximum of three
completed masks per image and run the area comparisons with:

```bash
.venv/bin/python build_quality_mask_measurements.py
.venv/bin/python analyze_stomatal_area.py \
  --stomata results/quality_mask_analysis_v1/stomatal_complex_measurements.csv \
  --output results/quality_mask_analysis_v1/area_comparisons \
  --area-field complex_area_px2 \
  --area-unit px2 \
  --status-field decision_status \
  --accepted-status accepted \
  --quality-field measurement_quality_passed
```

For independent validation of the automated segmentation method instead,
trace the larger balanced random subset with:

```bash
.venv/bin/python complex_mask_annotator.py \
  --reviewer YOUR_NAME \
  --selection-mode random \
  --max-per-image 8
```

Open `http://127.0.0.1:8765`. The random validation queue contains up to eight
deterministically sampled objects per plant (251 total in the current batch).
Validation mode deliberately hides the SAM proposal to prevent confirmation
bias. Trace the outer boundary of the entire stomatal complex; do not trace
only the pore. These independent manual polygons validate the automated
segmentation method; they are not detector-training boxes. The queue is
interleaved by plant, so reviewing the first 128 objects provides four masks
from every plant; complete more whenever a plant contains difficult or
heterogeneous examples.

`--show-proposals` enables a separate assisted-correction mode, but masks saved
in that mode must not be used as independent validation references.

Validate completed contours with:

```bash
.venv/bin/python validate_complex_masks.py --require-gate
```

The gate requires coverage of all 32 plants and both treatments, median mask
IoU at least 0.75, median Dice at least 0.85, median absolute area error at
most 10%, 90th-percentile absolute area error at most 25%, and absolute mean
area bias at most 5%. `apply_validated_complex_masks.py` refuses to populate
`complex_area_px2` unless the exact proposal model hash passes this gate.

After manual mask review is complete and the all-object proposal directory has
been downloaded from LCC, run the complete guarded workflow with:

```bash
.venv/bin/python finalize_combined_analysis.py
```

This command validates the mask method, rebuilds the expert object ledger,
applies only hash-matched validated masks, runs the paired plant-level area
analysis, creates the three requested figures, and writes
`plant_phenotypes.csv`, `variety_phenotypes.csv`, and
`treatment_comparison.csv` under `results/combined_final/`. It exits before
writing area results if any measurement gate fails.

## Validation status

The tomato-specific model is now a validated proposal engine, not a final
biological counter. On untouched full images, the broad review tier achieved
0.934 recall and 0.538 precision. The conservative accepted tier achieved
0.897 precision and 0.330 recall. The broad tier is therefore appropriate for
human review and downstream mask proposals, while the accepted tier is a
precision-controlled subset. Neither tier alone should be interpreted as a
complete biological count.

The earlier calibration checks (27/27 StomaQuant/YOLOS and 13/13
StomaDet/YOLOS crops) established that cross-model agreement improves
precision, but the larger audit showed that those small samples did not cover
the full image diversity. A new expert-labeled StomaDet test-subset benchmark
also confirms that the public checkpoint is excellent on its maize source
domain, but a six-image target audit shows that its confidence calibration
does not transfer to this tomato batch. See `VALIDATION_REPORT.md` for the
current evidence and required fine-tuning gate.

The morphology-aware YOLO26X-Pose checkpoint was also rejected after evaluation
on 968 expert-labeled dicot stomata: axis geometry could not reliably separate
small stomata from textured lookalikes, and confidence thresholds did not
transfer between soybean and crab apple. Tiled leave-one-species-out
StomaQuant adaptation also failed: its best untouched-species F1 was 0.107.
A mixed-domain, source-image-isolated test improved F1 to 0.352 but still had
precision 0.399, recall 0.316, and count MAE 17.83. These checkpoints remain
rejected. The next no-target-annotation experiment replaces the one-stage
proposal bottleneck with a native-resolution Faster R-CNN/FPN trained only on
public expert boxes. Its dataset/configuration and one native-resolution
training forward pass have been verified locally; the quantitative CUDA run is
`cluster/lcc_stomadet_external.sbatch`. See `MODEL_RESEARCH.md` and
`VALIDATION_REPORT.md`.

If that external accuracy gate passes, the same cluster job runs an
annotation-free six-view stability audit and calibrates its limits on accurate
public validation images. Rotation/flip/contrast consistency can flag brittle
target predictions, but cannot prove that a consistent prediction is a
stoma; both gates are required before tomato counts advance.

The counting definition and edge rules are recorded in `ANNOTATION_PROTOCOL.md`.
The model survey and decisions are recorded in `MODEL_RESEARCH.md`.
Use `audit_detection_crops.py` to generate source-specific crop sheets from any
completed `detections.csv`.
