# Stomatal Analysis Engine

## Scientific objective

The primary output is one traceable record for every retained stomatal
candidate. A final count is a derived image summary, not the organizing
principle of the pipeline.

The engine should:

1. detect stomata at high recall while retaining difficult candidates;
2. estimate the probability that each candidate is a real stoma;
3. preserve the evidence behind that probability;
4. measure stomatal-complex, pore, and spatial traits when the image supports
   them;
5. link every object to its image, plant, treatment, model run, and review
   history; and
6. distinguish direct measurements from calibrated or model-derived estimates.

This follows published phenotyping workflows that separate stomatal detection
from pore segmentation and report pore area, width, length, and opening state
per object. Spatial position is also biologically useful: nearest-neighbor and
aggregation/evenness measures can describe patterning that density alone
misses.

## Core pipeline

```text
image ingestion and provenance
        ↓
quality and scale assessment
        ↓
high-recall tiled proposal detection
        ↓
cross-tile and cross-model object fusion
        ↓
per-object morphology / pose / pore analysis
        ↓
calibrated correctness probability and review tier
        ↓
spatial-neighborhood analysis
        ↓
object table + image table + run manifest + visual audit
```

No stage should discard a candidate merely because it falls below the
high-confidence threshold. Candidates above a low proposal floor remain in
the object table with a review or rejected status.

## Data products

The canonical export should contain three normalized tables and a run
manifest. CSV is the human-readable interchange format; Parquet should be the
canonical typed format once the schema stabilizes.

### 1. `runs`

One record per analysis execution:

| Field | Meaning |
|---|---|
| `run_id` | Stable UUID for the execution |
| `pipeline_version` | Git commit or release |
| `started_at_utc` | Execution timestamp |
| `model_ids` | Names, versions, and checkpoint hashes |
| `configuration_hash` | Hash of inference and measurement settings |
| `calibration_id` | Confidence-calibration artifact, if available |
| `device` | CPU, CUDA, or MPS execution device |
| `input_manifest_hash` | Hash of the analyzed image manifest |

The complete configuration, package versions, model checksums, and command
line should also be saved as `run_manifest.json`.

### 2. `images`

One record per microscope field:

| Category | Fields |
|---|---|
| Identity | `image_id`, `filename`, `image_sha256`, `photo_serial` |
| Biological metadata | `specimen_id`, `plant_name`, `treatment`, `leaf_id`, `leaf_surface`, `field_id` |
| Pixel geometry | `width_px`, `height_px`, `analyzed_area_px2`, `roi_coverage_fraction` |
| Physical calibration | `micrometers_per_pixel`, `analyzed_area_mm2`, `scale_source`, `scale_validated` |
| Acquisition metadata | `microscope`, `objective`, `magnification`, `camera`, `capture_time`, `operator` |
| Quality | `focus_score`, `contrast_score`, `clipped_fraction`, `illumination_uniformity`, `quality_stratum`, `quality_flags` |
| Scope | `analysis_scope`, `exclusion_reason` |
| Shared experiment | `genotype_id`, `plant_unit_id`, `water_status`, `water_ml`, `thermal_session_id`, `thermal_source_file` |

Image summaries may include detected, accepted, review, and rejected candidate
totals, but they must be generated from the object table.

### 3. `stomata`

One record per fused candidate:

#### Identity and localization

- `stoma_id`: stable within a run, such as
  `{run_id}:{image_id}:{detection_id}`
- `run_id`, `image_id`, `detection_id`
- `center_x_px`, `center_y_px`
- `bbox_x1_px`, `bbox_y1_px`, `bbox_x2_px`, `bbox_y2_px`
- `oriented_center_x_px`, `oriented_center_y_px`,
  `oriented_length_px`, `oriented_width_px`, `orientation_deg`
- `mask_path` or COCO RLE when an instance mask is available
- `crop_path` for visual audit
- `touches_image_edge`, `edge_censored`

#### Detection evidence

Raw model scores must not be labeled as probabilities.

- `detector_score_<model>`
- `detector_support_<model>`
- `model_agreement_count`
- `view_support`
- `tta_score_min`, `tta_score_median`, `tta_score_max`
- `tta_box_center_sd_px`, `tta_area_cv`
- `morphology_score`
- `pore_score`
- `out_of_distribution_score`
- `p_stoma`: held-out, target-calibrated probability that the fused candidate
  is a true stoma
- `p_stoma_lower`, `p_stoma_upper`: optional calibrated uncertainty interval
- `calibration_id`
- `decision_status`: `accepted`, `review`, or `rejected`
- `decision_reasons`: machine-readable list of supporting or warning flags

The component scores remain visible even after a combined probability is
calculated. This makes false positives and calibration drift diagnosable.

#### Stomatal-complex morphology

Direct pixel-space traits:

- `complex_length_px`
- `complex_width_px`
- `complex_area_px2`
- `complex_perimeter_px`
- `complex_aspect_ratio`
- `complex_eccentricity`
- `complex_solidity`
- `orientation_deg`
- keypoint coordinates and keypoint confidence, when available
- `complex_segmentation_quality`

Physical-unit versions (`*_um`, `*_um2`) are populated only when
`scale_validated=true`.

Bounding boxes support only coarse length, width, area, and orientation.
Perimeter, solidity, eccentricity, and precise area require a validated
instance mask or equivalent contour.

#### Pore and aperture traits

- `pore_detected`
- `pore_state`: `open`, `closed`, or `indeterminate`
- `pore_length_px`, `pore_width_px`, `pore_area_px2`
- `pore_aspect_ratio`, `pore_eccentricity`, `pore_solidity`
- `pore_centroid_offset_norm`
- `aperture_width_length_ratio`
- `pore_complex_area_ratio`
- `pore_segmentation_quality`

Physical-unit versions require a validated scale. Open/closed state must have
an explicit indeterminate class; absence of a segmented pore is not
automatically equivalent to a biologically closed stoma.

#### Local image context

- `local_focus_score`
- `local_contrast_score`
- `local_brightness_mean`
- `local_brightness_sd`
- `local_edge_density`
- `local_saturation_fraction`
- `occlusion_score`
- `debris_or_boundary_warning`

These fields help stratify failure modes and distinguish biological variation
from acquisition quality.

#### Spatial context

- `nearest_neighbor_center_distance_px`
- `nearest_neighbor_edge_distance_px`
- `nearest_neighbor_bearing_deg`
- `neighbors_within_radius`
- `local_density_px2`
- `local_orientation_alignment_deg`
- `distance_to_image_edge_px`
- `spatial_edge_censored`

Physical distances and densities are emitted only with a validated
micrometers-per-pixel value and a defined analyzed region. Image-edge
correction is required for formal point-pattern statistics.

#### Human review

- `review_status`
- `reviewer`
- `reviewed_at_utc`
- `corrected_class`
- corrected box, keypoints, or mask references
- `review_notes`

Machine predictions are never overwritten by review. Corrections are stored
as separate fields so the original evidence remains reproducible.

## Scientifically useful image- and specimen-level summaries

Summaries should report distributions, not only means:

- number of candidates by decision tier;
- expected number of true stomata, `sum(p_stoma)`, only when probabilities
  are calibrated;
- stomatal density per square millimeter, when analyzed area is calibrated;
- median, interquartile range, and robust dispersion of complex length,
  width, area, aspect ratio, orientation, and pore traits;
- fraction open, closed, and indeterminate;
- mean and distribution of nearest-neighbor distance;
- spatial aggregation/evenness statistics with edge correction;
- proportion of measurements excluded for low quality or edge censoring; and
- summaries grouped first by field, then plant, then treatment so fields are
  not treated as independent biological replicates.

The required study-level summaries are:

- comparison of plant-level mean stomatal-complex area between watered and
  drought-stressed plants, paired by variety; and
- descriptive comparison of plant-level mean stomatal-complex area among all
  16 varieties, shown separately for both treatments.

`Regular` means watered (100 mL); `Control` means not watered /
drought-stressed (0 mL). The combined experimental design and thermal join are
defined in [`COMBINED_STUDY.md`](COMBINED_STUDY.md).

## Quantities that require additional information

The engine must leave these fields null rather than infer unsupported values:

| Quantity | Requirement |
|---|---|
| Micrometer dimensions | Stage micrometer or trustworthy acquisition calibration |
| Stomatal density (`mm^-2`) | Valid scale and known analyzed leaf area |
| Stomatal index | Both stomata and epidermal-cell counts/segmentation |
| Guard-cell dimensions | Guard-cell instance masks or validated keypoints |
| Pore aperture | Validated pore segmentation at sufficient focus/resolution |
| Maximum conductance | Density, pore geometry, pore-depth assumption, physical units, and an explicit physiological model |

Conductance estimates should be a downstream optional analysis, never a
default detector output.

## Confidence semantics

`p_stoma` answers one question: “Given this candidate and this acquisition
domain, what is the calibrated probability that an expert would label it as a
stoma under the annotation protocol?”

It should combine, without hiding:

- detector logits/scores;
- cross-model agreement;
- TTA repeatability and box stability;
- morphology evidence;
- pore evidence;
- local quality; and
- domain-shift or out-of-distribution evidence.

Calibration must use target-domain annotations from plants excluded from model
training. Isotonic regression, Platt scaling, or another held-out calibration
method can be compared. Reliability curves, Brier score, expected calibration
error, and precision-recall curves are required. Thresholds are views over
`p_stoma`, not destructive filters:

- `accepted`: precision-controlled high-confidence detections;
- `review`: plausible detections whose uncertainty matters;
- `rejected`: retained low-probability proposals used for audit and future
  hard-negative learning.

## Validation

### Detection

- recall, precision, F1, AP50, and AP75;
- free-response ROC or recall versus false positives per image;
- duplicate rate and localization error;
- results stratified by plant, treatment, image quality, object size,
  orientation, contrast, and edge status.

Count MAE remains a secondary image-level diagnostic.

### Confidence

- reliability diagram;
- Brier score;
- expected calibration error;
- coverage and error rate for each decision tier;
- calibration drift by plant and acquisition condition.

### Measurement

- keypoint distance error;
- mask Dice and IoU;
- absolute and relative error for length, width, area, orientation, and pore
  aperture;
- Bland-Altman analysis and repeatability across independent reviewers;
- explicit failure rate and indeterminate rate.

Training, confidence calibration, and final testing must use disjoint plants,
not merely different images from the same plant.

## Implementation phases

### Phase 1 — Unified object ledger

Retain the existing proposal engines but replace count-first CSVs with one
fused object table. Add stable IDs, provenance, raw model evidence, TTA
stability, bounding-box geometry, image quality, plant metadata, crop paths,
and review states. Counts become generated summaries.

### Phase 2 — Target confidence calibration

Use a small tomato reference set split by specimen. Train or adapt the
detector only on training plants; fit confidence calibration on separate
plants; report final performance on untouched plants. Preserve all candidates
above a low proposal floor.

### Phase 3 — Validated object morphology

Annotate rotated axes/keypoints or instance masks for a representative subset.
Evaluate whether the existing pose model can be retrained on tomato; otherwise
use an oriented detector plus instance/pore segmentation. Do not release
fine-grained traits until their measurement errors are quantified.

The current implementation uses expert boxes as prompts to SAM 2.1 Tiny and
converts the prompted raster result to a filled outer convex contour. A
balanced 251-object queue spans all 32 plants and both treatments. Validation
annotations are drawn with SAM proposals hidden to prevent confirmation bias.
The automated method is
not released unless `validate_complex_masks.py` passes all predeclared
coverage, mask-overlap, area-error, and area-bias gates. The proposal-model
SHA-256 in the validation report must match the masks applied to the ledger.

### Phase 4 — Physical calibration

Record microscope/objective/camera settings and acquire a stage-micrometer
image for each acquisition configuration. Populate micrometers-per-pixel and
then enable physical dimensions and density.

### Phase 5 — Spatial and physiological analyses

Add edge-corrected nearest-neighbor and arrangement statistics. Add stomatal
index only after epidermal-cell segmentation. Add conductance-related
estimates only as an explicitly parameterized downstream model.

## Existing components to reuse

- `stomata_engine.py`: tiled high-recall proposals and image-quality metrics
- `ensemble_stomata_engine.py`: cross-model spatial agreement
- `stomadet_tta_engine.py`: view support and confidence/area stability
- `stomata_pose_engine.py`: provisional keypoint geometry
- `morphology_verifier.py`: morphology evidence
- `pore_verifier.py`: provisional pore evidence
- `import_stomata_metadata.py`: image-to-plant/treatment provenance
- `box_annotator.py`: expert correction and review
- `/Users/lenquanhone/SapkotaLab/SeedThermal`: thermal distributions,
  temperature relative to ambient, projected plant area, and thermal-mask
  provenance

These components provide evidence. None of their current heuristic scores
should be presented as calibrated biological probabilities until they pass
target-domain validation.

## Selected scientific basis

- [Image-Based Quantification of Arabidopsis thaliana Stomatal Aperture from
  Leaf Images](https://academic.oup.com/pcp/article/64/11/1301/7082629)
- [Stomata in-sight: Integrating live confocal microscopy with leaf gas
  exchange and environmental control](https://academic.oup.com/plphys/article/199/4/kiaf600/8325470)
- [An Inverse Scaling Relationship between Stomatal Density and Mean Nearest
  Neighbor Distance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10650524/)
- [Size matters: point pattern analysis biases the estimation of spatial
  properties of stomata distribution](https://doi.org/10.1111/nph.14305)
- [Cytokinin activity increases stomatal density and transpiration rate in
  tomato](https://academic.oup.com/jxb/article/67/22/6351/2411375)
