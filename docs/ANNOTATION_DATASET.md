# Tomato Stomata Annotation Dataset

## Dataset status

The 32 in-scope tomato microscope images have one expert box file each. The
completed dataset contains:

- 1,003 stomatal-complex boxes;
- 610 boxes from watered plants (`Regular`, 100 mL);
- 393 boxes from drought-stressed plants (`Control`, 0 mL);
- 16 varieties;
- one watered and one drought-stressed plant image per variety.

Older annotations for files classified as `extra_sample` remain outside this
dataset and are excluded through `image_metadata.csv`.

## Leakage-safe split

`build_annotated_dataset.py` assigns whole varieties—not tiles or images—to a
single split. Both treatment images for one variety therefore remain together.
The deterministic `annotated_v1` split contains:

| Split | Varieties | Source images | Expert boxes | Tiles |
|---|---:|---:|---:|---:|
| Train | 10 | 20 | 541 | 717 |
| Validation | 3 | 6 | 250 | 293 |
| Test | 3 | 6 | 212 | 264 |

The generated tiles are reproducible build artifacts and are not committed.
They are rebuilt locally or on LCC from source images, annotations, and
metadata.

## Interpretation

These annotations support stomata detection and detector-score calibration.
They do not provide precise stomatal-complex area, perimeter, guard-cell area,
or pore area because the labels are axis-aligned boxes rather than instance
masks.

`bbox_area_px2` is a coarse localization feature and must not be substituted
for `complex_area_px2`.

## Quality-control findings

The full-field overlay audit and randomized crop audits show the intended
coffee-shaped stomatal morphology across both treatments. Some fields are
blurred, obstructed, or sparse. Several matched pairs have large differences
in box count, including:

- 25-014: 77 watered versus 8 drought-stressed;
- 26-016: 27 watered versus 4 drought-stressed;
- 25-013: 30 watered versus 8 drought-stressed.

These differences may reflect field composition or acquisition quality, but
they should receive a second full-field completeness review before counts are
treated as biological results. Detector training can proceed, provided the
test split remains untouched and all performance claims are reported
separately from the later area-segmentation task.

## Next validation stages

Fine-tuning completed on UKY LCC. On the three untouched test varieties, the
tomato-specific detector achieved tile-level precision 0.814, recall 0.812,
mAP50 0.882, and mAP50–95 0.415. These figures cover 264 test tiles. Tiling
produced 788 evaluated instances from 212 unique source-image boxes, so the
figures describe object detection on tiles rather than independent biological
counts.

Full-image score calibration and duplicate fusion are now complete. The frozen
test split produced:

- review floor: precision 0.538, recall 0.934;
- maximum-validation-F1 threshold: precision 0.708, recall 0.844;
- precision-controlled accepted threshold: precision 0.897, recall 0.330.

The all-plant production ledger contains 325 accepted, 1,495 review, and 3,813
rejected low-floor candidates. The remaining stages are:

1. Trace the balanced 251-object complex-mask validation queue.
2. Generate automated box-guided complex masks.
3. Measure mask Dice, IoU, and area error on the manual polygons.
4. Populate `complex_area_px2` only after the measurement-quality gate passes.
5. Run paired treatment and descriptive variety area comparisons.
