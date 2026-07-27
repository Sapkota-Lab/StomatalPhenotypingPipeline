# Stomata model research and selection

## Practical conclusion

The current batch contains multiple microscopy styles, including pale
bright-field epidermis, dense dark epidermis, debris-heavy fields, fungal
structures, and blank fields. No tested public checkpoint transferred with
both adequate recall and acceptable precision on every style. The production
engine therefore uses conservative cross-model agreement and records
single-model candidates for review instead of treating every model score as a
calibrated probability.

## Models retained

### StomaQuant

- Source: https://github.com/kjxlau/StomaQuant
- Architecture: YOLOv12
- Published training domain: barley, rice, and sugarcane microscopy
- Role: proposal generator
- Observed behavior: useful recall on darker fields, but unsafe as a standalone
  counter and poor recall on pale low-contrast images

### YOLOS stomata checkpoint

- Source: https://huggingface.co/DunnBC22/yolos-small-Stomata_Cells
- Dataset: https://huggingface.co/datasets/Francesco/stomata-cells
- Architecture: YOLOS-small object detector
- Role: independent verifier and review-candidate generator
- Observed behavior: higher recall than StomaQuant, with image-dependent
  pavement-cell and debris false positives

### StomaDet

- Source: https://github.com/grimmlab/StomaDet
- Published data and checkpoint:
  https://data.mendeley.com/datasets/2bfcpc6bpj/1
- Architecture: Detectron2 Faster R-CNN with ResNet-50 FPN
- Published training domain: 250 bright-field maize leaf-imprint images,
  containing more than 10,000 annotated stomata
- Role: proposal generator for the pale bright-field domain; a proposal is
  counted only when YOLOS confidence is at least 0.92 and agrees spatially
- Observed behavior: much stronger recall on bright-field images, but unsafe
  as a standalone counter on debris-heavy images

### StomaDet transformation consistency

- Six views: identity, 90/180/270-degree rotations, horizontal flip, and
  grayscale CLAHE
- Official held-out StomaDet subset: support from at least two views and median
  confidence at least 0.75 retained 99.2% recall at 98.8% precision
- Current tomato batch: the same threshold rejects many visually plausible
  stomata, showing that source-domain confidence is not target-calibrated
- Role: research candidate and agreement-feature generator, not the production
  count until representative tomato labels are available
- Size filtering: retained as a diagnostic only; the tested 0.55-1.50
  within-image median-area window removed too many valid source-domain stomata

## Approaches reviewed but not adopted

- LabelStoma: https://github.com/ancasag/labelStoma
  - Useful precedent for model-assisted correction and few-image adaptation.
  - Its older YOLO tooling is less maintainable than the current pipeline.
- StomaAI: https://github.com/xdynames/sai-app
  - Supports the supervised detection/segmentation direction, but did not
    provide a clearly superior drop-in checkpoint for this batch.
- StomaYOLO: https://github.com/yangziqi2003/StomaYOLO
  - Recent small-object work, but its SEM maize domain differs substantially
    from the present bright-field images.
- Generic contour, ellipse, circle, and contrast-enhancement methods
  - Rejected because pavement-cell walls and debris share the same local edge
    geometry. CLAHE and percentile stretching increased false positives
    without recovering trustworthy StomaQuant detections.
- Single-model YOLOS acceptance
  - Rejected after high-confidence cell-wall false positives were observed.
- Single-model StomaDet acceptance
  - Rejected after high-confidence false positives appeared on IMG_3028.
- Small provisional-box fine-tune as a standalone detector
  - Trained from 24 inspected boxes across six independent source images.
  - Failed on the held-out pale field and produced an unusably large proposal
    tier on untouched full images.
  - Retained only as research evidence: when YOLOS independently verified its
    proposals, most audited accepted crops in three clear fields had stomatal
    morphology and candidate recall increased.
- Expanded provisional seed v2 as a production model
  - Trained from 87 provisional boxes across eight source images, including
    model-assisted boxes that were visually screened but not expert-verified.
  - Increased YOLOS-gated clear-field candidates from 44 to 88 across three
    untouched images, with 82 visually clear and six ambiguous crops.
  - Still produced hundreds of standalone proposals and had zero recall on
    five held-out pale stomata, so it remains research-only.
- OWLv2 open-vocabulary and image-guided detection
  - Tested with both a tight true-stoma reference crop and three microscopy
    text prompts on a tile containing a clear missed stoma from `IMG_2991`.
  - Image-guided mode produced large background-region matches; text mode
    produced hundreds of low-score proposals, with the true object not
    separable by score.
  - Rejected as an ensemble component because it adds proposal volume without
    a reliable acceptance signal.
- Arabidopsis YOLOX detector and pore U-Net
  - The published bright-field Arabidopsis detector produced cluster-sized and
    texture-sized boxes rather than stable individual-stoma boxes on the target
    images. Its checkpoint and generated overlays were removed after recording
    the audit because local storage is constrained.
  - The pore U-Net defined a high-precision tier on its source domain but did
    not transfer reliably to pale target images, where debris could also
    receive high pore scores.
- Four-keypoint YOLO26X-Pose
  - Source:
    https://huggingface.co/Sainath001/stomata-keypoint-benchmark-cvpr-agrivision-2026-models
  - Benchmark:
    https://huggingface.co/datasets/Sainath001/stomata-keypoint-benchmark-cvpr-agrivision-2026-Dataset
  - The four landmarks encode two polar length tips and two lateral width
    endpoints. Derived checks require approximately perpendicular axes, nearly
    coincident axis midpoints, and plausible length/width ratio.
  - On 524 expert soybean boxes under compounded small-object, dicot, and
    greenhouse shift, full-image best F1 was 0.070; morphology filtering did
    not improve it. Tiling raised morphology-filtered best F1 only to 0.099,
    and luminance CLAHE reached raw best F1 0.084.
  - On 444 crab-apple boxes, an oracle threshold selected on that same split
    reached F1 0.661, but the soybean-selected threshold transferred at only
    F1 0.383 raw or 0.457 with morphology filtering. A learned geometry
    calibrator was also directionally unstable.
  - Target overlays show missed obvious stomata and plausible-looking crosses
    on pavement-cell texture. Keypoint geometry is therefore an audit feature,
    not an acceptance rule.

## External-domain adaptation experiment

The strongest no-manual-label path is now supervised adaptation using public
expert labels with a strict leave-one-species-out test:

- `T-SBGH`: 13 soybean images, 524 small stomata under compounded shift.
- `T-MSAA`: 12 low-contrast crab-apple images, 444 stomata.
- Train/validation boxes come from one species only.
- The other species remains untouched until the final test.
- The operating confidence is selected on the validation images and then
  applied unchanged to the held-out species.

`build_external_domain_dataset.py` creates both directions using overlapping
640-pixel crops with non-overlapping annotation-ownership cores. Confidence is
selected on full validation images using tiled inference and then frozen for
the untouched full-image test. The local results reject both directions:

| Training → untouched test | Calibration F1 | Test precision | Test recall | Test F1 | Count MAE |
|---|---:|---:|---:|---:|---:|
| soybean → crab apple | 0.525 | 0.306 | 0.025 | 0.046 | 34.00 |
| crab apple → soybean | 0.695 | 0.139 | 0.088 | 0.107 | 16.77 |

Tiling fixed the worst small-object resolution failure, but confidence and
appearance did not transfer between species. A mixed-domain experiment then
used 15 source images for fitting, four different images for calibration, and
six untouched images for testing, with both domains represented in every
split. It improved untouched-test F1 to 0.352, but precision 0.399, recall
0.316, count MAE 17.83, and bias -9.17 still fail every deployment gate.
At confidence 0.001, proposal recall was only 0.684 while producing 8,552
false positives, so a proposal classifier alone cannot recover the required
recall.

The next model family should therefore replace the one-stage proposal
bottleneck rather than tune its confidence. A native-resolution Faster R-CNN
with FPN is the leading candidate because it supplies multi-scale proposals
and an explicit background class, matching the successful StomaDet design.
Target-image transformation consistency remains useful only after an external
held-out accuracy gate passes; consistency cannot establish correctness by
itself.

The transformation check is now implemented as a separate, auditable gate.
`stomadet_tta_engine.py` exports per-image raw view counts, geometric count
variation, identity repeatability, majority-view cluster support, and
identity/CLAHE agreement. `calibrate_stability_gate.py` first verifies that the
untouched public accuracy gate passed, then estimates allowable stability
ranges from only the individually accurate public calibration images. Those
ranges can be applied unchanged to the tomato images after the LCC model
qualifies. A stability pass alone never authorizes a biological count.

That two-stage experiment is implemented in
`train_stomadet_external.py` and `cluster/lcc_stomadet_external.sbatch`.
It initializes from the published 315 MB StomaDet checkpoint, preserves the
learned RPN/ROI heads, changes p2–p6 anchor sizes to 8–128 pixels, and adds
strong luminance/contrast plus flip augmentation. A CPU construction and
native-resolution forward smoke test loaded all pretrained parameters,
registered 15 full training images with 555 boxes, retained 30 boxes in the
first augmented batch, and produced finite loss 1.6851. Quantitative
validation still requires the LCC CUDA run.

`cluster/lcc_external_domain_adaptation.sbatch` reproduces the tiled
leave-one-species-out experiment on LCC. A direction passes only with held-out
precision at least 0.90, recall at least 0.85, per-image count MAE at most 10%
of mean ground-truth count (or three), and absolute bias at most 5% (or two).

## Current decision rule

1. Count a StomaQuant proposal only when YOLOS confidence is at least 0.85 and
   the boxes agree spatially.
2. Add a StomaDet proposal that StomaQuant missed only when YOLOS confidence is
   at least 0.92 and the boxes agree spatially.
3. Keep every unmatched single-model proposal as a review candidate.
4. Use the scope manifest to skip fungal, blank, and non-epidermal fields.

The thresholds are empirical for this first batch. They must be recalibrated
against expert annotations before the counts are treated as biological ground
truth.
