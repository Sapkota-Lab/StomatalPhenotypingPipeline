# Validation report: first microscopy batch

## Current conclusion

The pipeline runs end to end and produces inspectable coordinates, counts, and
overlays, but the current public checkpoints do not yet meet the accuracy bar
for unattended biological counting on this batch. The consensus count is a
high-precision-oriented candidate count, not ground truth.

## Plant/photo metadata

The supplied `Stomata.xlsx` contains 16 plant identifiers, each paired with one
regular and one control photo serial. All 32 listed serials exist exactly once
in `Stomata_Images`; the remaining 42 images are not assigned a plant name in
the workbook. `image_metadata.csv` preserves this distinction rather than
guessing assignments.

The regular/control images for a plant share a `specimen_id`. Future
train/validation/test construction must group on that field so paired images
cannot leak across splits.

## Batch and output integrity

- 74 input images were processed.
- 19 fungal, blank, or non-epidermal images were marked `out_of_scope`.
- 55 leaf-epidermal fields were analyzed.
- The current consensus-only output contains 323 accepted candidates:
  - 151 StomaQuant/YOLOS agreements
  - 172 StomaDet/YOLOS agreements
- 855 single-model proposals remain in the review tier.
- All 74 count rows, 74 overlays, detection rows, and source subtotals passed
  consistency checks.

The audited output is under `results_consensus/`.

## Visual audit findings

### False positives

The earlier small calibration samples were encouraging: 27/27 inspected
StomaQuant/YOLOS agreements and 13/13 newly added StomaDet/YOLOS agreements
looked like true stomata. The complete-batch crop audit exposed image styles
that those samples did not cover:

- Single-model StomaQuant boxes frequently included cell walls, blur, round
  bubbles, and debris. The 159 unconfirmed high-threshold boxes were therefore
  removed from the accepted count.
- Among the 57 high-threshold StomaQuant boxes that also had YOLOS agreement,
  several visually obvious round/debris false positives remained.
- A random 80-crop StomaDet/YOLOS sheet was much cleaner, but still contained
  several circular debris or bubble detections without the characteristic
  coffee-bean guard-cell pair and central pore.

Source-specific crop sheets are in `results_strict/audit/`.

### False negatives

- `IMG_2991.JPG` contains multiple obvious coffee-bean stomata but has zero
  accepted detections.
- `IMG_2993.JPG` is heavily blurred. Reinspection did not support the earlier
  claim that it contains obvious countable stomata; it remains a required
  quality-control adjudication case rather than a confirmed false negative.
- `IMG_3010.JPG` and `IMG_3046.JPG` contain a much smaller, cluttered
  morphology that the public checkpoints largely miss.
- Zero-count fields therefore cannot be interpreted as confirmed biological
  zeros.

## Gate for reliable counts

Before the output is used for phenotyping statistics:

1. Create representative bounding-box annotations across pale clean fields,
   dense dark fields, debris-heavy fields, and the small-morphology fields.
2. Split annotations by source image, not by crop, so the held-out test set
   measures generalization rather than tile leakage.
3. Fine-tune a detector on LCC, using an A100 for training and a V100 for
   routine inference.
4. Report object-level precision, recall, and average precision, plus
   per-image count mean absolute error and bias on the held-out images.
5. Select the operating threshold from the held-out precision/recall curve and
   preserve a review tier for uncertain detections.

Until that gate passes, `accepted_count` should be described as a conservative
automated candidate count.

## Expert-labeled external validation

A deterministic subset of 18 images was drawn from the official StomaDet test
split: six default, six imperfect, and six out-of-focus bright-field maize
images, with 765 expert bounding boxes. This subset tests implementation
correctness and source-domain behavior; it does not measure accuracy on the
current tomato images.

At confidence 0.50, the published StomaDet checkpoint achieved precision
0.9870, recall 0.9935, F1 0.9902, AP50 0.9893, and per-image count MAE 0.50.
The current production YOLOS agreement gate achieved precision 0.9886 but only
recall 0.5673 and count MAE 18.11 on the same images, demonstrating that the
gate is much too conservative outside its original ad-hoc calibration sample.

A six-view consistency experiment used the identity image, three rotations,
a horizontal flip, and grayscale CLAHE. Requiring support from at least two
views and median confidence at least 0.75 achieved:

- 759 true positives, 9 false positives, and 6 false negatives;
- precision 0.9883, recall 0.9922, F1 0.9902, and AP50 0.9895;
- per-image count MAE 0.50 and bias +0.17.

A within-image size cutoff of 0.55-1.50 times the median candidate area was
rejected: it raised precision only slightly while reducing recall to about
0.95. Area ratio remains available as an audit field but is disabled as a
counting rule by default.

## Six-image target-domain consistency audit

The externally calibrated rule was applied to pale (`IMG_2990`, `IMG_2991`),
small-morphology (`IMG_3010`, `IMG_3046`), typical (`IMG_3012`), and
debris-heavy (`IMG_3028`) fields. Accepted/review counts were:

| Image | Accepted | Review |
|---|---:|---:|
| IMG_2990.JPG | 0 | 5 |
| IMG_2991.JPG | 0 | 15 |
| IMG_3010.JPG | 0 | 46 |
| IMG_3012.JPG | 8 | 123 |
| IMG_3028.JPG | 5 | 137 |
| IMG_3046.JPG | 1 | 41 |

Visual review showed that many review-tier boxes on the clearer fields align
with coffee-shaped stomata, so the source-domain confidence threshold
undercounts this batch. Pale fields also retain artifact proposals. Because
there are no independent exhaustive labels for these six images, these numbers
are qualitative evidence only and must not be reported as accuracy.

## Pseudo-label fine-tuning experiment

A local smoke experiment fine-tuned the StomaQuant checkpoint for 12 epochs on
fixed crops generated from the consensus detections. The split was correctly
made by source image. On the pseudo-labeled validation set the best checkpoint
reported precision 0.766, recall 0.721, mAP50 0.759, and mAP50-95 0.432.

Those numbers only measure reproduction of model-generated labels. Qualitative
tests on image styles that the consensus system missed showed that the
checkpoint did not solve the biological detection problem:

- `IMG_2991.JPG`: zero accepted detections and 12 low-confidence proposals,
  mostly on cell interiors or edges rather than the obvious stomata.
- `IMG_2993.JPG`: zero accepted detections and 31 low-confidence proposals
  placed on ordinary cell regions; the image itself is too blurred to use as
  a confirmed recall example without expert quality adjudication.
- `IMG_2992.JPG`: 17 accepted detections but 607 low-confidence proposals,
  making the review output unusably noisy.

This checkpoint is retained as an experiment under
`training_runs/consensus_v1_local/`, but must not replace the current ensemble.
The result demonstrates that consensus pseudo-labels reinforce the ensemble's
blind spots. Independent representative annotations are required for the next
training run.

## Open-vocabulary detector experiment

OWLv2 was tested as an independent proposal source using both image-guided
one-shot detection and text-conditioned detection. The visual query was a
tight, confirmed stoma crop from `IMG_3012.JPG`; the target tile contained a
clear missed stoma from `IMG_2991.JPG`. Image-guided output matched large
background regions, while text prompting yielded hundreds of low-confidence
boxes and did not rank the true stoma reliably. This path is rejected for the
current engine.

The replacement annotation workflow is `box_annotator.py`. It records
full-resolution, EXIF-normalized boxes and presents the diverse priority set
from `annotation_priority_v1.csv` first.

## Provisional independent-box experiment

Six source images were provisionally annotated with 24 visually inspected
boxes. Every file is explicitly marked `provisional`; the normal dataset
exporter excludes these labels unless `--include-provisional` is supplied.
The pilot dataset kept source images in separate train, validation, and test
groups and omitted negative tiles because ambiguous regions were not
exhaustively labeled.

A restrained StomaQuant fine-tune did not pass the deployment gate:

- validation mAP50 was 0.0732 on two objects;
- the original and fine-tuned checkpoints both had zero recall on five
  held-out pale-object tiles from `IMG_2990.JPG`;
- as a standalone full-image detector, the fine-tuned checkpoint generated
  42-50 accepted proposals and 127-252 review proposals per untouched image,
  including many obvious background boxes.

The checkpoint was also tested only as a proposal source behind the
independent YOLOS agreement gate. On three untouched clear/high-contrast
images, accepted candidates increased from 20 to 34 (`IMG_2992`), 12 to 20
(`IMG_3012`), and 26 to 29 (`IMG_3018`). Visual crop sheets show that most of
these accepted objects have the expected paired guard cells and pore, but
some remain ambiguous and there is no expert ground truth for a quantitative
precision claim. The checkpoint therefore remains research-only under
`training_runs/provisional_seed_v1/`.

Persistent audit sheets and the comparison table are under
`validation_artifacts/provisional_seed_v1/`. This result supports a future
dual-proposal ensemble after expert annotation, but does not justify replacing
the current conservative output.

## Provisional seed v2 experiment

The seed dataset was expanded to 87 provisional boxes across eight source
images. Sixty-three added boxes were model-assisted candidates on two clear
fields that were visually screened for paired guard cells and a pore; they are
not independent expert labels. The source-image split contains six train
groups, one two-object validation group, and one five-object pale test group.

The best validation checkpoint reported precision 0.895, recall 0.500, mAP50
0.501, and mAP50-95 0.301. These metrics are weak evidence because validation
contains only two objects. More importantly, v2 had zero recall on the five
held-out pale stomata from `IMG_2990.JPG`.

Standalone full-image inference remained unusably noisy, generating 63-118
accepted proposals and 112-199 review proposals on three untouched images.
Behind the independent YOLOS agreement gate, however, accepted candidates
increased from 12 to 31 on `IMG_3012`, 23 to 31 on `IMG_3027`, and 9 to 26 on
`IMG_3032`. In a qualitative crop audit, 82 of 88 candidates had clear
stomatal morphology and six were ambiguous. This is not expert ground truth
and must not be treated as a precision estimate.

The v2 checkpoint therefore remains research-only under
`training_runs/provisional_seed_v2/`. Its audit sheets and comparison table
are under `validation_artifacts/provisional_seed_v2/`. The experiment
strengthens the case for a domain-adapted proposal model behind an independent
agreement gate, but it does not justify changing production counts.

## Morphology-aware keypoint benchmark

An additional independent validation set was drawn from the 2026 CVPR
AgriVision stomata-keypoint benchmark. It contains 25 dicot images with 968
expert boxes and four landmarks per stoma:

- `T-SBGH`: 13 soybean images and 524 small stomata under compounded
  object-size, taxonomic, and environment shift.
- `T-MSAA`: 12 low-contrast crab-apple images and 444 medium stomata under
  taxonomic shift.

The published YOLO26X-Pose checkpoint was evaluated without target-image
labels. For every prediction, the pipeline measured keypoint confidence,
length/width ratio, deviation from perpendicular axes, distance between axis
midpoints, and whether landmarks remained inside the predicted box.

The result rejects keypoint geometry as a counting rule:

| Evaluation | Precision | Recall | F1 | Count MAE |
|---|---:|---:|---:|---:|
| Soybean full, best raw threshold | 0.045 | 0.155 | 0.070 | 97.69 |
| Soybean full, morphology filter | 0.055 | 0.094 | 0.069 | 31.62 |
| Soybean tiled, morphology filter | 0.076 | 0.143 | 0.099 | 35.46 |
| Soybean CLAHE, best raw threshold | 0.068 | 0.111 | 0.084 | 26.46 |
| Crab apple, soybean-selected threshold | 0.243 | 0.901 | 0.383 | 100.08 |
| Crab apple, same threshold + morphology | 0.414 | 0.509 | 0.457 | 16.83 |

The crab-apple oracle threshold reached F1 0.661, but that threshold was
selected on the test split and is reported only to demonstrate severe
confidence-domain instability. It is not a deployable result. A logistic
geometry calibrator trained in each direction also failed to transfer
symmetrically.

## Tiled external-domain adaptation

Overlapping 640-pixel training crops were added to preserve the native scale
of small stomata. Each expert box belongs to one non-overlapping ownership
core, so overlapping tiles do not duplicate labels. Calibration and final
evaluation remain on untouched full images using tiled inference and global
duplicate suppression.

Neither leave-one-species-out direction passed:

| Training → test | Calibration F1 | Test P | Test R | Test F1 | Count MAE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| soybean → crab apple | 0.525 | 0.306 | 0.025 | 0.046 | 34.00 | -34.00 |
| crab apple → soybean | 0.695 | 0.139 | 0.088 | 0.107 | 16.77 | -14.77 |

A second experiment mixed both public domains while keeping source images
strictly isolated: 15 train, four calibration, and six untouched test images.
It reached calibration F1 0.442 and untouched-test F1 0.352, with precision
0.399, recall 0.316, count MAE 17.83, and bias -9.17. At the lowest exported
confidence, recall was still only 0.684 and 8,552 detections were false
positives.

These results show that tiling helps object scale but does not solve
appearance shift or discrimination from background texture. None of these
checkpoints is approved for tomato counting. The next external-label path is
a native-resolution two-stage FPN detector with explicit hard-background
learning, followed by the same frozen-threshold held-out gates.

This agrees with the paper's stage-wise diagnosis: published end-to-end
detectors collapse on the compounded soybean split even though keypoint
localization recovers when expert boxes are supplied. StomaDet also failed as a
replacement proposal source: tiled soybean F1 was 0.069 with only 0.040 recall;
on crab apple it reached precision 0.707, recall 0.358, and F1 0.475.

On the six target audit images, the pose geometry rule returned plausible
counts of 8, 13, 37, 31, 17, and 55 for `IMG_2990`, `IMG_2991`, `IMG_3010`,
`IMG_3012`, `IMG_3028`, and `IMG_3046`. Overlays show both obvious misses and
plausible crosses on non-stomatal texture, so these are not biological counts.

## No-manual-label adaptation gate

Every public-label adaptation experiment uses calibration images that are
separate from the untouched source-image test. Confidence is selected on
calibration only. The automated report requires:

- precision at least 0.90;
- recall at least 0.85;
- count MAE at most 10% of mean ground-truth count, with a three-object floor;
- absolute count bias at most 5%, with a two-object floor.

The completed one-stage tiled experiments above fail this gate. A
native-resolution Faster R-CNN/FPN is the next candidate because the current
low-threshold detector proposal ceiling is only 0.684 recall. The
implementation initializes from the published StomaDet checkpoint, uses
8–128-pixel FPN anchors and full public images, and passed a CPU
native-resolution forward smoke test with finite loss 1.6851. Its LCC job is
`cluster/lcc_stomadet_external.sbatch`; no quantitative claim is permitted
until its frozen-threshold six-image external test report passes.

## Annotation-free stability safeguard

`stomadet_tta_engine.py` now writes `stability.csv` in addition to counts and
detections. For each image it records raw counts in identity, 90/180/270-degree
rotation, horizontal-flip, and grayscale-CLAHE views, then measures:

- geometric count coefficient of variation and maximum relative deviation;
- identity-detection repeatability across geometric views;
- the fraction of geometric detection clusters supported by a majority of
  views;
- identity-versus-CLAHE detection agreement (Dice score).

`calibrate_stability_gate.py` refuses to learn stability limits unless the
untouched external accuracy report first passes the precision, recall,
count-MAE, and bias gate above. It learns the limits only from individually
accurate public calibration images and applies them unchanged to new images.
This is a secondary failure detector, not a substitute for labels: a model can
be repeatably wrong. The LCC two-stage job now runs this audit only after
external accuracy succeeds.
