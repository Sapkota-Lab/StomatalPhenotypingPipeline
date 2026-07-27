# Provisional seed experiment

This run is an experiment, not a validated production model.

## Training data

- Six source-image groups
- 24 Codex-provisional boxes
- Four train groups, one validation group, and one untouched test group
- Positive 1024-pixel tiles only; unlabeled ambiguous regions were not used as
  negative examples
- StomaQuant YOLOv12 starting checkpoint
- 640-pixel training resolution, frozen backbone, 27 CPU epochs with early
  stopping

The exact split and counts are in
`../../training_data/provisional_seed_v1/split_manifest.csv` and
`../../training_data/provisional_seed_v1/summary.json`.

## Results

The best checkpoint had validation mAP50 0.0732 and mAP50-95 0.0439 on two
validation objects. Both the original checkpoint and this checkpoint had zero
recall on the five held-out pale-object tiles from `IMG_2990.JPG`.

As a standalone tiled detector, the checkpoint over-proposed badly:

| Image | Original accepted/review | Seed accepted/review |
|---|---:|---:|
| IMG_2992.JPG | 1 / 33 | 42 / 127 |
| IMG_3012.JPG | 2 / 33 | 50 / 190 |
| IMG_3018.JPG | 3 / 58 | 43 / 252 |

The checkpoint must therefore not replace the production proposal model.

Used only as a proposal source behind the independent YOLOS agreement gate, it
produced 34, 20, and 29 accepted candidates on the three untouched images.
The persistent crop sheets under
`../../validation_artifacts/provisional_seed_v1/` show that most accepted
objects have stomatal morphology, but this is a qualitative audit rather than
expert ground truth. The very large review tier and the pale-field failure
prevent promotion.

## Decision

Retain the checkpoint for research only. The experiment suggests that
domain-adapted proposals plus independent agreement can improve recall in
clear, high-contrast fields, but more complete expert annotations and
negative examples are required before this branch can be calibrated or
deployed.
