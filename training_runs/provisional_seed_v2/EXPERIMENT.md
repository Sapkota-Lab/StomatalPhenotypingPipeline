# Provisional seed v2 experiment

This run is a research experiment, not a validated production model.

## Training data

- Eight source-image groups
- 87 provisional boxes
- Six train groups with 80 boxes
- One validation group (`IMG_3010.JPG`) with two boxes
- One test group (`IMG_2990.JPG`) with five pale stomata
- Positive 512-pixel tiles only
- StomaQuant YOLOv12 starting checkpoint
- 20 CPU epochs at 512-pixel training resolution

The 63 boxes added after v1 came from model-assisted proposals on
`IMG_2992.JPG` and `IMG_3018.JPG` that were visually checked for stomatal
morphology. They are not independent expert annotations. The exact split is
in `../../training_data/provisional_seed_v2/split_manifest.csv`.

## Results

The best checkpoint reported precision 0.895, recall 0.500, mAP50 0.501, and
mAP50-95 0.301 on the two validation objects. This validation set is far too
small and non-independent to support an accuracy claim.

The checkpoint had zero recall on all five held-out pale stomata from
`IMG_2990.JPG`. The pale/low-contrast domain therefore remains unsolved.

As a standalone tiled detector, v2 over-proposed severely:

| Image | Original accepted/review | Seed v2 accepted/review |
|---|---:|---:|
| IMG_3012.JPG | 2 / 33 | 118 / 134 |
| IMG_3027.JPG | 5 / 33 | 63 / 112 |
| IMG_3032.JPG | 0 / 19 | 94 / 199 |

The checkpoint must not be used as a standalone counter.

Used only as a proposal source behind the independent YOLOS confidence and
spatial-agreement gate, v2 produced:

| Image | Production ensemble accepted | Seed v2 + YOLOS accepted |
|---|---:|---:|
| IMG_3012.JPG | 12 | 31 |
| IMG_3027.JPG | 23 | 31 |
| IMG_3032.JPG | 9 | 26 |

The persistent contact sheets under
`../../validation_artifacts/provisional_seed_v2/` were visually audited.
Codex judged 82 of the 88 crops to have clear paired guard cells and a central
pore; six were ambiguous. This is a morphology screen, not expert ground
truth, and it is not a valid precision estimate.

## Decision

Retain v2 as an optional research proposal model only. It provides strong
qualitative evidence that domain adaptation can recover clear/high-contrast
stomata missed by the public proposal checkpoints, but three facts prevent
production promotion:

1. the seed labels are provisional and partly model-assisted;
2. standalone proposals are extremely noisy;
3. held-out pale-stomata recall is zero.

The next defensible step is an expert-reviewed, image-grouped test set that
includes pale, clear, dense, debris-heavy, and small-morphology fields.
