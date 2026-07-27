# Stomatal Phenotyping Pipeline

Automated detection and per-image counting of coffee-bean-shaped stomata in high-resolution microscope images.

## Recommended engine

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

## Count images

```bash
.venv/bin/python ensemble_stomata_engine.py \
  --input Stomata_Images \
  --output results \
  --device auto
```

`auto` selects CUDA, then Apple Metal, then CPU. CPU works but transformer
inference is substantially slower. Output files are:

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

## Validation status

The engine targets stomata rather than pavement-cell contours, but the current
public checkpoints do **not** yet produce validated biological counts on this
mixed batch. A complete 74-image run produced 323 two-model consensus
detections across 55 eligible fields and placed 855 proposals in review. A
batch-wide crop audit found residual round-debris false positives even among
some consensus detections, while pale and small-morphology fields contain
obvious missed stomata. Treat `accepted_count` as a conservative candidate
count, not ground truth.

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
