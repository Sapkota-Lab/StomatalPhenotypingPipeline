#!/bin/bash
# Full-resolution inference, duplicate fusion, and held-out calibration.

#SBATCH --job-name=stomata-full-cal
#SBATCH --account=gol_msa520_uksr
#SBATCH --partition=A2V80_ICE56M256_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-stomata-full-cal-%j.out
#SBATCH --error=slurm-stomata-full-cal-%j.err

set -euo pipefail

module --force purge
module load ccs/conda/python-3.10.12
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"

DATASET="$PROJECT_ROOT/training_data/annotated_v1"
OUTPUT="$PROJECT_ROOT/validation_artifacts/tomato_full_image"
MODEL="$PROJECT_ROOT/training_runs/annotated_v1/weights/best.pt"

test -s "$MODEL"
test -s "$DATASET/split_manifest.csv"

python build_full_image_reference.py \
  --annotations box_annotations \
  --split-manifest "$DATASET/split_manifest.csv" \
  --output "$DATASET/full_image_reference"

python stomata_engine.py \
  --model "$MODEL" \
  --input Stomata_Images \
  --output "$OUTPUT/validation_predictions" \
  --confidence 0.001 \
  --review-confidence 0.001 \
  --tile-size 1024 \
  --overlap 0.20 \
  --nms-iou 0.45 \
  --center-distance-nms 0.25 \
  --device 0 \
  --split-manifest "$DATASET/split_manifest.csv" \
  --split val \
  --no-overlays

python stomata_engine.py \
  --model "$MODEL" \
  --input Stomata_Images \
  --output "$OUTPUT/test_predictions" \
  --confidence 0.001 \
  --review-confidence 0.001 \
  --tile-size 1024 \
  --overlap 0.20 \
  --nms-iou 0.45 \
  --center-distance-nms 0.25 \
  --device 0 \
  --split-manifest "$DATASET/split_manifest.csv" \
  --split test \
  --no-overlays

python calibrate_tomato_confidence.py \
  --validation-ground-truth "$DATASET/full_image_reference/val_coco.json" \
  --validation-predictions "$OUTPUT/validation_predictions/detections.csv" \
  --test-ground-truth "$DATASET/full_image_reference/test_coco.json" \
  --test-predictions "$OUTPUT/test_predictions/detections.csv" \
  --output "$OUTPUT/calibration"
