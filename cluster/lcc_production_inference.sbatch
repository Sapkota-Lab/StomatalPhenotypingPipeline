#!/bin/bash
# Run the calibrated tomato detector on all 32 in-scope plant images.

#SBATCH --job-name=stomata-production
#SBATCH --account=gol_msa520_uksr
#SBATCH --partition=V4V32_SKY32M192_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-stomata-production-%j.out
#SBATCH --error=slurm-stomata-production-%j.err

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
CALIBRATION="$PROJECT_ROOT/validation_artifacts/tomato_full_image/calibration"
OUTPUT="$PROJECT_ROOT/validation_artifacts/tomato_production_v1"
MODEL="$PROJECT_ROOT/training_runs/annotated_v1/weights/best.pt"

test -s "$MODEL"
test -s "$DATASET/split_manifest.csv"
test -s "$CALIBRATION/calibration_report.json"
test -s combined_study_manifest.csv

python stomata_engine.py \
  --model "$MODEL" \
  --input Stomata_Images \
  --output "$OUTPUT/raw_predictions" \
  --confidence 0.001 \
  --review-confidence 0.001 \
  --tile-size 1024 \
  --overlap 0.20 \
  --nms-iou 0.45 \
  --center-distance-nms 0.25 \
  --device 0 \
  --split-manifest "$DATASET/split_manifest.csv" \
  --split all \
  --no-overlays

python apply_tomato_calibration.py \
  --predictions "$OUTPUT/raw_predictions/detections.csv" \
  --calibration-report "$CALIBRATION/calibration_report.json" \
  --output "$OUTPUT/calibrated_detections.csv"

python build_prediction_ledger.py \
  --manifest combined_study_manifest.csv \
  --predictions "$OUTPUT/calibrated_detections.csv" \
  --images Stomata_Images \
  --output "$OUTPUT/stomata.csv" \
  --run-id tomato_yolov12_full_v1 \
  --calibration-id tomato_platt_validation_v1

python write_run_manifest.py \
  --output "$OUTPUT/run_manifest.json" \
  --ledger "$OUTPUT/stomata.csv" \
  --model "$MODEL" \
  --calibration-report "$CALIBRATION/calibration_report.json" \
  --input-manifest combined_study_manifest.csv \
  --run-id tomato_yolov12_full_v1 \
  --device cuda \
  --tile-size 1024 \
  --overlap 0.20 \
  --nms-iou 0.45 \
  --center-distance-nms 0.25
