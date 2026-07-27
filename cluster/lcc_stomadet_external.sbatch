#!/bin/bash
# Native-resolution two-stage adaptation using only public expert labels.

#SBATCH --job-name=stomadet-external
#SBATCH --account=REPLACE_WITH_UKY_PROJECT_ACCOUNT
#SBATCH --partition=A2V80_ICE56M256_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-stomadet-external-%j.out
#SBATCH --error=slurm-stomadet-external-%j.err

set -euo pipefail

module purge
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

python build_external_mixed_dataset.py \
  --output training_data/external_mixed_tiled640 \
  --validation-images-per-domain 2 \
  --test-images-per-domain 3 \
  --tile-size 640 \
  --tile-overlap 0.20

python train_stomadet_external.py \
  --dataset training_data/external_mixed_tiled640 \
  --output training_runs/stomadet_external_mixed \
  --device cuda \
  --workers 8 \
  --images-per-batch 2 \
  --max-iterations 3000 \
  --evaluation-period 250 \
  --checkpoint-period 500

python stomadet_engine.py \
  --model training_runs/stomadet_external_mixed/model_final.pth \
  --config training_runs/stomadet_external_mixed/config.yaml \
  --input training_data/external_mixed_tiled640/full_images/val \
  --output validation_artifacts/stomadet_external_mixed/validation \
  --confidence 0.001 \
  --device cuda \
  --no-overlays

python stomadet_engine.py \
  --model training_runs/stomadet_external_mixed/model_final.pth \
  --config training_runs/stomadet_external_mixed/config.yaml \
  --input training_data/external_mixed_tiled640/full_images/test \
  --output validation_artifacts/stomadet_external_mixed/test \
  --confidence 0.001 \
  --device cuda \
  --no-overlays

python calibrate_external_threshold.py \
  --calibration-ground-truth training_data/external_mixed_tiled640/val_coco.json \
  --calibration-predictions validation_artifacts/stomadet_external_mixed/validation/detections.csv \
  --test-ground-truth training_data/external_mixed_tiled640/test_coco.json \
  --test-predictions validation_artifacts/stomadet_external_mixed/test/detections.csv \
  --output validation_artifacts/stomadet_external_mixed/calibrated_test_report.json

# The stability audit is secondary and is intentionally skipped unless the
# untouched external accuracy gate passes.
python -c 'import json,sys; report=json.load(open("validation_artifacts/stomadet_external_mixed/calibrated_test_report.json")); sys.exit(0 if report["gate"]["passed"] else 2)'
CONFIDENCE="$(python -c 'import json; print(json.load(open("validation_artifacts/stomadet_external_mixed/calibrated_test_report.json"))["selected_confidence"])')"

python evaluate_external_predictions.py \
  --ground-truth training_data/external_mixed_tiled640/val_coco.json \
  --predictions validation_artifacts/stomadet_external_mixed/validation/detections.csv \
  --confidence "$CONFIDENCE" \
  --output validation_artifacts/stomadet_external_mixed/validation_accuracy

python stomadet_tta_engine.py \
  --model training_runs/stomadet_external_mixed/model_final.pth \
  --config training_runs/stomadet_external_mixed/config.yaml \
  --input training_data/external_mixed_tiled640/full_images/val \
  --output validation_artifacts/stomadet_external_mixed/validation_tta \
  --min-view-confidence "$CONFIDENCE" \
  --min-support 1 \
  --min-median-confidence 0 \
  --device cuda \
  --no-overlays

python stomadet_tta_engine.py \
  --model training_runs/stomadet_external_mixed/model_final.pth \
  --config training_runs/stomadet_external_mixed/config.yaml \
  --input training_data/external_mixed_tiled640/full_images/test \
  --output validation_artifacts/stomadet_external_mixed/test_tta \
  --min-view-confidence "$CONFIDENCE" \
  --min-support 1 \
  --min-median-confidence 0 \
  --device cuda \
  --no-overlays

python calibrate_stability_gate.py \
  --calibration-stability validation_artifacts/stomadet_external_mixed/validation_tta/stability.csv \
  --calibration-accuracy validation_artifacts/stomadet_external_mixed/validation_accuracy/per_image.csv \
  --accuracy-summary validation_artifacts/stomadet_external_mixed/calibrated_test_report.json \
  --output-thresholds validation_artifacts/stomadet_external_mixed/stability_thresholds.json \
  --apply-stability validation_artifacts/stomadet_external_mixed/test_tta/stability.csv \
  --output-audit validation_artifacts/stomadet_external_mixed/test_stability_audit.csv
