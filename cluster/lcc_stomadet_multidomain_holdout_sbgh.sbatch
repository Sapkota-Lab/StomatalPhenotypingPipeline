#!/bin/bash
# Balanced multi-domain training with every T-SBGH image held out for testing.

#SBATCH --job-name=stoma-lodo-sbgh
#SBATCH --account=gol_msa520_uksr
#SBATCH --partition=A2V80_ICE56M256_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-stoma-lodo-sbgh-%j.out
#SBATCH --error=slurm-stoma-lodo-sbgh-%j.err

set -euo pipefail

module --force purge
module load ccs/conda/python-3.10.12
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"

DATASET="training_data/external_multidomain_holdout_sbgh"
RUN="training_runs/stomadet_multidomain_holdout_sbgh"
ARTIFACTS="validation_artifacts/stomadet_multidomain_holdout_sbgh"

python build_external_multidomain_dataset.py \
  --output "$DATASET" \
  --heldout-domain T-SBGH \
  --validation-images-per-domain 3 \
  --training-records-per-domain 40 \
  --seed 20260728

python train_stomadet_external.py \
  --dataset "$DATASET" \
  --output "$RUN" \
  --dataset-prefix stomata_multidomain_holdout_sbgh \
  --device cuda \
  --workers 8 \
  --images-per-batch 2 \
  --max-iterations 5000 \
  --evaluation-period 500 \
  --checkpoint-period 1000

# Save low-threshold predictions once. The accepted threshold is selected only
# from the calibration split and then applied unchanged to held-out T-SBGH.
python stomadet_engine.py \
  --model "$RUN/model_final.pth" \
  --config "$RUN/config.yaml" \
  --input "$DATASET/full_images/val" \
  --output "$ARTIFACTS/calibration" \
  --confidence 0.001 \
  --device cuda \
  --no-overlays

python stomadet_engine.py \
  --model "$RUN/model_final.pth" \
  --config "$RUN/config.yaml" \
  --input "$DATASET/full_images/test" \
  --output "$ARTIFACTS/test" \
  --confidence 0.001 \
  --device cuda \
  --no-overlays

python calibrate_external_threshold.py \
  --calibration-ground-truth "$DATASET/val_coco.json" \
  --calibration-predictions "$ARTIFACTS/calibration/detections.csv" \
  --test-ground-truth "$DATASET/test_coco.json" \
  --test-predictions "$ARTIFACTS/test/detections.csv" \
  --output "$ARTIFACTS/calibrated_test_report.json"

CONFIDENCE="$(
  python -c \
    'import json; print(json.load(open("validation_artifacts/stomadet_multidomain_holdout_sbgh/calibrated_test_report.json"))["selected_confidence"])'
)"

# Always write tabular metrics and visual audits, even when the accuracy gate
# fails, so a failed experiment remains diagnosable.
python evaluate_external_predictions.py \
  --ground-truth "$DATASET/val_coco.json" \
  --predictions "$ARTIFACTS/calibration/detections.csv" \
  --manifest "$DATASET/manifest.csv" \
  --confidence "$CONFIDENCE" \
  --output "$ARTIFACTS/calibration_accuracy"

python evaluate_external_predictions.py \
  --ground-truth "$DATASET/test_coco.json" \
  --predictions "$ARTIFACTS/test/detections.csv" \
  --manifest "$DATASET/manifest.csv" \
  --confidence "$CONFIDENCE" \
  --output "$ARTIFACTS/test_accuracy"

python render_evaluation_overlays.py \
  --images "$DATASET/full_images/val" \
  --ground-truth "$DATASET/val_coco.json" \
  --predictions "$ARTIFACTS/calibration/detections.csv" \
  --confidence "$CONFIDENCE" \
  --output "$ARTIFACTS/calibration_visual_audit"

python render_evaluation_overlays.py \
  --images "$DATASET/full_images/test" \
  --ground-truth "$DATASET/test_coco.json" \
  --predictions "$ARTIFACTS/test/detections.csv" \
  --confidence "$CONFIDENCE" \
  --output "$ARTIFACTS/test_visual_audit"

# Exit 2 is an intentional scientific gate result, not an infrastructure error.
python -c \
  'import json,sys; report=json.load(open("validation_artifacts/stomadet_multidomain_holdout_sbgh/calibrated_test_report.json")); sys.exit(0 if report["gate"]["passed"] else 2)'
