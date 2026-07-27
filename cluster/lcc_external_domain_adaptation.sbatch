#!/bin/bash
# Leave-one-species-out stomata domain adaptation on UKY LCC.
# Replace the account value below with the lab's UKY SLURM allocation.

#SBATCH --job-name=stomata-external-domain
#SBATCH --account=REPLACE_WITH_UKY_PROJECT_ACCOUNT
#SBATCH --partition=A2V80_ICE56M256_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-stomata-external-%j.out
#SBATCH --error=slurm-stomata-external-%j.err

set -euo pipefail

module purge
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

run_direction() {
  local train_domain="$1"
  local test_domain="$2"
  local tag="$3"
  local dataset_dir="training_data/external_${tag}"

  python build_external_domain_dataset.py \
    --output "$dataset_dir" \
    --train-domain "$train_domain" \
    --test-domain "$test_domain" \
    --validation-images 3 \
    --tile-size 640 \
    --tile-overlap 0.20

  yolo detect train \
    model=models/stomaquant_yolov12.pt \
    data="$dataset_dir/dataset.yaml" \
    imgsz=640 \
    epochs=100 \
    batch=-1 \
    patience=25 \
    device=0 \
    workers=8 \
    degrees=180 \
    translate=0.10 \
    scale=0.50 \
    fliplr=0.50 \
    flipud=0.50 \
    hsv_h=0.02 \
    hsv_s=0.35 \
    hsv_v=0.35 \
    mosaic=1.0 \
    close_mosaic=10 \
    project="$PROJECT_ROOT/training_runs" \
    name="external_${tag}"

  local model_path="$PROJECT_ROOT/training_runs/external_${tag}/weights/best.pt"
  local eval_dir="$PROJECT_ROOT/validation_artifacts/external_${tag}"

  python yolo_prediction_export.py \
    --model "$model_path" \
    --input "$dataset_dir/full_images/val" \
    --output "$eval_dir/validation_predictions.csv" \
    --device 0 \
    --image-size 640 \
    --tile-size 640 \
    --tile-overlap 0.20

  python yolo_prediction_export.py \
    --model "$model_path" \
    --input "$dataset_dir/full_images/test" \
    --output "$eval_dir/test_predictions.csv" \
    --device 0 \
    --image-size 640 \
    --tile-size 640 \
    --tile-overlap 0.20

  python calibrate_external_threshold.py \
    --calibration-ground-truth "$dataset_dir/val_coco.json" \
    --calibration-predictions "$eval_dir/validation_predictions.csv" \
    --test-ground-truth "$dataset_dir/test_coco.json" \
    --test-predictions "$eval_dir/test_predictions.csv" \
    --output "$eval_dir/calibrated_test_report.json"
}

run_direction T-SBGH T-MSAA sbgh_to_msaa
run_direction T-MSAA T-SBGH msaa_to_sbgh
