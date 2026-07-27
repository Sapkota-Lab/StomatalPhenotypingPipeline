#!/bin/bash
# Independently annotated stomata fine-tuning on UKY LCC.
# Replace the account value below with the lab's UKY SLURM allocation.

#SBATCH --job-name=stomata-finetune
#SBATCH --account=REPLACE_WITH_UKY_PROJECT_ACCOUNT
#SBATCH --partition=A2V80_ICE56M256_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-stomata-train-%j.out
#SBATCH --error=slurm-stomata-train-%j.err

set -euo pipefail

module purge
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

python build_annotated_dataset.py \
  --images Stomata_Images \
  --annotations box_annotations \
  --metadata annotation_priority_v1.csv \
  --group-column specimen_id \
  --output training_data/annotated_v1 \
  --tile-size 1024

yolo detect train \
  model=models/stomaquant_yolov12.pt \
  data=training_data/annotated_v1/dataset.yaml \
  imgsz=1024 \
  epochs=100 \
  batch=-1 \
  patience=20 \
  device=0 \
  workers=8 \
  project=training_runs \
  name=annotated_v1

yolo detect val \
  model=training_runs/annotated_v1/weights/best.pt \
  data=training_data/annotated_v1/dataset.yaml \
  split=test \
  imgsz=1024 \
  batch=16 \
  device=0 \
  workers=8 \
  plots=True \
  project=training_runs \
  name=annotated_v1_test
