#!/bin/bash
# Evaluate the trained tomato detector once on untouched test varieties.

#SBATCH --job-name=stomata-test
#SBATCH --account=gol_msa520_uksr
#SBATCH --partition=A2V80_ICE56M256_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-stomata-test-%j.out
#SBATCH --error=slurm-stomata-test-%j.err

set -euo pipefail

module --force purge
module load ccs/conda/python-3.10.12
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"

test -s training_runs/annotated_v1/weights/best.pt
test -s training_data/annotated_v1/dataset.yaml

yolo detect val \
  model="$PROJECT_ROOT/training_runs/annotated_v1/weights/best.pt" \
  data="$PROJECT_ROOT/training_data/annotated_v1/dataset.yaml" \
  split=test \
  imgsz=1024 \
  batch=16 \
  device=0 \
  workers=8 \
  plots=True \
  project="$PROJECT_ROOT/training_runs" \
  name=annotated_v1_test \
  exist_ok=True
