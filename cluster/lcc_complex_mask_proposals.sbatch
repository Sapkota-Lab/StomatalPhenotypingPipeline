#!/bin/bash
# Propose outer stomatal-complex contours for the balanced validation queue.

#SBATCH --job-name=stomata-masks
#SBATCH --account=gol_msa520_uksr
#SBATCH --partition=V4V32_SKY32M192_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-stomata-masks-%j.out
#SBATCH --error=slurm-stomata-masks-%j.err

set -euo pipefail

module --force purge
module load ccs/conda/python-3.10.12
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"

test -s models/sam2.1_t.pt
test -s combined_study_manifest.csv
test -d box_annotations
test -d Stomata_Images

python propose_complex_masks.py \
  --images Stomata_Images \
  --manifest combined_study_manifest.csv \
  --boxes box_annotations \
  --model models/sam2.1_t.pt \
  --output validation_artifacts/complex_mask_proposals_v1 \
  --max-per-image 8 \
  --seed 20260729 \
  --padding 0.75 \
  --device 0 \
  --overlays
