#!/bin/bash
# UKY Lipscomb Compute Cluster (LCC) GPU batch job.
# Replace the account value below with the lab's UKY SLURM allocation.

#SBATCH --job-name=stomata-ensemble
#SBATCH --account=REPLACE_WITH_UKY_PROJECT_ACCOUNT
#SBATCH --partition=V4V32_SKY32M192_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-stomata-%j.out
#SBATCH --error=slurm-stomata-%j.err

set -euo pipefail

module purge
module load ccs/cuda/12.2.0_535.54.03

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source .venv/bin/activate

python ensemble_stomata_engine.py \
  --input Stomata_Images \
  --output results \
  --device cuda
