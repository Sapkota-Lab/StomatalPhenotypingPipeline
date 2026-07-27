# UKY LCC execution

Use the Lipscomb Compute Cluster (LCC), not MCC, for model inference and
fine-tuning. LCC provides NVIDIA GPUs; MCC's documented compute nodes are CPU
only.

## One-time environment setup

Run setup from an LCC login node. Do not run inference on the login node.

```bash
cd /path/to/StomatalPhenotypingPipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-build-isolation -r requirements.txt
python download_model.py
```

The YOLOS checkpoint is downloaded from Hugging Face on first use. If compute
nodes cannot access the internet, run one small model-loading command on the
login node first so the checkpoint is present in the Hugging Face cache.

## Submit the batch

Edit the `#SBATCH --account` line in `lcc_ensemble.sbatch` to use the lab's
allocation account, then submit:

```bash
sbatch cluster/lcc_ensemble.sbatch
squeue -u "$USER"
```

The default V100 32 GB partition is sufficient for inference. The UKY A100
partition (`A2V80_ICE56M256_L`) is appropriate for larger fine-tuning or
hyperparameter sweeps, but is unnecessary for routine counting.

## Fine-tune on independent annotations

Before submitting, complete representative boxes with `box_annotator.py` and
fill `specimen_id` in `annotation_priority_v1.csv`. The exporter refuses to
build the dataset when completed images lack specimen grouping, preventing
related biological samples from leaking across splits.

`lcc_finetune.sbatch` builds 1024-pixel tiles, keeps every specimen in exactly
one train/validation/test split, fine-tunes on one A100, and evaluates the best
checkpoint on the untouched test split:

```bash
sbatch cluster/lcc_finetune.sbatch
```

The older consensus pseudo-label experiment is retained locally for
reproducibility, but is no longer used by this job. It failed to recover pale
stomata and generated hundreds of low-confidence false proposals.

## Fine-tune without target-image annotations

The public keypoint benchmark supplies expert boxes for low-contrast crab
apple and small soybean stomata. The leave-one-species-out job trains on one
species and tests untouched on the other, then reverses the direction:

```bash
sbatch cluster/lcc_external_domain_adaptation.sbatch
```

Before submission:

1. Replace `REPLACE_WITH_UKY_PROJECT_ACCOUNT` in the job file.
2. Ensure `external_validation/stomata_keypoint_benchmark/datasets/T-SBGH`
   and `T-MSAA` were copied with the project.
3. Ensure the LCC environment uses the versions in `requirements.txt`.

The job uses 640-pixel ownership-core training tiles and evaluates untouched
full images with overlapping tiled inference. Each direction selects its
confidence threshold only on its three validation images.
`calibrate_external_threshold.py` applies that threshold unchanged to the
other species and writes a pass/fail report under
`validation_artifacts/external_<direction>/calibrated_test_report.json`.
Local 30-epoch versions of both directions failed the transfer gate, so this
job is retained for reproducibility and larger-cluster experiments; its output
must not replace production counts unless the report explicitly passes.

## Native-resolution two-stage adaptation

The one-stage tiled detector's lowest-threshold proposal recall was only
0.684 on the mixed-domain untouched test. `lcc_stomadet_external.sbatch`
therefore fine-tunes the published StomaDet Faster R-CNN/FPN instead. It uses:

- all 15 public training images at native resolution;
- small p2–p6 anchors of 8, 16, 32, 64, and 128 pixels;
- brightness, contrast, saturation, and horizontal/vertical flip augmentation;
- four separate calibration images and six untouched test images containing
  both soybean and crab apple;
- the same frozen-confidence precision, recall, count-MAE, and bias report.
- a secondary six-view stability audit (rotations, flip, and CLAHE) that runs
  only after the untouched accuracy gate passes. Its thresholds are calibrated
  on individually accurate public validation images and applied unchanged to
  the public test images.

Submit after replacing the allocation placeholder:

```bash
sbatch cluster/lcc_stomadet_external.sbatch
```

The script uses only public expert boxes. It does not train on tomato images.
Passing the stability audit does not prove accuracy: a detector can be
consistently wrong. Tomato counts remain blocked unless both the external
accuracy gate and the unchanged stability thresholds pass.

Authoritative UKY documentation:

- https://ukyrcd.atlassian.net/wiki/spaces/RCDDocs/pages/162104254/LCC+system+overview
- https://ukyrcd.atlassian.net/wiki/spaces/RCDDocs/pages/162109880/SLURM+Queues
- https://ukyrcd.atlassian.net/wiki/spaces/RCDDocs/pages/472711225/How+to+profile+a+CUDA+job+with+NVIDIA+Nsight+on+LCC
