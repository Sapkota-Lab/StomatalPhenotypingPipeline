#!/usr/bin/env python3
"""Fine-tune StomaDet Faster R-CNN on public mixed-domain expert boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_train_loader,
    transforms as T,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage


class StomataTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.60, 1.40),
            T.RandomContrast(0.55, 1.45),
            T.RandomSaturation(0.70, 1.30),
        ]
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=augmentations,
                image_format="BGR",
                use_instance_mask=False,
                use_keypoint=False,
            ),
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / "evaluation"),
        )


def register_datasets(dataset_root: Path, prefix: str) -> tuple[str, str]:
    train_name = f"{prefix}_train"
    validation_name = f"{prefix}_validation"
    register_coco_instances(
        train_name,
        {},
        str(dataset_root / "train_coco.json"),
        str(dataset_root / "full_images" / "train"),
    )
    register_coco_instances(
        validation_name,
        {},
        str(dataset_root / "val_coco.json"),
        str(dataset_root / "full_images" / "val"),
    )
    return train_name, validation_name


def build_training_config(args: argparse.Namespace):
    train_name, validation_name = register_datasets(
        args.dataset,
        args.dataset_prefix,
    )
    cfg = get_cfg()
    cfg.merge_from_file(str(args.base_config))
    cfg.MODEL.WEIGHTS = str(args.weights.resolve())
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (validation_name,)
    cfg.DATALOADER.NUM_WORKERS = args.workers

    # Native-resolution input plus p2 anchors matched to 20–40 px stomata.
    cfg.INPUT.MIN_SIZE_TRAIN = (896, 1024, 1152)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 1536
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1536
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
        [0.50, 1.0, 2.0]
    ]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 4000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    cfg.SOLVER.IMS_PER_BATCH = args.images_per_batch
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iterations
    cfg.SOLVER.WARMUP_ITERS = min(200, max(10, args.max_iterations // 20))
    cfg.SOLVER.STEPS = (
        int(args.max_iterations * 0.70),
        int(args.max_iterations * 0.90),
    )
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.evaluation_period
    cfg.OUTPUT_DIR = str(args.output.resolve())
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("training_data/external_mixed_tiled640"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("models/external/stomadet/models/stoma_detector.yaml"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/external/stomadet/models/stoma_detector.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_runs/stomadet_external_mixed"),
    )
    parser.add_argument("--dataset-prefix", default="stomata_external_mixed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--images-per-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--checkpoint-period", type=int, default=500)
    parser.add_argument("--evaluation-period", type=int, default=250)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate registrations/config/model construction without training.",
    )
    parser.add_argument(
        "--forward-smoke",
        action="store_true",
        help="With --validate-only, run one training loss forward pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (
        args.dataset / "train_coco.json",
        args.dataset / "val_coco.json",
        args.base_config,
        args.weights,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    cfg = build_training_config(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.yaml").write_text(cfg.dump())
    if args.validate_only:
        model = build_model(cfg)
        checkpoint = DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        records = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        annotations = sum(len(record.get("annotations", [])) for record in records)
        loader = StomataTrainer.build_train_loader(cfg)
        batch = next(iter(loader))
        instances = sum(len(item["instances"]) for item in batch)
        loss_text = ""
        if args.forward_smoke:
            model.train()
            with torch.no_grad(), EventStorage():
                losses = model(batch)
            total_loss = sum(float(value.detach().cpu()) for value in losses.values())
            if not torch.isfinite(torch.tensor(total_loss)):
                raise ValueError(f"Non-finite smoke loss: {total_loss}")
            loss_text = f"; forward loss {total_loss:.4f}"
        print(
            f"Validated Faster R-CNN configuration: {parameters:,} parameters; "
            f"{len(records)} full images, {annotations} boxes; "
            f"first augmented batch retains {instances} boxes; "
            f"checkpoint keys: {', '.join(sorted(checkpoint)) or 'none'}"
            f"{loss_text}"
        )
        return
    trainer = StomataTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
