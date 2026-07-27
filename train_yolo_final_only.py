#!/usr/bin/env python3
"""Fine-tune YOLO without large per-epoch optimizer checkpoints.

This local-MPS helper keeps training state in memory and writes only the final
inference model. It is intended for disk-constrained exploratory runs; cluster
training should retain normal best/last checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("training_runs"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch,
        patience=0,
        device=args.device,
        workers=0,
        degrees=180,
        translate=0.10,
        scale=0.50,
        fliplr=0.50,
        flipud=0.50,
        hsv_h=0.02,
        hsv_s=0.35,
        hsv_v=0.35,
        mosaic=1.0,
        close_mosaic=5,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        cos_lr=True,
        save=False,
        plots=False,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_model))
    print(f"Saved final inference model to {args.output_model}")


if __name__ == "__main__":
    main()
