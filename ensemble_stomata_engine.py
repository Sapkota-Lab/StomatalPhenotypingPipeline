#!/usr/bin/env python3
"""High-precision three-model stomata counting for microscope images.

The engine combines:

* StomaQuant YOLOv12 proposals, which are conservative on this image domain.
* A fine-tuned YOLOS detector, which has higher recall but more cell-wall
  false positives.
* StomaDet proposals from a bright-field leaf-impression detector.

No model may add a count alone. StomaQuant and StomaDet proposals are counted
only when a high-confidence YOLOS detection agrees spatially. Unmatched
proposals remain review candidates rather than silently inflating the count.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from ultralytics import YOLO

from stomadet_engine import (
    DEFAULT_CONFIG as DEFAULT_STOMADET_CONFIG,
    DEFAULT_MODEL as DEFAULT_STOMADET_MODEL,
    build_stomadet,
    infer_stomadet,
)
from stomata_engine import (
    Detection,
    image_paths,
    image_quality,
    infer_image,
    iou,
    patch_yolov12_attention,
)
from yolos_stomata_engine import DEFAULT_MODEL as DEFAULT_YOLOS_MODEL
from yolos_stomata_engine import infer_yolos, open_oriented_image


COUNT_FIELDS = [
    "filename",
    "analysis_status",
    "accepted_count",
    "stomaquant_yolos_count",
    "stomadet_yolos_count",
    "review_count",
    "stomaquant_only_review_count",
    "yolos_only_review_count",
    "image_quality",
    "focus_score",
    "contrast_score",
    "clipped_fraction",
]
DETECTION_FIELDS = [
    "filename",
    "detection_id",
    "status",
    "source",
    "stomaquant_confidence",
    "yolos_confidence",
    "stomadet_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass(frozen=True)
class ClassifiedDetection:
    detection: Detection
    status: str
    source: str
    yolos_confidence: float | None = None
    stomadet_confidence: float | None = None


def normalized_center_distance(first: Detection, second: Detection) -> float:
    first_center = ((first.x1 + first.x2) / 2, (first.y1 + first.y2) / 2)
    second_center = ((second.x1 + second.x2) / 2, (second.y1 + second.y2) / 2)
    scale = max(first.x2 - first.x1, first.y2 - first.y1, 1e-9)
    return math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    ) / scale


def classify_ensemble(
    stomaquant: list[Detection],
    yolos: list[Detection],
    stomadet: list[Detection] | None = None,
    yolos_agreement: float = 0.85,
    stomadet_yolos_agreement: float = 0.92,
    yolos_only_review: float = 0.95,
    agreement_iou: float = 0.05,
    agreement_center_distance: float = 0.55,
) -> list[ClassifiedDetection]:
    """Classify proposals into accepted and review tiers.

    A YOLOS proposal may promote at most one StomaQuant proposal. If multiple
    YOLOS boxes match the same proposal, the highest YOLOS confidence is kept.
    """
    stomadet = stomadet or []
    matches: dict[int, float] = {}
    matched_yolos: set[int] = set()
    stomaquant_boxes = np.asarray([item.box for item in stomaquant], dtype=np.float32)

    for yolos_index, yolos_detection in enumerate(yolos):
        if yolos_detection.confidence < yolos_agreement or not stomaquant:
            continue
        overlaps = iou(np.asarray(yolos_detection.box, dtype=np.float32), stomaquant_boxes)
        distances = np.asarray([
            normalized_center_distance(yolos_detection, candidate)
            for candidate in stomaquant
        ])
        eligible = np.flatnonzero(
            (overlaps >= agreement_iou)
            | (distances <= agreement_center_distance)
        )
        if eligible.size == 0:
            continue
        # Prefer overlap; use center distance to break ties when boxes barely
        # overlap or differ in size between the two architectures.
        best = max(
            eligible.tolist(),
            key=lambda index: (float(overlaps[index]), -float(distances[index])),
        )
        matches[best] = max(matches.get(best, 0.0), yolos_detection.confidence)
        matched_yolos.add(yolos_index)

    classified: list[ClassifiedDetection] = []
    for index, detection in enumerate(stomaquant):
        agreement_confidence = matches.get(index)
        if agreement_confidence is not None:
            classified.append(ClassifiedDetection(
                detection, "accepted", "stomaquant_yolos",
                agreement_confidence,
            ))
        else:
            classified.append(ClassifiedDetection(
                detection, "review", "stomaquant_only",
            ))

    # StomaDet is from a closer bright-field domain but has image-dependent
    # false positives. It may add a count only when the independent YOLOS
    # detector also agrees. Do not add a second box for a stoma already
    # represented by StomaQuant.
    accepted_boxes = [
        item.detection for item in classified if item.status == "accepted"
    ]
    stomadet_boxes = np.asarray(
        [item.box for item in stomadet],
        dtype=np.float32,
    )
    used_stomadet: set[int] = set()
    for yolos_index, yolos_detection in enumerate(yolos):
        if (
            yolos_index in matched_yolos
            or yolos_detection.confidence < stomadet_yolos_agreement
            or not stomadet
        ):
            continue
        overlaps = iou(
            np.asarray(yolos_detection.box, dtype=np.float32),
            stomadet_boxes,
        )
        distances = np.asarray([
            normalized_center_distance(yolos_detection, candidate)
            for candidate in stomadet
        ])
        eligible = [
            index for index in np.flatnonzero(
                (overlaps >= agreement_iou)
                | (distances <= agreement_center_distance)
            ).tolist()
            if index not in used_stomadet
        ]
        if not eligible:
            continue
        best = max(
            eligible,
            key=lambda index: (float(overlaps[index]), -float(distances[index])),
        )
        candidate = stomadet[best]
        duplicates_existing = any(
            iou(
                np.asarray(candidate.box, dtype=np.float32),
                np.asarray([accepted.box], dtype=np.float32),
            )[0] >= agreement_iou
            or normalized_center_distance(candidate, accepted)
            <= agreement_center_distance
            for accepted in accepted_boxes
        )
        if duplicates_existing:
            matched_yolos.add(yolos_index)
            used_stomadet.add(best)
            continue
        classified.append(ClassifiedDetection(
            candidate,
            "accepted",
            "stomadet_yolos",
            yolos_confidence=yolos_detection.confidence,
            stomadet_confidence=candidate.confidence,
        ))
        accepted_boxes.append(candidate)
        matched_yolos.add(yolos_index)
        used_stomadet.add(best)

    for yolos_index, detection in enumerate(yolos):
        if (
            yolos_index not in matched_yolos
            and detection.confidence >= yolos_only_review
        ):
            classified.append(ClassifiedDetection(
                detection,
                "review",
                "yolos_only",
                yolos_confidence=detection.confidence,
            ))
    return classified


def draw_ensemble_overlay(
    image: np.ndarray,
    detections: list[ClassifiedDetection],
    output: Path,
    max_width: int = 2400,
) -> None:
    scale = min(1.0, max_width / image.shape[1])
    view = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1 else image.copy()
    )
    colors = {
        "accepted": (0, 220, 0),
        "review": (0, 180, 255),
        "stomaquant_only": (0, 180, 255),
        "stomaquant_yolos": (0, 220, 0),
        "yolos_only": (255, 180, 0),
        "stomadet_yolos": (220, 80, 220),
    }
    counters = {"accepted": 0, "review": 0}
    for item in detections:
        counters[item.status] += 1
        prefix = "A" if item.status == "accepted" else (
            "Y" if item.source == "yolos_only" else "R"
        )
        color = (
            colors[item.source]
            if item.source in {
                "stomaquant_only",
                "stomaquant_yolos",
                "yolos_only",
                "stomadet_yolos",
            }
            else colors[item.status]
        )
        x1, y1, x2, y2 = (round(value * scale) for value in item.detection.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, max(1, round(3 * scale)))
        cv2.putText(
            view,
            f"{prefix}{counters[item.status]}:{item.detection.confidence:.2f}",
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    stomaquant_count = sum(
        item.source == "stomaquant_yolos" for item in detections
    )
    stomadet_count = sum(
        item.source == "stomadet_yolos" for item in detections
    )
    cv2.putText(
        view,
        (
            f"accepted: {counters['accepted']} "
            f"(SQ/YOLOS {stomaquant_count}, StomaDet/YOLOS {stomadet_count}) "
            f" review: {counters['review']}"
        ),
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 40, 40),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), view):
        raise OSError(f"Could not write overlay {output}")


def draw_status_overlay(
    image: np.ndarray,
    output: Path,
    status: str,
    max_width: int = 2400,
) -> None:
    scale = min(1.0, max_width / image.shape[1])
    view = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1 else image.copy()
    )
    cv2.putText(
        view,
        status.replace("_", " ").upper(),
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 230),
        3,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), view):
        raise OSError(f"Could not write overlay {output}")


def automatic_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_scope_manifest(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "analysis_scope"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest {path} must contain filename and analysis_scope columns"
            )
        scopes: dict[str, str] = {}
        for row in reader:
            filename = row["filename"].strip()
            scope = row["analysis_scope"].strip()
            if scope not in {"eligible", "out_of_scope"}:
                raise ValueError(
                    f"Invalid analysis_scope {scope!r} for {filename}"
                )
            if filename in scopes:
                raise ValueError(f"Duplicate manifest row for {filename}")
            scopes[filename] = scope
    return scopes


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_checkpoint(output: Path) -> tuple[list[dict], list[dict]]:
    counts_path = output / "counts.csv"
    detections_path = output / "detections.csv"
    if not counts_path.exists() or not detections_path.exists():
        return [], []
    with counts_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COUNT_FIELDS:
            print("Ignoring incompatible counts checkpoint")
            return [], []
        counts = list(reader)
    with detections_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DETECTION_FIELDS:
            print("Ignoring incompatible detections checkpoint")
            return [], []
        detections = list(reader)
    count_names = {row["filename"] for row in counts}
    if any(row["filename"] not in count_names for row in detections):
        print("Ignoring inconsistent checkpoint")
        return [], []
    return counts, detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--stomaquant-model",
        type=Path,
        default=Path("models/stomaquant_yolov12.pt"),
    )
    parser.add_argument("--yolos-model", default=DEFAULT_YOLOS_MODEL)
    parser.add_argument(
        "--stomadet-model",
        type=Path,
        default=DEFAULT_STOMADET_MODEL,
    )
    parser.add_argument(
        "--stomadet-config",
        type=Path,
        default=DEFAULT_STOMADET_CONFIG,
    )
    parser.add_argument("--stomadet-confidence", type=float, default=0.05)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("image_manifest.csv"),
        help="optional CSV with filename and analysis_scope columns",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or CUDA index")
    parser.add_argument("--stomaquant-review", type=float, default=0.05)
    parser.add_argument("--yolos-agreement", type=float, default=0.85)
    parser.add_argument(
        "--stomadet-yolos-agreement",
        type=float,
        default=0.92,
    )
    parser.add_argument("--yolos-only-review", type=float, default=0.95)
    parser.add_argument("--agreement-iou", type=float, default=0.05)
    parser.add_argument("--agreement-center-distance", type=float, default=0.55)
    parser.add_argument("--tile-size", type=int, default=1216)
    parser.add_argument("--yolos-tile-width", type=int, default=864)
    parser.add_argument("--yolos-tile-height", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from compatible per-image CSV checkpoints (default: true)",
    )
    args = parser.parse_args()

    if not args.stomaquant_model.exists():
        parser.error(f"Model not found: {args.stomaquant_model}")
    if not args.stomadet_model.exists():
        parser.error(f"Model not found: {args.stomadet_model}")
    if not args.stomadet_config.exists():
        parser.error(f"Config not found: {args.stomadet_config}")
    if not 0 <= args.stomaquant_review <= 1:
        parser.error("Require 0 <= stomaquant-review <= 1")
    paths = image_paths(args.input)
    if not paths:
        parser.error(f"No supported images found in {args.input}")
    try:
        scope_manifest = load_scope_manifest(args.manifest)
    except ValueError as error:
        parser.error(str(error))

    device = automatic_device() if args.device == "auto" else args.device
    patch_yolov12_attention()
    stomaquant_model = YOLO(str(args.stomaquant_model))
    processor = AutoImageProcessor.from_pretrained(args.yolos_model, use_fast=False)
    yolos_model = (
        AutoModelForObjectDetection.from_pretrained(args.yolos_model)
        .eval()
        .to(device)
    )
    stomadet_device = "cuda" if str(device).startswith("cuda") else "cpu"
    stomadet_predictor = build_stomadet(
        args.stomadet_config,
        args.stomadet_model,
        args.stomadet_confidence,
        stomadet_device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    count_rows, detection_rows = (
        load_checkpoint(args.output) if args.resume else ([], [])
    )
    completed = {row["filename"] for row in count_rows}
    for path in paths:
        if path.name in completed:
            print(f"{path.name}: already completed; skipping")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        scope = scope_manifest.get(path.name)
        if scope == "out_of_scope":
            quality = image_quality(image)
            count_rows.append({
                "filename": path.name,
                "analysis_status": "out_of_scope",
                "accepted_count": 0,
                "stomaquant_yolos_count": 0,
                "stomadet_yolos_count": 0,
                "review_count": 0,
                "stomaquant_only_review_count": 0,
                "yolos_only_review_count": 0,
                "image_quality": quality["image_quality"],
                "focus_score": quality["focus_score"],
                "contrast_score": quality["contrast_score"],
                "clipped_fraction": quality["clipped_fraction"],
            })
            draw_status_overlay(
                image,
                args.output / "overlays" / f"{path.stem}_overlay.jpg",
                "out_of_scope",
            )
            write_csv_atomic(args.output / "counts.csv", COUNT_FIELDS, count_rows)
            write_csv_atomic(
                args.output / "detections.csv",
                DETECTION_FIELDS,
                detection_rows,
            )
            print(f"{path.name}: out_of_scope; skipped")
            continue
        oriented = open_oriented_image(path)
        if (image.shape[1], image.shape[0]) != oriented.size:
            raise ValueError(
                f"Orientation mismatch for {path}: OpenCV={image.shape[1::-1]}, "
                f"Pillow={oriented.size}"
            )
        quality = image_quality(image)

        stomaquant = infer_image(
            stomaquant_model,
            image,
            args.tile_size,
            args.overlap,
            args.stomaquant_review,
            args.nms_iou,
            device,
        )
        yolos = infer_yolos(
            processor,
            yolos_model,
            oriented,
            min(args.yolos_agreement, args.yolos_only_review),
            args.overlap,
            args.yolos_tile_width,
            args.yolos_tile_height,
            device,
        )
        stomadet = infer_stomadet(stomadet_predictor, image)
        classified = classify_ensemble(
            stomaquant=stomaquant,
            yolos=yolos,
            stomadet=stomadet,
            yolos_agreement=args.yolos_agreement,
            stomadet_yolos_agreement=args.stomadet_yolos_agreement,
            yolos_only_review=args.yolos_only_review,
            agreement_iou=args.agreement_iou,
            agreement_center_distance=args.agreement_center_distance,
        )
        accepted = [item for item in classified if item.status == "accepted"]
        review = [item for item in classified if item.status == "review"]
        stomaquant_count = sum(
            item.source == "stomaquant_yolos" for item in accepted
        )
        stomadet_count = sum(
            item.source == "stomadet_yolos" for item in accepted
        )
        stomaquant_only_count = sum(
            item.source == "stomaquant_only" for item in review
        )
        yolos_only_count = sum(item.source == "yolos_only" for item in review)
        count_rows.append({
            "filename": path.name,
            "analysis_status": (
                "counted" if scope == "eligible" else "counted_unreviewed"
            ),
            "accepted_count": len(accepted),
            "stomaquant_yolos_count": stomaquant_count,
            "stomadet_yolos_count": stomadet_count,
            "review_count": len(review),
            "stomaquant_only_review_count": stomaquant_only_count,
            "yolos_only_review_count": yolos_only_count,
            "image_quality": quality["image_quality"],
            "focus_score": quality["focus_score"],
            "contrast_score": quality["contrast_score"],
            "clipped_fraction": quality["clipped_fraction"],
        })
        for detection_id, item in enumerate(classified, 1):
            detection_rows.append({
                "filename": path.name,
                "detection_id": detection_id,
                "status": item.status,
                "source": item.source,
                "stomaquant_confidence": (
                    round(item.detection.confidence, 6)
                    if item.source in {"stomaquant_yolos", "stomaquant_only"}
                    else ""
                ),
                "yolos_confidence": (
                    round(item.yolos_confidence, 6)
                    if item.yolos_confidence is not None else ""
                ),
                "stomadet_confidence": (
                    round(item.stomadet_confidence, 6)
                    if item.stomadet_confidence is not None else ""
                ),
                "x1": round(item.detection.x1, 2),
                "y1": round(item.detection.y1, 2),
                "x2": round(item.detection.x2, 2),
                "y2": round(item.detection.y2, 2),
            })
        draw_ensemble_overlay(
            image,
            classified,
            args.output / "overlays" / f"{path.stem}_overlay.jpg",
        )
        write_csv_atomic(args.output / "counts.csv", COUNT_FIELDS, count_rows)
        write_csv_atomic(
            args.output / "detections.csv",
            DETECTION_FIELDS,
            detection_rows,
        )
        print(
            f"{path.name}: accepted={len(accepted)} "
            f"(SQ/YOLOS {stomaquant_count}, "
            f"StomaDet/YOLOS {stomadet_count}) review={len(review)} "
            f"quality={quality['image_quality']}"
        )

    write_csv_atomic(args.output / "counts.csv", COUNT_FIELDS, count_rows)
    write_csv_atomic(
        args.output / "detections.csv",
        DETECTION_FIELDS,
        detection_rows,
    )


if __name__ == "__main__":
    main()
