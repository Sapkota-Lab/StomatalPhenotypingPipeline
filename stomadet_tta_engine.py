#!/usr/bin/env python3
"""StomaDet with geometric/photometric consistency and size filtering."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median, pstdev

import cv2
import numpy as np

from stomadet_engine import (
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    build_stomadet,
    infer_stomadet,
)
from stomata_engine import Detection, draw_overlay, image_paths, iou


VIEW_NAMES = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "horizontal_flip",
    "gray_clahe",
)
GEOMETRIC_VIEW_NAMES = VIEW_NAMES[:-1]


@dataclass(frozen=True)
class ViewDetection:
    view: str
    detection: Detection


@dataclass(frozen=True)
class StableDetection:
    detection: Detection
    support: int
    median_confidence: float
    minimum_confidence: float
    maximum_confidence: float
    area_ratio: float
    status: str


@dataclass(frozen=True)
class TTAResult:
    detections: list[StableDetection]
    clusters: list[list[ViewDetection]]
    view_detections: dict[str, list[Detection]]


def transform_image(image: np.ndarray, view: str) -> np.ndarray:
    if view == "identity":
        return image
    if view == "rotate90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if view == "rotate180":
        return cv2.rotate(image, cv2.ROTATE_180)
    if view == "rotate270":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if view == "horizontal_flip":
        return cv2.flip(image, 1)
    if view == "gray_clahe":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25, 25))
        normalized = clahe.apply(gray)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unknown view: {view}")


def inverse_box(
    detection: Detection,
    view: str,
    width: int,
    height: int,
) -> Detection:
    x1, y1, x2, y2 = detection.box
    if view in {"identity", "gray_clahe"}:
        values = (x1, y1, x2, y2)
    elif view == "horizontal_flip":
        values = (width - x2, y1, width - x1, y2)
    elif view == "rotate90":
        values = (y1, height - x2, y2, height - x1)
    elif view == "rotate180":
        values = (width - x2, height - y2, width - x1, height - y1)
    elif view == "rotate270":
        values = (width - y2, x1, width - y1, x2)
    else:
        raise ValueError(f"Unknown view: {view}")
    return Detection(*values, detection.confidence)


def normalized_symmetric_center_distance(
    first: Detection,
    second: Detection,
) -> float:
    first_center = np.asarray([
        (first.x1 + first.x2) / 2,
        (first.y1 + first.y2) / 2,
    ])
    second_center = np.asarray([
        (second.x1 + second.x2) / 2,
        (second.y1 + second.y2) / 2,
    ])
    scale = max(
        first.x2 - first.x1,
        first.y2 - first.y1,
        second.x2 - second.x1,
        second.y2 - second.y1,
        1e-9,
    )
    return float(np.linalg.norm(first_center - second_center) / scale)


def boxes_agree(
    first: Detection,
    second: Detection,
    iou_threshold: float,
    center_threshold: float,
) -> bool:
    overlap = iou(
        np.asarray(first.box, dtype=np.float32),
        np.asarray([second.box], dtype=np.float32),
    )[0]
    return (
        overlap >= iou_threshold
        or normalized_symmetric_center_distance(first, second)
        <= center_threshold
    )


def representative(items: list[ViewDetection]) -> Detection:
    weights = np.asarray(
        [max(item.detection.confidence, 1e-6) for item in items],
        dtype=float,
    )
    boxes = np.asarray([item.detection.box for item in items], dtype=float)
    values = np.average(boxes, axis=0, weights=weights)
    confidence = float(median(
        item.detection.confidence for item in items
    ))
    return Detection(*values.tolist(), confidence)


def cluster_views(
    detections: list[ViewDetection],
    iou_threshold: float = 0.25,
    center_threshold: float = 0.35,
) -> list[list[ViewDetection]]:
    remaining = sorted(
        detections,
        key=lambda item: item.detection.confidence,
        reverse=True,
    )
    clusters: list[list[ViewDetection]] = []
    while remaining:
        cluster = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            current = representative(cluster)
            used_views = {item.view for item in cluster}
            for view in VIEW_NAMES:
                if view in used_views:
                    continue
                candidates = [
                    (index, item)
                    for index, item in enumerate(remaining)
                    if item.view == view and boxes_agree(
                        current,
                        item.detection,
                        iou_threshold,
                        center_threshold,
                    )
                ]
                if not candidates:
                    continue
                best_index, best = max(
                    candidates,
                    key=lambda pair: pair[1].detection.confidence,
                )
                cluster.append(best)
                remaining.pop(best_index)
                changed = True
        clusters.append(cluster)
    return clusters


def classify_clusters(
    clusters: list[list[ViewDetection]],
    min_support: int,
    min_median_confidence: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
) -> list[StableDetection]:
    provisional = [
        (
            cluster,
            representative(cluster),
            [item.detection.confidence for item in cluster],
        )
        for cluster in clusters
    ]
    reference_candidates = [
        item
        for item in provisional
        if len(item[0]) >= min_support and median(item[2]) >= 0.90
    ] or [
        item for item in provisional if len(item[0]) >= min_support
    ]
    reference_area = median(
        (item[1].x2 - item[1].x1) * (item[1].y2 - item[1].y1)
        for item in reference_candidates
    ) if reference_candidates else 1.0

    result = []
    for cluster, detection, confidences in provisional:
        area = (
            (detection.x2 - detection.x1)
            * (detection.y2 - detection.y1)
        )
        area_ratio = area / max(reference_area, 1e-9)
        accepted = (
            len(cluster) >= min_support
            and median(confidences) >= min_median_confidence
            and minimum_area_ratio <= area_ratio <= maximum_area_ratio
        )
        result.append(StableDetection(
            detection=detection,
            support=len(cluster),
            median_confidence=float(median(confidences)),
            minimum_confidence=float(min(confidences)),
            maximum_confidence=float(max(confidences)),
            area_ratio=float(area_ratio),
            status="accepted" if accepted else "review",
        ))
    return result


def infer_tta_details(
    predictor,
    image: np.ndarray,
    min_view_confidence: float,
    min_support: int,
    min_median_confidence: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    agreement_iou: float,
    agreement_center_distance: float,
) -> TTAResult:
    height, width = image.shape[:2]
    detections: list[ViewDetection] = []
    view_detections: dict[str, list[Detection]] = {
        view: [] for view in VIEW_NAMES
    }
    for view in VIEW_NAMES:
        transformed = transform_image(image, view)
        for detection in infer_stomadet(predictor, transformed):
            if detection.confidence < min_view_confidence:
                continue
            restored = inverse_box(detection, view, width, height)
            detections.append(ViewDetection(view, restored))
            view_detections[view].append(restored)
    clusters = cluster_views(
        detections,
        agreement_iou,
        agreement_center_distance,
    )
    return TTAResult(
        detections=classify_clusters(
            clusters,
            min_support,
            min_median_confidence,
            minimum_area_ratio,
            maximum_area_ratio,
        ),
        clusters=clusters,
        view_detections=view_detections,
    )


def infer_tta(
    predictor,
    image: np.ndarray,
    min_view_confidence: float,
    min_support: int,
    min_median_confidence: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    agreement_iou: float,
    agreement_center_distance: float,
) -> list[StableDetection]:
    """Compatibility wrapper returning the filtered cluster detections."""
    return infer_tta_details(
        predictor,
        image,
        min_view_confidence,
        min_support,
        min_median_confidence,
        minimum_area_ratio,
        maximum_area_ratio,
        agreement_iou,
        agreement_center_distance,
    ).detections


def stability_metrics(result: TTAResult) -> dict[str, float | int]:
    """Compute annotation-free diagnostics; these do not establish accuracy."""
    counts = {
        view: len(result.view_detections[view]) for view in VIEW_NAMES
    }
    geometric_counts = [counts[view] for view in GEOMETRIC_VIEW_NAMES]
    geometric_mean = float(np.mean(geometric_counts))
    geometric_median = float(median(geometric_counts))
    count_cv = (
        float(pstdev(geometric_counts) / geometric_mean)
        if geometric_mean else 0.0
    )
    maximum_relative_deviation = (
        max(abs(value - geometric_median) for value in geometric_counts)
        / max(geometric_median, 1.0)
    )

    identity_count = counts["identity"]
    clahe_count = counts["gray_clahe"]
    identity_repeated = 0
    contrast_matches = 0
    geometric_consensus = 0
    geometric_union = 0
    for cluster in result.clusters:
        views = {item.view for item in cluster}
        geometric_support = len(views.intersection(GEOMETRIC_VIEW_NAMES))
        if geometric_support:
            geometric_union += 1
        if geometric_support >= 3:
            geometric_consensus += 1
        if "identity" in views and geometric_support >= 2:
            identity_repeated += 1
        if "identity" in views and "gray_clahe" in views:
            contrast_matches += 1

    identity_repeatability = (
        identity_repeated / identity_count if identity_count else 0.0
    )
    geometric_consensus_fraction = (
        geometric_consensus / geometric_union if geometric_union else 0.0
    )
    contrast_dice = (
        2 * contrast_matches / (identity_count + clahe_count)
        if identity_count + clahe_count else 0.0
    )
    return {
        **{f"{view}_count": counts[view] for view in VIEW_NAMES},
        "geometric_count_mean": round(geometric_mean, 6),
        "geometric_count_cv": round(count_cv, 6),
        "geometric_max_relative_deviation": round(
            maximum_relative_deviation, 6
        ),
        "identity_repeatability": round(identity_repeatability, 6),
        "geometric_consensus_fraction": round(
            geometric_consensus_fraction, 6
        ),
        "contrast_dice": round(contrast_dice, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-view-confidence", type=float, default=0.05)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-median-confidence", type=float, default=0.75)
    parser.add_argument(
        "--minimum-area-ratio",
        type=float,
        default=0.0,
        help="Optional within-image size filter; disabled by default.",
    )
    parser.add_argument(
        "--maximum-area-ratio",
        type=float,
        default=float("inf"),
        help="Optional within-image size filter; disabled by default.",
    )
    parser.add_argument("--agreement-iou", type=float, default=0.25)
    parser.add_argument("--agreement-center-distance", type=float, default=0.35)
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip rendered overlays (useful for cluster validation jobs).",
    )
    args = parser.parse_args()

    if not 1 <= args.min_support <= len(VIEW_NAMES):
        parser.error(f"min-support must be between 1 and {len(VIEW_NAMES)}")
    paths = image_paths(args.input)
    if not paths:
        parser.error(f"No supported images found in {args.input}")
    predictor = build_stomadet(
        args.config,
        args.model,
        args.min_view_confidence,
        args.device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    count_rows = []
    detection_rows = []
    stability_rows = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        result = infer_tta_details(
            predictor,
            image,
            args.min_view_confidence,
            args.min_support,
            args.min_median_confidence,
            args.minimum_area_ratio,
            args.maximum_area_ratio,
            args.agreement_iou,
            args.agreement_center_distance,
        )
        stable = result.detections
        accepted = [item for item in stable if item.status == "accepted"]
        review = [item for item in stable if item.status == "review"]
        count_rows.append({
            "filename": path.name,
            "accepted_count": len(accepted),
            "review_count": len(review),
        })
        stability_rows.append({
            "filename": path.name,
            **stability_metrics(result),
        })
        for detection_id, item in enumerate(stable, 1):
            detection_rows.append({
                "filename": path.name,
                "detection_id": detection_id,
                "status": item.status,
                "confidence": round(item.median_confidence, 6),
                "view_support": item.support,
                "minimum_confidence": round(item.minimum_confidence, 6),
                "maximum_confidence": round(item.maximum_confidence, 6),
                "area_ratio": round(item.area_ratio, 6),
                "x1": round(item.detection.x1, 2),
                "y1": round(item.detection.y1, 2),
                "x2": round(item.detection.x2, 2),
                "y2": round(item.detection.y2, 2),
            })
        if not args.no_overlays:
            draw_overlay(
                image,
                [item.detection for item in accepted],
                args.output / "overlays" / f"{path.stem}_overlay.jpg",
                [item.detection for item in review],
            )
        print(
            f"{path.name}: accepted={len(accepted)} review={len(review)}"
        )

    with (args.output / "counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "accepted_count", "review_count"],
        )
        writer.writeheader()
        writer.writerows(count_rows)
    with (args.output / "detections.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "detection_id",
                "status",
                "confidence",
                "view_support",
                "minimum_confidence",
                "maximum_confidence",
                "area_ratio",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(detection_rows)
    with (args.output / "stability.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(stability_rows[0]),
        )
        writer.writeheader()
        writer.writerows(stability_rows)


if __name__ == "__main__":
    main()
