#!/usr/bin/env python3
"""Train and apply an orientation-robust stomatal morphology verifier.

The verifier learns the paired-guard-cell/pore appearance from expert COCO
boxes and hard background crops. It is intentionally separate from the object
detector so its score can serve as an independent acceptance signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_VERSION = "hog_gray_structure_v1"
PATCH_SIZE = 64
CONTEXT_SCALE = 1.35
SCORE_VIEWS = ("identity", "rotate90", "rotate180", "rotate270", "hflip")
HOG = cv2.HOGDescriptor(
    (PATCH_SIZE, PATCH_SIZE),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)


def box_iou(first: tuple[float, float, float, float],
            second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    return intersection / max(first_area + second_area - intersection, 1e-9)


def expanded_crop(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    scale: float = CONTEXT_SCALE,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = max(4.0, (x2 - x1) * scale)
    crop_height = max(4.0, (y2 - y1) * scale)
    left = max(0, int(round(center_x - crop_width / 2)))
    top = max(0, int(round(center_y - crop_height / 2)))
    right = min(width, int(round(center_x + crop_width / 2)))
    bottom = min(height, int(round(center_y + crop_height / 2)))
    if right <= left or bottom <= top:
        raise ValueError(f"Degenerate crop for box {box}")
    return image[top:bottom, left:right]


def normalize_patch(crop: np.ndarray) -> np.ndarray:
    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop.ndim == 3 else crop
    )
    resized = cv2.resize(
        gray,
        (PATCH_SIZE, PATCH_SIZE),
        interpolation=cv2.INTER_AREA
        if max(gray.shape[:2]) >= PATCH_SIZE else cv2.INTER_CUBIC,
    )
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(resized)


def transform_patch(patch: np.ndarray, view: str) -> np.ndarray:
    if view == "identity":
        return patch
    if view == "rotate90":
        return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    if view == "rotate180":
        return cv2.rotate(patch, cv2.ROTATE_180)
    if view == "rotate270":
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if view == "hflip":
        return cv2.flip(patch, 1)
    raise ValueError(f"Unknown patch view: {view}")


def patch_features(patch: np.ndarray) -> np.ndarray:
    normalized = patch.astype(np.float32) / 255.0
    hog = HOG.compute(patch).reshape(-1).astype(np.float32)
    thumbnail = cv2.resize(
        normalized,
        (16, 16),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1)
    histogram = cv2.calcHist([patch], [0], None, [16], [0, 256]).reshape(-1)
    histogram = histogram.astype(np.float32) / max(float(histogram.sum()), 1.0)
    edges = cv2.Canny(patch, 40, 120)
    horizontal_symmetry = np.mean(
        np.abs(normalized - np.fliplr(normalized))
    )
    vertical_symmetry = np.mean(
        np.abs(normalized - np.flipud(normalized))
    )
    center = normalized[20:44, 20:44]
    border_mask = np.ones_like(normalized, dtype=bool)
    border_mask[12:52, 12:52] = False
    structure = np.asarray(
        [
            float(edges.mean() / 255.0),
            float(cv2.Laplacian(patch, cv2.CV_32F).var() / 10000.0),
            float(horizontal_symmetry),
            float(vertical_symmetry),
            float(center.mean() - normalized[border_mask].mean()),
            float(center.std()),
        ],
        dtype=np.float32,
    )
    return np.concatenate((hog, thumbnail, histogram, structure))


def view_features(crop: np.ndarray) -> list[np.ndarray]:
    patch = normalize_patch(crop)
    return [
        patch_features(transform_patch(patch, view))
        for view in SCORE_VIEWS
    ]


def load_coco(path: Path) -> tuple[
    dict[str, list[tuple[float, float, float, float]]],
    dict[str, tuple[int, int]],
]:
    coco = json.loads(path.read_text())
    filename_by_id = {
        int(item["id"]): item["file_name"] for item in coco["images"]
    }
    dimensions = {
        item["file_name"]: (int(item["width"]), int(item["height"]))
        for item in coco["images"]
    }
    boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        boxes[filename_by_id[int(annotation["image_id"])]].append(
            (float(x), float(y), float(x + width), float(y + height))
        )
    return dict(boxes), dimensions


def hard_negative_boxes(
    image: np.ndarray,
    positives: list[tuple[float, float, float, float]],
    count: int,
    rng: random.Random,
) -> list[tuple[float, float, float, float]]:
    height, width = image.shape[:2]
    if not positives or count <= 0:
        return []
    candidates = []
    attempts = max(200, count * 30)
    for _ in range(attempts):
        template = rng.choice(positives)
        scale = rng.uniform(0.75, 1.35)
        box_width = max(8.0, (template[2] - template[0]) * scale)
        box_height = max(8.0, (template[3] - template[1]) * scale)
        if box_width >= width or box_height >= height:
            continue
        x1 = rng.uniform(0, width - box_width)
        y1 = rng.uniform(0, height - box_height)
        candidate = (x1, y1, x1 + box_width, y1 + box_height)
        if max(box_iou(candidate, item) for item in positives) > 0.02:
            continue
        crop = expanded_crop(image, candidate)
        gray = normalize_patch(crop)
        edge_density = float(cv2.Canny(gray, 40, 120).mean())
        contrast = float(gray.std())
        candidates.append((edge_density + contrast, candidate))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [box for _score, box in candidates[:count]]


def training_samples(
    images_dir: Path,
    annotations: Path,
    negatives_per_positive: float,
    seed: int,
    hard_negative_predictions: Path | None = None,
    hard_negative_repeats: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    positives_by_file, _dimensions = load_coco(annotations)
    mined_by_file: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    if hard_negative_predictions:
        with hard_negative_predictions.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row["filename"]
                if filename not in positives_by_file:
                    continue
                candidate = tuple(
                    float(row[key]) for key in ("x1", "y1", "x2", "y2")
                )
                maximum_iou = max(
                    box_iou(candidate, positive)
                    for positive in positives_by_file[filename]
                )
                if maximum_iou < 0.20:
                    mined_by_file[filename].append(candidate)
    rng = random.Random(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []
    base_positive = 0
    base_negative = 0
    for filename in sorted(positives_by_file):
        image = cv2.imread(str(images_dir / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {images_dir / filename}")
        positive_boxes = positives_by_file[filename]
        negative_count = round(len(positive_boxes) * negatives_per_positive)
        negative_boxes = hard_negative_boxes(
            image,
            positive_boxes,
            negative_count,
            rng,
        )
        negative_boxes.extend(
            mined_by_file.get(filename, []) * hard_negative_repeats
        )
        base_positive += len(positive_boxes)
        base_negative += len(negative_boxes)
        for label, boxes in ((1, positive_boxes), (0, negative_boxes)):
            for box in boxes:
                crop = expanded_crop(image, box)
                for values in view_features(crop):
                    features.append(values)
                    labels.append(label)
    metadata = {
        "feature_version": FEATURE_VERSION,
        "patch_size": PATCH_SIZE,
        "context_scale": CONTEXT_SCALE,
        "score_views": list(SCORE_VIEWS),
        "source_images": len(positives_by_file),
        "base_positive_crops": base_positive,
        "base_negative_crops": base_negative,
        "detector_mined_negative_crops": sum(
            len(items) for items in mined_by_file.values()
        ),
        "hard_negative_repeats": hard_negative_repeats,
        "augmented_samples": len(labels),
        "seed": seed,
    }
    return np.asarray(features, dtype=np.float32), np.asarray(labels), metadata


def fit_model(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            C=0.25,
            solver="liblinear",
            random_state=0,
        ),
    )
    model.fit(features, labels)
    return model


def score_crop(model, crop: np.ndarray) -> tuple[float, float]:
    probabilities = model.predict_proba(
        np.asarray(view_features(crop), dtype=np.float32)
    )[:, 1]
    return float(np.median(probabilities)), float(np.min(probabilities))


def score_csv(
    images_dir: Path,
    predictions_path: Path,
    model_payload: dict,
    output: Path,
) -> None:
    model = model_payload["model"]
    output.parent.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, np.ndarray] = {}
    with predictions_path.open(newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        required = {"filename", "x1", "y1", "x2", "y2"}
        if not required.issubset(fieldnames):
            raise ValueError("Prediction CSV lacks coordinate columns")
        output_fields = fieldnames + [
            "morphology_confidence",
            "morphology_min_confidence",
        ]
        with output.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=output_fields)
            writer.writeheader()
            for row in reader:
                filename = row["filename"]
                if filename not in image_cache:
                    image_cache[filename] = cv2.imread(
                        str(images_dir / filename),
                        cv2.IMREAD_COLOR,
                    )
                    if image_cache[filename] is None:
                        raise ValueError(f"Could not read {images_dir / filename}")
                box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
                crop = expanded_crop(image_cache[filename], box)
                score, minimum = score_crop(model, crop)
                row["morphology_confidence"] = f"{score:.6f}"
                row["morphology_min_confidence"] = f"{minimum:.6f}"
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--images", type=Path, required=True)
    train.add_argument("--annotations", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--negatives-per-positive", type=float, default=2.0)
    train.add_argument(
        "--hard-negative-predictions",
        type=Path,
        help="Detector proposals with IoU <0.20 to every expert box become negatives.",
    )
    train.add_argument("--hard-negative-repeats", type=int, default=8)
    train.add_argument("--seed", type=int, default=20260726)

    score = subparsers.add_parser("score")
    score.add_argument("--images", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "train":
        features, labels, metadata = training_samples(
            args.images,
            args.annotations,
            args.negatives_per_positive,
            args.seed,
            args.hard_negative_predictions,
            args.hard_negative_repeats,
        )
        model = fit_model(features, labels)
        probabilities = model.predict_proba(features)[:, 1]
        predictions = probabilities >= 0.5
        precision, recall, f1, _support = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        metadata["training_metrics_resubstitution_only"] = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "average_precision": round(
                float(average_precision_score(labels, probabilities)), 6
            ),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "metadata": metadata}, args.output)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
    else:
        payload = joblib.load(args.model)
        if payload["metadata"]["feature_version"] != FEATURE_VERSION:
            raise ValueError("Incompatible morphology feature version")
        score_csv(args.images, args.predictions, payload, args.output)
        print(f"Wrote morphology scores to {args.output}")


if __name__ == "__main__":
    main()
