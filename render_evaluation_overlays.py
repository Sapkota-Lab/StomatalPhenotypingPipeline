#!/usr/bin/env python3
"""Render saved detections with expert-match status for visual review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from evaluate_external_predictions import Box, box_iou, match_image


def load_truth(path: Path) -> dict[str, list[Box]]:
    coco = json.loads(path.read_text())
    image_names = {
        image["id"]: Path(image["file_name"]).name
        for image in coco["images"]
    }
    truth = {name: [] for name in image_names.values()}
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        truth[image_names[annotation["image_id"]]].append(
            Box(x, y, x + width, y + height)
        )
    return truth


def load_predictions(path: Path, confidence: float) -> dict[str, list[Box]]:
    predictions: dict[str, list[Box]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row.get("confidence", 1.0))
            if score < confidence:
                continue
            predictions.setdefault(row["filename"], []).append(Box(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                score,
            ))
    return predictions


def draw_box(image, box: Box, color, label: str) -> None:
    first = (round(box.x1), round(box.y1))
    second = (round(box.x2), round(box.y2))
    cv2.rectangle(image, first, second, color, 3)
    cv2.putText(
        image,
        label,
        (first[0], max(20, first[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def render(
    image_dir: Path,
    truth: dict[str, list[Box]],
    predictions: dict[str, list[Box]],
    output: Path,
    iou_threshold: float,
) -> list[dict[str, int | str]]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for filename in sorted(truth):
        image_path = image_dir / filename
        if not image_path.exists():
            matches = list(image_dir.rglob(filename))
            if not matches:
                raise FileNotFoundError(image_path)
            image_path = matches[0]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {image_path}")
        expected = truth[filename]
        predicted = predictions.get(filename, [])
        matched = match_image(predicted, expected, iou_threshold)
        matched_truth = set()
        for prediction, is_true_positive, _iou in matched:
            if is_true_positive:
                color = (0, 190, 0)  # green: matched prediction
                # Recover the matched truth box for a visual audit.
                candidates = [
                    (index, box_iou(prediction, candidate))
                    for index, candidate in enumerate(expected)
                    if index not in matched_truth
                ]
                if candidates:
                    best_index, best_iou = max(candidates, key=lambda item: item[1])
                    if best_iou >= iou_threshold:
                        matched_truth.add(best_index)
                label = f"TP {prediction.score:.2f}"
            else:
                color = (0, 0, 220)  # red: unmatched prediction
                label = f"FP {prediction.score:.2f}"
            draw_box(image, prediction, color, label)
        for index, box in enumerate(expected):
            if index not in matched_truth:
                draw_box(image, box, (0, 215, 255), "FN expert")
            else:
                draw_box(image, box, (255, 120, 0), "expert")
        cv2.imwrite(str(output / f"{Path(filename).stem}_audit.jpg"), image)
        rows.append({
            "filename": filename,
            "ground_truth": len(expected),
            "predictions": len(predicted),
            "true_positive": sum(is_tp for _box, is_tp, _iou in matched),
            "false_positive": sum(not is_tp for _box, is_tp, _iou in matched),
            "false_negative": len(expected) - len(matched_truth),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    rows = render(
        args.images,
        load_truth(args.ground_truth),
        load_predictions(args.predictions, args.confidence),
        args.output,
        args.iou,
    )
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} audit overlays to {args.output}")


if __name__ == "__main__":
    main()
