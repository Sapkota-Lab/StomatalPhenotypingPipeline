#!/usr/bin/env python3
"""Build a leakage-safe tiled YOLO dataset from completed box annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from box_annotator import validate_boxes


EXCLUDED_METADATA_GROUP = "__excluded_from_analysis__"


@dataclass(frozen=True)
class AnnotatedImage:
    filename: str
    group: str
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_group_splits(
    groups: list[str], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    unique = sorted(set(groups), key=lambda value: stable_key(value, seed))
    if len(unique) < 3:
        raise ValueError("At least three independent groups are required")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be positive and sum to less than one")
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = max(1, round(len(unique) * val_fraction))
    while n_test + n_val >= len(unique):
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Not enough groups for non-empty train/val/test splits")
    result = {}
    for index, group in enumerate(unique):
        if index < n_test:
            result[group] = "test"
        elif index < n_test + n_val:
            result[group] = "val"
        else:
            result[group] = "train"
    return result


def tile_origin(center: float, length: int, tile_size: int) -> int:
    if length <= tile_size:
        return 0
    return max(0, min(round(center - tile_size / 2), length - tile_size))


def grid_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    values = list(range(0, length - tile_size + 1, stride))
    if values[-1] != length - tile_size:
        values.append(length - tile_size)
    return values


def boxes_in_window(
    boxes: tuple[tuple[float, float, float, float], ...],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    labels = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (x0 <= cx < x0 + width and y0 <= cy < y0 + height):
            continue
        clipped_x1 = max(x0, x1)
        clipped_y1 = max(y0, y1)
        clipped_x2 = min(x0 + width, x2)
        clipped_y2 = min(y0 + height, y2)
        if clipped_x2 > clipped_x1 and clipped_y2 > clipped_y1:
            labels.append((clipped_x1, clipped_y1, clipped_x2, clipped_y2))
    return labels


def yolo_line(
    box: tuple[float, float, float, float],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2 - x0) / width
    cy = ((y1 + y2) / 2 - y0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def load_metadata(path: Path, group_column: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "filename" not in (reader.fieldnames or []):
            raise ValueError("metadata CSV needs a filename column")
        if group_column not in (reader.fieldnames or []):
            raise ValueError(f"metadata CSV needs a {group_column} column")
        result = {}
        for row in reader:
            filename = row.get("filename", "")
            if not filename:
                continue
            mapping_status = row.get("mapping_status", "").strip()
            if mapping_status and mapping_status != "mapped":
                result[filename] = EXCLUDED_METADATA_GROUP
            else:
                result[filename] = row.get(group_column, "").strip()
        return result


def load_annotations(
    annotations_dir: Path,
    metadata: dict[str, str],
    allow_image_groups: bool,
    include_provisional: bool = False,
) -> list[AnnotatedImage]:
    records = []
    missing_groups = []
    for path in sorted(annotations_dir.glob("*.json")):
        with path.open() as handle:
            payload = json.load(handle)
        allowed_statuses = (
            {"complete", "provisional"} if include_provisional else {"complete"}
        )
        if payload.get("status") not in allowed_statuses:
            continue
        filename = str(payload["image"])
        width, height = int(payload["width"]), int(payload["height"])
        clean = validate_boxes(payload.get("boxes", []), width, height)
        group = metadata.get(filename, "")
        if group == EXCLUDED_METADATA_GROUP:
            continue
        if not group:
            if not allow_image_groups:
                missing_groups.append(filename)
                continue
            group = filename
        records.append(
            AnnotatedImage(
                filename=filename,
                group=group,
                width=width,
                height=height,
                boxes=tuple(
                    (box["x1"], box["y1"], box["x2"], box["y2"]) for box in clean
                ),
            )
        )
    if missing_groups:
        examples = ", ".join(missing_groups[:5])
        raise ValueError(
            f"Completed images are missing grouping metadata: {examples}. "
            "Fill the requested group column or explicitly use --allow-image-groups."
        )
    if not records:
        raise ValueError("No completed annotations were found")
    return records


def build_windows(
    record: AnnotatedImage,
    tile_size: int,
    overlap: float,
    negative_ratio: float,
    zero_image_negatives: int,
    seed: int,
) -> list[tuple[int, int]]:
    positive = {
        (
            tile_origin((box[0] + box[2]) / 2, record.width, tile_size),
            tile_origin((box[1] + box[3]) / 2, record.height, tile_size),
        )
        for box in record.boxes
    }
    grid = [
        (x0, y0)
        for y0 in grid_origins(record.height, tile_size, overlap)
        for x0 in grid_origins(record.width, tile_size, overlap)
    ]
    negatives = [
        origin
        for origin in grid
        if not boxes_in_window(
            record.boxes,
            origin[0],
            origin[1],
            min(tile_size, record.width),
            min(tile_size, record.height),
        )
    ]
    rng = random.Random(f"{seed}:{record.filename}")
    rng.shuffle(negatives)
    negative_count = (
        min(len(negatives), zero_image_negatives)
        if not positive
        else min(len(negatives), math.ceil(len(positive) * negative_ratio))
    )
    return sorted(positive) + sorted(negatives[:negative_count])


def build_dataset(args: argparse.Namespace) -> dict:
    metadata = load_metadata(args.metadata, args.group_column)
    records = load_annotations(
        args.annotations,
        metadata,
        args.allow_image_groups,
        args.include_provisional,
    )
    split_by_group = assign_group_splits(
        [record.group for record in records],
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )
    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    summary = {
        "source_images": 0,
        "stomata": 0,
        "tiles": 0,
        "positive_tiles": 0,
        "negative_tiles": 0,
        "splits": {
            split: {"groups": set(), "source_images": 0, "stomata": 0, "tiles": 0}
            for split in ("train", "val", "test")
        },
    }
    manifest_rows = []
    for record in records:
        source_path = args.images / record.filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        split = split_by_group[record.group]
        with Image.open(source_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.size != (record.width, record.height):
                raise ValueError(
                    f"{record.filename}: annotation dimensions {record.width}x"
                    f"{record.height} do not match oriented image {image.width}x{image.height}"
                )
            windows = build_windows(
                record,
                args.tile_size,
                args.overlap,
                args.negative_ratio,
                args.zero_image_negatives,
                args.seed,
            )
            for x0, y0 in windows:
                width = min(args.tile_size, image.width)
                height = min(args.tile_size, image.height)
                labels = boxes_in_window(record.boxes, x0, y0, width, height)
                stem = f"{Path(record.filename).stem}_{x0}_{y0}"
                image.crop((x0, y0, x0 + width, y0 + height)).save(
                    args.output / "images" / split / f"{stem}.jpg",
                    quality=95,
                )
                label_text = "\n".join(
                    yolo_line(box, x0, y0, width, height) for box in labels
                )
                if label_text:
                    label_text += "\n"
                (args.output / "labels" / split / f"{stem}.txt").write_text(
                    label_text
                )
                summary["tiles"] += 1
                summary["splits"][split]["tiles"] += 1
                if labels:
                    summary["positive_tiles"] += 1
                else:
                    summary["negative_tiles"] += 1
        summary["source_images"] += 1
        summary["stomata"] += len(record.boxes)
        summary["splits"][split]["groups"].add(record.group)
        summary["splits"][split]["source_images"] += 1
        summary["splits"][split]["stomata"] += len(record.boxes)
        manifest_rows.append(
            {
                "filename": record.filename,
                "group": record.group,
                "split": split,
                "stomata": len(record.boxes),
                "tiles": len(windows),
            }
        )

    yaml = (
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: stoma\n"
    )
    (args.output / "dataset.yaml").write_text(yaml)
    with (args.output / "split_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "group", "split", "stomata", "tiles"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    for split in summary["splits"].values():
        split["groups"] = len(split["groups"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--annotations", type=Path, default=Path("box_annotations"))
    parser.add_argument("--metadata", type=Path, default=Path("image_metadata.csv"))
    parser.add_argument("--group-column", default="specimen_id")
    parser.add_argument("--output", type=Path, default=Path("training_data/annotated_v1"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--negative-ratio", type=float, default=0.5)
    parser.add_argument("--zero-image-negatives", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--allow-image-groups",
        action="store_true",
        help="Use filename as group when metadata is missing (pilot only; risks biological leakage)",
    )
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="Include Codex-provisional boxes for experiments; never use for final validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
