#!/usr/bin/env python3
"""Download a deterministic, expert-labelled StomaDet split subset.

The complete public archive is about 1.4 GB. This downloader uses HTTP range
requests to read the ZIP remotely and retrieves only a quality-stratified
subset of the official held-out test split plus its COCO annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from remotezip import RemoteZip


DATASET_DOI = "10.17632/2bfcpc6bpj.1"
DATASET_URL = (
    "https://data.mendeley.com/public-files/datasets/2bfcpc6bpj/files/"
    "c70645d1-4bbf-4343-9c51-f943f4fd02bf/file_downloaded"
)
QUALITY_ORDER = ("imperfect", "default", "out_of_focus")


def read_single_column_csv(payload: bytes, column: str) -> list[str]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if column not in (reader.fieldnames or []):
        raise ValueError(f"CSV does not contain {column!r}")
    return [row[column].strip() for row in reader if row[column].strip()]


def read_quality_map(payload: bytes) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    required = {"file_name", "image_type"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("dataset.csv lacks file_name or image_type")
    return {
        row["file_name"].strip(): row["image_type"].strip()
        for row in reader
    }


def stratified_selection(
    split_names: list[str],
    quality_by_name: dict[str, str],
    per_quality: int,
    seed: int,
) -> list[tuple[str, str]]:
    if per_quality <= 0:
        raise ValueError("per_quality must be positive")
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for quality in QUALITY_ORDER:
        candidates = sorted(
            name for name in split_names if quality_by_name.get(name) == quality
        )
        if len(candidates) < per_quality:
            raise ValueError(
                f"Only {len(candidates)} {quality!r} test images; "
                f"cannot select {per_quality}"
            )
        for name in sorted(rng.sample(candidates, per_quality)):
            selected.append((name, quality))
    return selected


def filter_coco(coco: dict, selected_stems: set[str]) -> dict:
    images = [
        item
        for item in coco["images"]
        if Path(item["file_name"]).stem in selected_stems
    ]
    image_ids = {item["id"] for item in images}
    annotations = [
        item for item in coco["annotations"] if item["image_id"] in image_ids
    ]
    return {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    } | {
        "images": images,
        "annotations": annotations,
    }


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_if_changed(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def encode_jpeg(png_payload: bytes, quality: int) -> tuple[bytes, int, int]:
    image = cv2.imdecode(
        np.frombuffer(png_payload, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("Could not decode source PNG")
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise ValueError("Could not encode validation JPEG")
    height, width = image.shape[:2]
    return encoded.tobytes(), width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("external_validation/stomadet_test_subset"),
    )
    parser.add_argument("--per-quality", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="official StomaDet split to sample",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="recompress source PNGs to conserve local disk space",
    )
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be between 1 and 100")
    if shutil.disk_usage(Path.cwd()).free < 64 * 1024 * 1024:
        parser.error("At least 64 MB free disk space is required")

    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    images_dir.mkdir(exist_ok=True)

    with RemoteZip(DATASET_URL) as archive:
        split_names = read_single_column_csv(
            archive.read(f"data/splits/{args.split}.csv"),
            "file_name",
        )
        quality_by_name = read_quality_map(archive.read("data/dataset.csv"))
        selected = stratified_selection(
            split_names,
            quality_by_name,
            args.per_quality,
            args.seed,
        )
        selected_stems = {name for name, _quality in selected}
        full_coco = json.loads(
            archive.read(f"data/splits/{args.split}.json").decode("utf-8")
        )
        subset_coco = filter_coco(full_coco, selected_stems)

        manifest_rows: list[dict[str, str | int]] = []
        annotations_by_image: dict[int, int] = {}
        for annotation in subset_coco["annotations"]:
            image_id = annotation["image_id"]
            annotations_by_image[image_id] = (
                annotations_by_image.get(image_id, 0) + 1
            )
        coco_image_by_stem = {
            Path(item["file_name"]).stem: item
            for item in subset_coco["images"]
        }

        for name, quality in selected:
            member = f"data/images/{name}.png"
            source_payload = archive.read(member)
            payload, width, height = encode_jpeg(
                source_payload,
                args.jpeg_quality,
            )
            destination = images_dir / f"{name}.jpg"
            write_if_changed(destination, payload)
            image_record = coco_image_by_stem[name]
            if (
                image_record["width"] != width
                or image_record["height"] != height
            ):
                raise ValueError(f"Dimension mismatch for {name}")
            image_record["file_name"] = destination.name
            manifest_rows.append({
                "filename": destination.name,
                "quality_stratum": quality,
                "ground_truth_count": annotations_by_image.get(
                    image_record["id"],
                    0,
                ),
                "source_sha256": sha256(source_payload),
                "local_sha256": sha256(payload),
            })

    ground_truth_payload = (
        json.dumps(subset_coco, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    write_if_changed(args.output / "ground_truth.json", ground_truth_payload)

    manifest_buffer = io.StringIO()
    writer = csv.DictWriter(
        manifest_buffer,
        fieldnames=[
            "filename",
            "quality_stratum",
            "ground_truth_count",
            "source_sha256",
            "local_sha256",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(manifest_rows)
    write_if_changed(
        args.output / "manifest.csv",
        manifest_buffer.getvalue().encode("utf-8"),
    )

    metadata = {
        "dataset_doi": DATASET_DOI,
        "source_url": DATASET_URL,
        "official_split": args.split,
        "selection": "deterministic random sample within image_type",
        "seed": args.seed,
        "per_quality": args.per_quality,
        "local_storage_format": "JPEG",
        "jpeg_quality": args.jpeg_quality,
        "images": len(manifest_rows),
        "ground_truth_objects": len(subset_coco["annotations"]),
        "quality_strata": list(QUALITY_ORDER),
    }
    write_if_changed(
        args.output / "metadata.json",
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        f"Downloaded {metadata['images']} official {args.split} images with "
        f"{metadata['ground_truth_objects']} expert-labelled stomata to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
