#!/usr/bin/env python3
"""Build a domain-stratified mixed-species YOLO stomata benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from build_external_domain_dataset import add_split, load_coco, reset_output


DOMAINS = ("T-SBGH", "T-MSAA")


def domain_split_ids(
    image_ids: list[int],
    *,
    validation_images: int,
    test_images: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if validation_images <= 0 or test_images <= 0:
        raise ValueError("validation_images and test_images must be positive")
    if validation_images + test_images >= len(image_ids):
        raise ValueError("split sizes must leave at least one training image")
    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    test_ids = sorted(shuffled[:test_images])
    validation_ids = sorted(
        shuffled[test_images:test_images + validation_images]
    )
    fitting_ids = sorted(shuffled[test_images + validation_images:])
    return fitting_ids, validation_ids, test_ids


def write_merged_coco(
    selections: list[tuple[Path, list[int], str]],
    destination: Path,
) -> None:
    merged_images = []
    merged_annotations = []
    categories = None
    filenames: set[str] = set()
    next_image_id = 1
    next_annotation_id = 1
    for source_dir, selected_image_ids, domain in selections:
        document = json.loads((source_dir / "coco.json").read_text())
        categories = categories or document["categories"]
        selected = set(selected_image_ids)
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in document["annotations"]:
            annotations_by_image.setdefault(
                int(annotation["image_id"]),
                [],
            ).append(annotation)
        for image in document["images"]:
            old_id = int(image["id"])
            if old_id not in selected:
                continue
            filename = Path(image["file_name"]).name
            if filename in filenames:
                raise ValueError(f"Duplicate filename across domains: {filename}")
            filenames.add(filename)
            normalized = dict(image)
            normalized["id"] = next_image_id
            normalized["file_name"] = filename
            normalized["source_domain"] = domain
            merged_images.append(normalized)
            for annotation in annotations_by_image.get(old_id, []):
                normalized_annotation = dict(annotation)
                normalized_annotation["id"] = next_annotation_id
                normalized_annotation["image_id"] = next_image_id
                merged_annotations.append(normalized_annotation)
                next_annotation_id += 1
            next_image_id += 1
    destination.write_text(
        json.dumps(
            {
                "images": merged_images,
                "annotations": merged_annotations,
                "categories": categories or [],
            },
            indent=2,
        )
        + "\n"
    )


def build_mixed_dataset(
    *,
    benchmark: Path,
    output: Path,
    validation_images_per_domain: int,
    test_images_per_domain: int,
    seed: int,
    tile_size: int,
    tile_overlap: float,
    jpeg_quality: int,
) -> list[dict[str, object]]:
    reset_output(output)
    rows: list[dict[str, object]] = []
    training_selections = []
    validation_selections = []
    test_selections = []
    for domain_index, domain in enumerate(DOMAINS):
        source_dir = benchmark / "datasets" / domain
        images, _ = load_coco(source_dir)
        fitting_ids, validation_ids, test_ids = domain_split_ids(
            list(images),
            validation_images=validation_images_per_domain,
            test_images=test_images_per_domain,
            seed=seed + domain_index,
        )
        rows.extend(
            add_split(
                source_dir=source_dir,
                image_ids=fitting_ids,
                destination=output,
                destination_split="train",
                source_domain=domain,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                jpeg_quality=jpeg_quality,
            )
        )
        rows.extend(
            add_split(
                source_dir=source_dir,
                image_ids=validation_ids,
                destination=output,
                destination_split="val",
                source_domain=domain,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                jpeg_quality=jpeg_quality,
            )
        )
        rows.extend(
            add_split(
                source_dir=source_dir,
                image_ids=test_ids,
                destination=output,
                destination_split="test",
                source_domain=domain,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                jpeg_quality=jpeg_quality,
            )
        )
        training_selections.append((source_dir, fitting_ids, domain))
        validation_selections.append((source_dir, validation_ids, domain))
        test_selections.append((source_dir, test_ids, domain))

    write_merged_coco(training_selections, output / "train_coco.json")
    write_merged_coco(
        validation_selections,
        output / "val_coco.json",
    )
    write_merged_coco(test_selections, output / "test_coco.json")
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: stomata\n"
    )
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("external_validation/stomata_keypoint_benchmark"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-images-per-domain", type=int, default=2)
    parser.add_argument("--test-images-per-domain", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_mixed_dataset(
        benchmark=args.benchmark,
        output=args.output,
        validation_images_per_domain=args.validation_images_per_domain,
        test_images_per_domain=args.test_images_per_domain,
        seed=args.seed,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        jpeg_quality=args.jpeg_quality,
    )
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["dataset_split"] == split]
        print(
            f"{split}: {len(subset)} records from "
            f"{len({row['source_filename'] for row in subset})} source images, "
            f"{sum(int(row['objects']) for row in subset)} expert boxes"
        )


if __name__ == "__main__":
    main()
