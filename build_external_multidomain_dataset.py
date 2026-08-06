#!/usr/bin/env python3
"""Build a balanced leave-one-domain-out COCO stomata dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomainSource:
    name: str
    directory: Path
    image_directory: Path

    @property
    def coco_path(self) -> Path:
        return self.directory / "coco.json"


def default_sources(project_root: Path) -> list[DomainSource]:
    benchmark = (
        project_root
        / "external_validation"
        / "stomata_keypoint_benchmark"
        / "datasets"
    )
    hub = project_root / "external_validation" / "stomatahub_raw"
    sources = [
        DomainSource(name, benchmark / name, benchmark / name / "images")
        for name in ("T-SBGH", "T-MSAA")
    ]
    sources.extend(
        DomainSource(name, hub / name, hub / name)
        for name in (
            "Gibbs2021_Poplar",
            "Kwong2021_OilPalm",
            "Sultana2021Soybean",
            "Casado-Garcia2020_LabelStoma",
        )
    )
    return sources


def split_and_balance(
    image_ids: list[int],
    *,
    validation_images: int,
    training_records: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if validation_images <= 0:
        raise ValueError("validation_images must be positive")
    if validation_images >= len(image_ids):
        raise ValueError("validation split must leave training images")
    if training_records <= 0:
        raise ValueError("training_records must be positive")
    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    validation = sorted(shuffled[:validation_images])
    candidates = shuffled[validation_images:]
    balanced = [
        candidates[index % len(candidates)]
        for index in range(training_records)
    ]
    return balanced, validation


def load_source(source: DomainSource) -> tuple[dict, dict[int, list[dict]], int]:
    if not source.coco_path.exists():
        raise FileNotFoundError(source.coco_path)
    document = json.loads(source.coco_path.read_text())
    category_by_id = {
        int(item["id"]): item["name"].strip().lower()
        for item in document["categories"]
    }
    stomata_ids = {
        category_id
        for category_id, name in category_by_id.items()
        if name in {"stoma", "stomata"}
    }
    if not stomata_ids:
        raise ValueError(f"{source.coco_path} has no stomata category")
    annotations: dict[int, list[dict]] = defaultdict(list)
    ignored = 0
    for annotation in document["annotations"]:
        if int(annotation["category_id"]) not in stomata_ids:
            ignored += 1
            continue
        annotations[int(annotation["image_id"])].append(annotation)
    return document, dict(annotations), ignored


def safe_domain(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name)


def reset_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "full_images" / split).mkdir(parents=True, exist_ok=True)


def build_dataset(
    *,
    project_root: Path,
    output: Path,
    heldout_domain: str,
    validation_images_per_domain: int,
    training_records_per_domain: int,
    seed: int,
) -> dict:
    sources = default_sources(project_root)
    source_by_name = {source.name: source for source in sources}
    if heldout_domain not in source_by_name:
        raise ValueError(f"Unknown heldout domain {heldout_domain!r}")
    reset_output(output)
    documents = {}
    annotations_by_source = {}
    ignored_by_source = {}
    for source in sources:
        document, annotations, ignored = load_source(source)
        documents[source.name] = document
        annotations_by_source[source.name] = annotations
        ignored_by_source[source.name] = ignored

    selections: dict[str, list[tuple[DomainSource, int]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for domain_index, source in enumerate(sources):
        image_ids = [int(image["id"]) for image in documents[source.name]["images"]]
        if source.name == heldout_domain:
            selections["test"].extend((source, image_id) for image_id in image_ids)
            continue
        training_ids, validation_ids = split_and_balance(
            image_ids,
            validation_images=validation_images_per_domain,
            training_records=training_records_per_domain,
            seed=seed + domain_index,
        )
        selections["train"].extend((source, image_id) for image_id in training_ids)
        selections["val"].extend((source, image_id) for image_id in validation_ids)

    manifests = []
    summary = {
        "heldout_domain": heldout_domain,
        "seed": seed,
        "validation_images_per_training_domain": validation_images_per_domain,
        "training_records_per_domain": training_records_per_domain,
        "ignored_non_stomata_annotations": ignored_by_source,
        "splits": {},
    }
    for split, selected in selections.items():
        images = []
        annotations = []
        next_annotation_id = 1
        image_lookup = {
            source.name: {
                int(image["id"]): image
                for image in documents[source.name]["images"]
            }
            for source in sources
        }
        occurrence: Counter[tuple[str, int]] = Counter()
        source_counts: Counter[str] = Counter()
        source_unique: dict[str, set[int]] = defaultdict(set)
        source_boxes: Counter[str] = Counter()
        for new_image_id, (source, old_image_id) in enumerate(selected, 1):
            image = image_lookup[source.name][old_image_id]
            original_filename = Path(image["file_name"]).name
            source_image = source.image_directory / original_filename
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            target_filename = f"{safe_domain(source.name)}__{original_filename}"
            target_image = output / "full_images" / split / target_filename
            if not target_image.exists():
                target_image.symlink_to(source_image.resolve())
            occurrence[(source.name, old_image_id)] += 1
            normalized_image = dict(image)
            normalized_image.update({
                "id": new_image_id,
                "file_name": target_filename,
                "source_domain": source.name,
                "source_image_id": old_image_id,
                "source_filename": original_filename,
                "training_replicate": occurrence[(source.name, old_image_id)],
            })
            images.append(normalized_image)
            image_annotations = annotations_by_source[source.name].get(
                old_image_id,
                [],
            )
            for annotation in image_annotations:
                normalized_annotation = dict(annotation)
                normalized_annotation.update({
                    "id": next_annotation_id,
                    "image_id": new_image_id,
                    "category_id": 1,
                    "source_domain": source.name,
                })
                annotations.append(normalized_annotation)
                next_annotation_id += 1
            source_counts[source.name] += 1
            source_unique[source.name].add(old_image_id)
            source_boxes[source.name] += len(image_annotations)
            manifests.append({
                "dataset_split": split,
                "source_domain": source.name,
                "quality_stratum": source.name,
                "source_image_id": old_image_id,
                "source_filename": original_filename,
                "filename": target_filename,
                "dataset_filename": target_filename,
                "replicate": occurrence[(source.name, old_image_id)],
                "objects": len(image_annotations),
            })
        document = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "stomata"}],
        }
        (output / f"{split}_coco.json").write_text(
            json.dumps(document, indent=2) + "\n"
        )
        summary["splits"][split] = {
            domain: {
                "records": source_counts[domain],
                "unique_source_images": len(source_unique[domain]),
                "stomata_boxes": source_boxes[domain],
            }
            for domain in sorted(source_counts)
        }

    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifests[0]))
        writer.writeheader()
        writer.writerows(manifests)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--heldout-domain",
        choices=("T-SBGH", "T-MSAA"),
        required=True,
    )
    parser.add_argument("--validation-images-per-domain", type=int, default=3)
    parser.add_argument("--training-records-per-domain", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    summary = build_dataset(
        project_root=args.project_root.resolve(),
        output=args.output,
        heldout_domain=args.heldout_domain,
        validation_images_per_domain=args.validation_images_per_domain,
        training_records_per_domain=args.training_records_per_domain,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
