#!/usr/bin/env python3
"""Build full-image COCO references from expert boxes and fixed splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from box_annotator import validate_boxes


def split_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} needs filename and split columns")
        result = {}
        for row in reader:
            filename = row["filename"]
            if filename in result:
                raise ValueError(f"Duplicate split row: {filename}")
            result[filename] = row["split"]
    return result


def build_references(
    annotations_dir: Path,
    splits: dict[str, str],
) -> dict[str, dict]:
    references = {
        split: {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "stoma"}],
        }
        for split in ("train", "val", "test")
    }
    next_image_id = {split: 1 for split in references}
    next_annotation_id = {split: 1 for split in references}
    seen = set()
    for filename, split in sorted(splits.items()):
        if split not in references:
            raise ValueError(f"Unknown split {split!r} for {filename}")
        path = annotations_dir / f"{Path(filename).stem}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"{filename} is not marked complete")
        if Path(payload["image"]).name != filename:
            raise ValueError(f"Annotation filename mismatch for {filename}")
        width = int(payload["width"])
        height = int(payload["height"])
        boxes = validate_boxes(payload.get("boxes", []), width, height)
        image_id = next_image_id[split]
        next_image_id[split] += 1
        references[split]["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        for box in boxes:
            x1, y1, x2, y2 = (
                float(box[field])
                for field in ("x1", "y1", "x2", "y2")
            )
            box_width = x2 - x1
            box_height = y2 - y1
            references[split]["annotations"].append({
                "id": next_annotation_id[split],
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            })
            next_annotation_id[split] += 1
        seen.add(filename)
    if seen != set(splits):
        raise ValueError("Not every split image received a reference")
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("box_annotations"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("training_data/annotated_v1/split_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/annotated_v1/full_image_reference"),
    )
    args = parser.parse_args()
    references = build_references(
        args.annotations,
        split_map(args.split_manifest),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for split, reference in references.items():
        output = args.output / f"{split}_coco.json"
        output.write_text(json.dumps(reference, indent=2) + "\n")
        print(
            f"{split}: {len(reference['images'])} images, "
            f"{len(reference['annotations'])} stomata -> {output}"
        )


if __name__ == "__main__":
    main()
