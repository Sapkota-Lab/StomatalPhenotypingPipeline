#!/usr/bin/env python3
"""Convert paired StomataHub XML/JPEG annotations to validated COCO boxes."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


CATEGORY_IDS = {"stomata": 1, "pore": 2}


def parse_box(
    node: ET.Element,
    image_width: int,
    image_height: int,
) -> tuple[list[float] | None, bool]:
    values = {
        name: float(node.findtext(name, "nan"))
        for name in ("xmin", "ymin", "xmax", "ymax")
    }
    x1 = min(max(values["xmin"], 0.0), float(image_width))
    y1 = min(max(values["ymin"], 0.0), float(image_height))
    x2 = min(max(values["xmax"], 0.0), float(image_width))
    y2 = min(max(values["ymax"], 0.0), float(image_height))
    clipped = (
        x1 != values["xmin"]
        or y1 != values["ymin"]
        or x2 != values["xmax"]
        or y2 != values["ymax"]
        or x1 <= 0
        or y1 <= 0
        or x2 >= image_width
        or y2 >= image_height
    )
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None, clipped
    return [x1, y1, width, height], clipped


def convert(source: Path, domain: str) -> tuple[dict, dict]:
    xml_paths = sorted(source.glob("*.xml"))
    image_paths = sorted(source.glob("*.jpg"))
    if not xml_paths or not image_paths:
        raise ValueError(f"{source} must contain paired XML and JPEG files")
    image_by_stem = {path.stem: path for path in image_paths}
    xml_stems = {path.stem for path in xml_paths}
    if set(image_by_stem) != xml_stems:
        missing_xml = sorted(set(image_by_stem) - xml_stems)
        missing_image = sorted(xml_stems - set(image_by_stem))
        raise ValueError(
            f"Unpaired files: missing XML={missing_xml[:5]}, "
            f"missing image={missing_image[:5]}"
        )

    images = []
    annotations = []
    clipped_boxes = 0
    rejected_boxes = 0
    class_counts: Counter[str] = Counter()
    filename_metadata_mismatches = 0
    dimension_metadata_mismatches = 0
    annotation_id = 1
    for image_id, xml_path in enumerate(xml_paths, 1):
        root = ET.parse(xml_path).getroot()
        source_filename = Path(root.findtext("filename", "")).name
        image_path = image_by_stem[xml_path.stem]
        filename = image_path.name
        if Path(source_filename).stem != xml_path.stem:
            filename_metadata_mismatches += 1
        xml_width = int(root.findtext("size/width", "0"))
        xml_height = int(root.findtext("size/height", "0"))
        if xml_width <= 0 or xml_height <= 0:
            raise ValueError(f"{xml_path}: invalid image dimensions")
        with Image.open(image_path) as image:
            decoded = ImageOps.exif_transpose(image)
            width, height = decoded.size
            decoded.load()
        if (width, height) != (xml_width, xml_height):
            dimension_metadata_mismatches += 1
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "annotation_width": xml_width,
            "annotation_height": xml_height,
            "source_domain": domain,
            "source_original_filename": source_filename,
        })
        for object_node in root.findall("object"):
            class_name = (
                object_node.findtext("class")
                or object_node.findtext("name")
                or ""
            ).strip().lower()
            if class_name == "stoma":
                class_name = "stomata"
            if class_name not in CATEGORY_IDS:
                continue
            box, clipped = parse_box(
                object_node.find("bndbox"),
                width,
                height,
            )
            clipped_boxes += int(clipped)
            if box is None:
                rejected_boxes += 1
                continue
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": CATEGORY_IDS[class_name],
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
                "source_domain": domain,
                "edge_clipped": clipped,
            })
            class_counts[class_name] += 1
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for name, category_id in CATEGORY_IDS.items()
        ],
    }
    summary = {
        "source_domain": domain,
        "images": len(images),
        "annotations": len(annotations),
        "annotations_by_class": dict(sorted(class_counts.items())),
        "clipped_boxes": clipped_boxes,
        "rejected_degenerate_boxes": rejected_boxes,
        "filename_metadata_mismatches": filename_metadata_mismatches,
        "dimension_metadata_mismatches": dimension_metadata_mismatches,
    }
    return coco, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coco, summary = convert(args.source, args.domain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, indent=2) + "\n")
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
