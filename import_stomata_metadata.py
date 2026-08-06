#!/usr/bin/env python3
"""Import plant/photo metadata from the supplied XLSX without modifying it."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"x": MAIN_NS}
SERIAL_PATTERN = re.compile(r"IMG_(\d+)\.(?:JPG|JPEG|PNG)$", re.IGNORECASE)


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.findall(".//x:t", NS))
        for item in root.findall("x:si", NS)
    ]


def cell_value(cell: ET.Element, strings: list[str]) -> str:
    value = cell.findtext("x:v", default="", namespaces=NS)
    if cell.get("t") == "s" and value:
        return strings[int(value)]
    return value


def read_workbook_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        strings = shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    for row in sheet.findall(".//x:sheetData/x:row", NS):
        values = {
            cell.get("r", "")[0]: cell_value(cell, strings).strip()
            for cell in row.findall("x:c", NS)
        }
        if values.get("D") == "PN":
            continue
        if not values.get("D"):
            continue
        rows.append(
            {
                "plant_name": values["D"],
                "regular_serial": values.get("E", ""),
                "control_serial": values.get("G", ""),
                "source_row": row.get("r", ""),
            }
        )
    return rows


def build_serial_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    serial_map: dict[str, dict[str, str]] = {}
    for row in rows:
        for treatment, key in (
            ("regular", "regular_serial"),
            ("control", "control_serial"),
        ):
            serial = row[key]
            if not serial:
                continue
            if serial in serial_map:
                raise ValueError(f"Duplicate photo serial in workbook: {serial}")
            serial_map[serial] = {
                "specimen_id": row["plant_name"],
                "plant_name": row["plant_name"],
                "treatment": treatment,
                "water_status": (
                    "watered" if treatment == "regular"
                    else "drought_stressed"
                ),
                "water_ml": "100" if treatment == "regular" else "0",
                "source_sheet": "Sheet1",
                "source_row": row["source_row"],
            }
    return serial_map


def image_rows(image_dir: Path, serial_map: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen_serials: set[str] = set()
    for image_path in sorted(
        path for path in image_dir.rglob("*") if path.is_file()
    ):
        match = SERIAL_PATTERN.fullmatch(image_path.name)
        if not match:
            continue
        serial = match.group(1)
        metadata = serial_map.get(serial)
        if metadata:
            seen_serials.add(serial)
        output.append(
            {
                "filename": image_path.name,
                "photo_serial": serial,
                "specimen_id": metadata["specimen_id"] if metadata else "",
                "plant_name": metadata["plant_name"] if metadata else "",
                "treatment": metadata["treatment"] if metadata else "",
                "water_status": metadata["water_status"] if metadata else "",
                "water_ml": metadata["water_ml"] if metadata else "",
                "mapping_status": "mapped" if metadata else "not_in_workbook",
                "source_sheet": metadata["source_sheet"] if metadata else "",
                "source_row": metadata["source_row"] if metadata else "",
            }
        )

    missing_images = sorted(set(serial_map) - seen_serials)
    if missing_images:
        raise ValueError(
            "Workbook serials missing from the image folder: " + ", ".join(missing_images)
        )
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "filename",
        "photo_serial",
        "specimen_id",
        "plant_name",
        "treatment",
        "water_status",
        "water_ml",
        "mapping_status",
        "source_sheet",
        "source_row",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("/Users/lenquanhone/Downloads/Stomata.xlsx"),
    )
    parser.add_argument("--images", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--output", type=Path, default=Path("image_metadata.csv"))
    args = parser.parse_args()

    workbook_rows = read_workbook_rows(args.workbook)
    serial_map = build_serial_map(workbook_rows)
    rows = image_rows(args.images, serial_map)
    write_csv(args.output, rows)

    mapped = sum(row["mapping_status"] == "mapped" for row in rows)
    regular = sum(row["treatment"] == "regular" for row in rows)
    control = sum(row["treatment"] == "control" for row in rows)
    print(
        f"Wrote {len(rows)} image records to {args.output}: "
        f"{mapped} mapped ({regular} regular, {control} control), "
        f"{len(rows) - mapped} not listed in the workbook."
    )


if __name__ == "__main__":
    main()
