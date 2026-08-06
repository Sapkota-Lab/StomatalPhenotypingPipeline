#!/usr/bin/env python3
"""Populate precise complex traits only from a mask method that passed QC."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from validate_complex_masks import shape_metrics


def object_id(row: dict[str, str]) -> str:
    return (
        f"{Path(row['stomata_image']).stem}:"
        f"{int(row['detection_id']):04d}"
    )


def polygon_traits(polygon: list[list[float]]) -> dict[str, float]:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    base = shape_metrics(polygon)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = base["area_px2"] / hull_area if hull_area > 0 else 0.0
    eccentricity = (
        math.sqrt(
            max(
                0.0,
                1.0 - (base["width_px"] / base["length_px"]) ** 2,
            )
        )
        if base["length_px"] > 0 else 0.0
    )
    return {
        "complex_length_px": base["length_px"],
        "complex_width_px": base["width_px"],
        "complex_area_px2": base["area_px2"],
        "complex_perimeter_px": base["perimeter_px"],
        "complex_aspect_ratio": (
            base["length_px"] / base["width_px"]
            if base["width_px"] > 0 else ""
        ),
        "complex_eccentricity": eccentricity,
        "complex_solidity": solidity,
        "orientation_deg": base["orientation_deg"],
    }


def apply_masks(
    ledger_rows: list[dict[str, str]],
    proposals: dict[str, tuple[dict, Path]],
    validation_report: dict,
) -> list[dict[str, object]]:
    measurement_gate = validation_report.get("measurement_gate", {})
    if not measurement_gate.get("passed"):
        raise ValueError("Complex-mask measurement gate has not passed")
    hashes = validation_report.get("proposal_model_hashes", [])
    if len(hashes) != 1:
        raise ValueError("Validation report must identify exactly one model hash")
    validated_hash = hashes[0]
    geometry_methods = validation_report.get(
        "proposal_geometry_methods",
        [],
    )
    if len(geometry_methods) != 1:
        raise ValueError(
            "Validation report must identify exactly one geometry method"
        )
    validated_geometry = geometry_methods[0]
    output = []
    for row in ledger_rows:
        current = dict(row)
        key = object_id(row)
        pair = proposals.get(key)
        exclusion = ""
        if pair is None:
            exclusion = "validated_complex_mask_unavailable"
        else:
            proposal, path = pair
            if proposal.get("model_sha256") != validated_hash:
                raise ValueError(f"Mask model hash mismatch for {key}")
            if proposal.get("geometry_method") != validated_geometry:
                raise ValueError(f"Mask geometry method mismatch for {key}")
            if proposal.get("status") != "proposed" or not proposal.get("polygon"):
                exclusion = "complex_mask_proposal_failed"
            elif str(row.get("edge_censored", "")).lower() in {"true", "1"}:
                exclusion = "complex_mask_edge_censored"
            else:
                current.update(polygon_traits(proposal["polygon"]))
                current.update({
                    "complex_segmentation_quality": (
                        "passed_global_manual_validation_gate"
                    ),
                    "complex_mask_path": str(path),
                    "complex_mask_model_sha256": validated_hash,
                    "complex_mask_geometry_method": validated_geometry,
                    "complex_mask_model_confidence": proposal.get(
                        "model_confidence",
                        "",
                    ),
                    "measurement_quality_passed": True,
                    "measurement_exclusion_reason": "",
                })
        if exclusion:
            current.update({
                "measurement_quality_passed": False,
                "measurement_exclusion_reason": exclusion,
            })
        output.append(current)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument(
        "--validation-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.ledger.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    proposals = {}
    for path in sorted(args.masks.glob("*.json")):
        proposal = json.loads(path.read_text())
        key = proposal["object_id"]
        if key in proposals:
            raise ValueError(f"Duplicate complex mask {key}")
        proposals[key] = (proposal, path)
    report = json.loads(args.validation_report.read_text())
    output = apply_masks(rows, proposals, report)
    fieldnames = list(rows[0])
    for row in output:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output)
    passed = sum(
        row["measurement_quality_passed"] is True for row in output
    )
    print(f"Wrote {len(output)} objects with {passed} validated measurements")


if __name__ == "__main__":
    main()
