#!/usr/bin/env python3
"""Run the gated post-mask workflow and emit final combined study products."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(*arguments: object) -> None:
    command = [sys.executable, *(str(value) for value in arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual-masks",
        type=Path,
        default=Path("complex_mask_annotations"),
    )
    parser.add_argument(
        "--validation-masks",
        type=Path,
        default=Path(
            "validation_artifacts/complex_mask_proposals_v1/json"
        ),
    )
    parser.add_argument(
        "--all-masks",
        type=Path,
        default=Path("validation_artifacts/complex_masks_all_v1/json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "validation_artifacts/tomato_production_v1/stomata.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/combined_final"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    validation = args.output / "complex_mask_validation"
    box_ledger = args.output / "stomata_boxes.csv"
    stomata = args.output / "stomata.csv"
    area = args.output / "stomatal_area"

    run(
        "validate_complex_masks.py",
        "--manual", args.manual_masks,
        "--proposals", args.validation_masks,
        "--output", validation,
        "--require-gate",
    )
    run(
        "build_stomata_ledger.py",
        "--output", box_ledger,
        "--run-id", "expert_boxes_tomato_v1",
    )
    run(
        "apply_validated_complex_masks.py",
        "--ledger", box_ledger,
        "--masks", args.all_masks,
        "--validation-report", validation / "report.json",
        "--output", stomata,
    )
    run(
        "analyze_stomatal_area.py",
        "--stomata", stomata,
        "--output", area,
    )
    run(
        "build_combined_phenotypes.py",
        "--area-results", area,
        "--stomata", stomata,
        "--predictions", args.predictions,
        "--output", args.output,
    )
    print(f"Final combined analysis is available under {args.output}")


if __name__ == "__main__":
    main()
