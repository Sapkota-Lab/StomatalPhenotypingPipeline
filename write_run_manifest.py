#!/usr/bin/env python3
"""Write reproducibility metadata for a calibrated stomata analysis run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    parser.add_argument("--nms-iou", type=float, required=True)
    parser.add_argument("--center-distance-nms", type=float, required=True)
    args = parser.parse_args()
    with args.ledger.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    status_counts = Counter(row["decision_status"] for row in rows)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": args.device,
        "pipeline_git_commit": git_value("rev-parse", "HEAD"),
        "pipeline_git_dirty": bool(git_value("status", "--porcelain")),
        "model": {
            "path": str(args.model),
            "sha256": sha256(args.model),
        },
        "calibration": {
            "path": str(args.calibration_report),
            "sha256": sha256(args.calibration_report),
        },
        "input_manifest": {
            "path": str(args.input_manifest),
            "sha256": sha256(args.input_manifest),
        },
        "inference": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "nms_iou": args.nms_iou,
            "center_distance_nms": args.center_distance_nms,
            "proposal_floor": 0.001,
        },
        "ledger": {
            "path": str(args.ledger),
            "sha256": sha256(args.ledger),
            "rows": len(rows),
            "images": len({row["stomata_image"] for row in rows}),
            "plants": len({row["plant_unit_id"] for row in rows}),
            "decision_status_counts": dict(sorted(status_counts.items())),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote run manifest to {args.output}")


if __name__ == "__main__":
    main()
