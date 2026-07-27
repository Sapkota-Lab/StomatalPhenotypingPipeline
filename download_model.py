#!/usr/bin/env python3
"""Download and verify the published stomata detector checkpoints."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


STOMAQUANT_URL = "https://huggingface.co/spaces/TemasekLifesciLab/StomaQuant/resolve/main/yolov12_300epochs.pt?download=true"
STOMAQUANT_SHA256 = "562e4851c32813f204e97c92ef6c741f1d0c5773954db98e24937e1e81919500"
STOMAQUANT_DESTINATION = Path("models/stomaquant_yolov12.pt")
STOMADET_URL = "https://data.mendeley.com/public-files/datasets/2bfcpc6bpj/files/8de4cbce-1627-4829-bf1f-35ffe953e385/file_downloaded"
STOMADET_ARCHIVE_SHA256 = (
    "88ef26439ac190f30679883863430da232596e4f49242a8a789e49416c6c4a69"
)
STOMADET_ROOT = Path("models/external/stomadet")
STOMADET_MODEL = STOMADET_ROOT / "models/stoma_detector.pth"
STOMADET_MODEL_SHA256 = (
    "ace02e6e93d386a640f7c09f8d0500836618f475c412aea8904b5fcf5f44290e"
)
STOMADET_CONFIG = STOMADET_ROOT / "models/stoma_detector.yaml"
STOMADET_CONFIG_SHA256 = (
    "6c66942dbad6dea36bdbc065cd2638782f4e8d20e92d66c0b84d4fe27bccacf0"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.exists() and digest(destination) == expected_sha256:
        print(f"Already verified: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = digest(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(destination)
    print(f"Downloaded and verified: {destination}")


def main() -> None:
    download(
        STOMAQUANT_URL,
        STOMAQUANT_DESTINATION,
        STOMAQUANT_SHA256,
    )
    if (
        STOMADET_MODEL.exists()
        and digest(STOMADET_MODEL) == STOMADET_MODEL_SHA256
        and STOMADET_CONFIG.exists()
        and digest(STOMADET_CONFIG) == STOMADET_CONFIG_SHA256
    ):
        print(f"Already verified: {STOMADET_MODEL}")
        return
    archive = STOMADET_ROOT / "models.zip"
    download(STOMADET_URL, archive, STOMADET_ARCHIVE_SHA256)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(STOMADET_ROOT)
    archive.unlink()
    if (
        digest(STOMADET_MODEL) != STOMADET_MODEL_SHA256
        or digest(STOMADET_CONFIG) != STOMADET_CONFIG_SHA256
    ):
        raise RuntimeError("Extracted StomaDet files failed checksum validation")
    print(f"Extracted and verified: {STOMADET_MODEL}")


if __name__ == "__main__":
    main()
