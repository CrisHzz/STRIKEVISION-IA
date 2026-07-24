"""Shared local paths for pretrained model weights."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_MARKERS = ("pyproject.toml", "environment.yml")


def project_root(start: Path | None = None) -> Path:
    """Resolve the StrikeVision repo root by walking parents."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "Could not locate project root (missing pyproject.toml / environment.yml)."
    )


def weights_dir(root: Path | None = None) -> Path:
    """Directory for shared local model weights: ``models/weights``."""
    directory = (root or project_root()) / "models" / "weights"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_pretrained_weight(
    filename: str = "yolo11n.pt",
    *,
    download: bool = True,
    root: Path | None = None,
) -> Path:
    """Return path to a weight file under ``models/weights``.

    If the file is missing and ``download`` is True, Ultralytics downloads it
    once into that folder so notebooks and app code can reuse the same file.
    """
    destination = weights_dir(root)
    path = destination / filename
    if path.exists():
        return path

    if not download:
        raise FileNotFoundError(
            f"Weight file not found: {path}. "
            "Place the .pt file there or call with download=True."
        )

    # Ultralytics downloads into the current working directory by name;
    # run the download from weights_dir so the file lands in the shared folder.
    from ultralytics import YOLO

    previous_cwd = Path.cwd()
    try:
        os.chdir(destination)
        YOLO(filename)
    finally:
        os.chdir(previous_cwd)

    if not path.exists():
        raise FileNotFoundError(f"Download finished but weight is still missing: {path}")
    return path
