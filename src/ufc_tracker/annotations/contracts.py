"""Schema configuration and validation for ``strike_annotations_v1``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "strike_annotations_v1"
ALLOWED_LABELS = frozenset({"strike", "no_strike", "unknown_occluded"})
QUALITY_STATUSES = frozenset({"review", "auto_unknown"})


@dataclass(frozen=True)
class QualityThresholds:
    min_two_fighter_frame_ratio: float
    min_two_pose_valid_frame_ratio: float
    min_required_keypoint_ratio: float


@dataclass(frozen=True)
class AnnotationConfig:
    schema_version: str
    pose_version: str
    window_frames: int
    stride_frames: int
    expected_fighters: int
    quality_thresholds: QualityThresholds
    labels: tuple[str, ...]
    required_keypoints: tuple[str, ...]


def _ratio(value: Any, field: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1, received: {result}")
    return result


def load_annotation_config(path: str | Path) -> AnnotationConfig:
    """Read and validate the YAML configuration for annotation windows."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Annotation config does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Annotation config must be a mapping: {config_path}")

    schema_version = str(data.get("schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported annotation schema: {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    window_frames = int(data.get("window_frames", 0))
    stride_frames = int(data.get("stride_frames", 0))
    expected_fighters = int(data.get("expected_fighters", 0))
    if window_frames <= 0 or stride_frames <= 0:
        raise ValueError("window_frames and stride_frames must be greater than zero")
    if stride_frames > window_frames:
        raise ValueError("stride_frames cannot exceed window_frames")
    if expected_fighters <= 0:
        raise ValueError("expected_fighters must be greater than zero")

    labels = tuple(str(label) for label in data.get("labels", ()))
    if set(labels) != ALLOWED_LABELS:
        raise ValueError(
            f"labels must contain exactly {sorted(ALLOWED_LABELS)}, received: {labels}"
        )
    required_keypoints = tuple(str(name) for name in data.get("required_keypoints", ()))
    if not required_keypoints:
        raise ValueError("required_keypoints cannot be empty")

    raw_thresholds = data.get("quality_thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ValueError("quality_thresholds must be a mapping")
    thresholds = QualityThresholds(
        min_two_fighter_frame_ratio=_ratio(
            raw_thresholds.get("min_two_fighter_frame_ratio"),
            "min_two_fighter_frame_ratio",
        ),
        min_two_pose_valid_frame_ratio=_ratio(
            raw_thresholds.get("min_two_pose_valid_frame_ratio"),
            "min_two_pose_valid_frame_ratio",
        ),
        min_required_keypoint_ratio=_ratio(
            raw_thresholds.get("min_required_keypoint_ratio"),
            "min_required_keypoint_ratio",
        ),
    )
    return AnnotationConfig(
        schema_version=schema_version,
        pose_version=str(data.get("pose_version", "")),
        window_frames=window_frames,
        stride_frames=stride_frames,
        expected_fighters=expected_fighters,
        quality_thresholds=thresholds,
        labels=labels,
        required_keypoints=required_keypoints,
    )


def validate_annotation_rows(
    rows: list[dict[str, Any]],
    config: AnnotationConfig,
    *,
    require_labeled: bool = False,
) -> list[str]:
    """Return all contract violations without stopping after the first row."""
    errors: list[str] = []
    seen_window_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        prefix = f"line {line_number}"
        window_id = str(row.get("window_id", ""))
        if not window_id:
            errors.append(f"{prefix}: window_id is required")
        elif window_id in seen_window_ids:
            errors.append(f"{prefix}: duplicate window_id {window_id!r}")
        seen_window_ids.add(window_id)

        if row.get("schema_version") != config.schema_version:
            errors.append(f"{prefix}: invalid schema_version")
        if not row.get("video_id"):
            errors.append(f"{prefix}: video_id is required")
        if not row.get("video_path"):
            errors.append(f"{prefix}: video_path is required")
        if not row.get("pose_path"):
            errors.append(f"{prefix}: pose_path is required")
        if row.get("pose_version") != config.pose_version:
            errors.append(f"{prefix}: pose_version does not match config")

        try:
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            frame_count = int(row["frame_count"])
            fps = float(row["fps"])
            start_seconds = float(row["start_seconds"])
            end_seconds = float(row["end_seconds"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: invalid frame/time fields")
        else:
            if start_frame < 0 or end_frame < start_frame:
                errors.append(f"{prefix}: invalid frame interval")
            if frame_count != end_frame - start_frame + 1:
                errors.append(f"{prefix}: frame_count does not match interval")
            if frame_count != config.window_frames:
                errors.append(f"{prefix}: window does not contain configured frame count")
            if fps <= 0:
                errors.append(f"{prefix}: fps must be greater than zero")
            elif not math.isclose(start_seconds, start_frame / fps, abs_tol=1e-5):
                errors.append(f"{prefix}: start_seconds does not match start_frame/fps")
            elif not math.isclose(end_seconds, end_frame / fps, abs_tol=1e-5):
                errors.append(f"{prefix}: end_seconds does not match end_frame/fps")

        label = row.get("label")
        if label is not None and label not in ALLOWED_LABELS:
            errors.append(f"{prefix}: invalid label {label!r}")
        if require_labeled and label is None:
            errors.append(f"{prefix}: label is required for a training-ready dataset")
        suggested = row.get("suggested_label")
        if suggested not in (None, "unknown_occluded"):
            errors.append(f"{prefix}: suggested_label may only be unknown_occluded or null")

        quality = row.get("quality")
        if not isinstance(quality, dict):
            errors.append(f"{prefix}: quality must be a mapping")
            continue
        if quality.get("status") not in QUALITY_STATUSES:
            errors.append(f"{prefix}: invalid quality status")
        for field in (
            "two_fighter_frame_ratio",
            "two_pose_valid_frame_ratio",
            "required_keypoint_ratio",
        ):
            try:
                _ratio(quality[field], field)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{prefix}: {error}")
        reasons = quality.get("reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) for reason in reasons
        ):
            errors.append(f"{prefix}: quality.reasons must be a list of strings")
    return errors

