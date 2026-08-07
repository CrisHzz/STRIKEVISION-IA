"""Generate fixed temporal annotation windows from pose JSONL artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ufc_tracker.annotations.contracts import AnnotationConfig


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file and report malformed lines with context."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {source}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL output."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _quality_for_window(
    records_by_frame: dict[int, list[dict[str, Any]]],
    start_frame: int,
    end_frame: int,
    config: AnnotationConfig,
) -> dict[str, Any]:
    frame_count = end_frame - start_frame + 1
    two_fighter_frames = 0
    two_pose_valid_frames = 0
    available_keypoints = 0
    expected_keypoints = (
        frame_count * config.expected_fighters * len(config.required_keypoints)
    )

    for frame_index in range(start_frame, end_frame + 1):
        records = records_by_frame.get(frame_index, [])
        visible = [record for record in records if bool(record.get("visible"))]
        pose_valid = [record for record in visible if bool(record.get("pose_valid"))]
        if len(visible) >= config.expected_fighters:
            two_fighter_frames += 1
        if len(pose_valid) >= config.expected_fighters:
            two_pose_valid_frames += 1
        for record in visible[: config.expected_fighters]:
            keypoints = record.get("keypoints")
            if not isinstance(keypoints, dict):
                continue
            available_keypoints += sum(
                keypoints.get(name) is not None for name in config.required_keypoints
            )

    two_fighter_ratio = two_fighter_frames / frame_count
    two_pose_valid_ratio = two_pose_valid_frames / frame_count
    required_keypoint_ratio = (
        available_keypoints / expected_keypoints if expected_keypoints else 0.0
    )
    thresholds = config.quality_thresholds
    reasons: list[str] = []
    if two_fighter_ratio < thresholds.min_two_fighter_frame_ratio:
        reasons.append("insufficient_two_fighter_coverage")
    if two_pose_valid_ratio < thresholds.min_two_pose_valid_frame_ratio:
        reasons.append("insufficient_two_pose_valid_coverage")
    if required_keypoint_ratio < thresholds.min_required_keypoint_ratio:
        reasons.append("insufficient_required_keypoints")

    return {
        "status": "auto_unknown" if reasons else "review",
        "two_fighter_frame_ratio": round(two_fighter_ratio, 6),
        "two_pose_valid_frame_ratio": round(two_pose_valid_ratio, 6),
        "required_keypoint_ratio": round(required_keypoint_ratio, 6),
        "reasons": reasons,
    }


def generate_annotation_windows(
    pose_records: list[dict[str, Any]],
    *,
    config: AnnotationConfig,
    video_id: str,
    video_path: str,
    pose_path: str,
    fps: float,
    frame_count: int,
) -> list[dict[str, Any]]:
    """Create unlabeled fixed windows with automatic pose-quality suggestions."""
    if fps <= 0:
        raise ValueError(f"fps must be greater than zero, received: {fps}")
    if frame_count < config.window_frames:
        raise ValueError(
            f"frame_count {frame_count} is shorter than one window "
            f"({config.window_frames} frames)"
        )

    records_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in pose_records:
        try:
            frame_index = int(record["frame_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Every pose record must contain an integer frame_index") from error
        if frame_index < 0 or frame_index >= frame_count:
            raise ValueError(
                f"Pose frame_index {frame_index} is outside frame_count {frame_count}"
            )
        records_by_frame[frame_index].append(record)

    windows: list[dict[str, Any]] = []
    final_start = frame_count - config.window_frames
    for start_frame in range(0, final_start + 1, config.stride_frames):
        end_frame = start_frame + config.window_frames - 1
        quality = _quality_for_window(records_by_frame, start_frame, end_frame, config)
        window_id = f"{video_id}__f{start_frame:06d}-f{end_frame:06d}"
        windows.append(
            {
                "schema_version": config.schema_version,
                "window_id": window_id,
                "video_id": video_id,
                "video_path": video_path,
                "pose_path": pose_path,
                "pose_version": config.pose_version,
                "fps": float(fps),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": round(start_frame / fps, 6),
                "end_seconds": round(end_frame / fps, 6),
                "frame_count": config.window_frames,
                "label": None,
                "suggested_label": (
                    "unknown_occluded" if quality["status"] == "auto_unknown" else None
                ),
                "quality": quality,
                "annotator": None,
                "annotated_at": None,
                "notes": "",
            }
        )
    return windows

