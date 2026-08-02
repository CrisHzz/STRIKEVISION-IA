
from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.estimator import (
    KEYPOINT_NAMES,
    MIN_VALID_REQUIRED_KEYPOINTS,
    REQUIRED_KEYPOINTS,
    PoseEstimator,
    empty_keypoints,
    MediaPipePoseEstimator,
)
from ufc_tracker.tracking.contracts import TrackingRecord
from ufc_tracker.tracking.export import extract_fighter_tracking
from ufc_tracker.tracking.merge import (
    MAX_MERGE_DISTANCE,
    MAX_MERGE_GAP_FRAMES,
    extract_merged_fighter_tracking,
)

SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)
FIGHTER_COLORS = ((69, 3, 252), (255, 51, 109))


@dataclass(frozen=True)
class PoseRecord:
    frame_index: int
    timestamp_seconds: float
    fighter_id: str
    track_id: int
    bbox_xyxy: tuple[float, float, float, float] | None
    visible: bool
    pose_valid: bool
    keypoints: dict[str, tuple[float, float, float] | None]
    inference_ms: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "fighter_id": self.fighter_id,
            "track_id": self.track_id,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "visible": self.visible,
            "pose_valid": self.pose_valid,
            "keypoints": {
                name: None if point is None else [round(value, 4) for value in point]
                for name, point in self.keypoints.items()
            },
            "inference_ms": round(self.inference_ms, 4) if self.inference_ms is not None else None,
        }


@dataclass(frozen=True)
class PosePipelineResult:
    output_dir: Path
    tracking_path: Path
    pose_path: Path
    preview_path: Path
    metrics_path: Path
    metadata_path: Path


EstimatorFactory = Callable[[], PoseEstimator]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _crop_from_bbox(
    frame: np.ndarray, bbox: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[int, int]] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right], (left, top)


def _pose_is_valid(keypoints: dict[str, tuple[float, float, float] | None]) -> bool:
    return (
        sum(keypoints[name] is not None for name in REQUIRED_KEYPOINTS)
        >= MIN_VALID_REQUIRED_KEYPOINTS
    )


def _records_by_frame(records: list[TrackingRecord]) -> dict[int, list[TrackingRecord]]:
    grouped: dict[int, list[TrackingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.frame_index].append(record)
    return grouped


def estimate_pose_records(
    video_path: str | Path,
    tracking_records: list[TrackingRecord],
    estimator: PoseEstimator,
    *,
    frame_count: int | None = None,
) -> list[PoseRecord]:
    """Run one pose backend over all stable fighter tracks in source-frame order."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    by_frame = _records_by_frame(tracking_records)
    expected_frames = frame_count
    if expected_frames is None:
        expected_frames = max(by_frame, default=-1) + 1
    output: list[PoseRecord] = []
    frame_index = 0
    try:
        while True:
            if frame_index >= expected_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            for tracking in by_frame.get(frame_index, []):
                keypoints = empty_keypoints()
                inference_ms: float | None = None
                if tracking.visible and tracking.bbox_xyxy is not None:
                    crop_info = _crop_from_bbox(frame, tracking.bbox_xyxy)
                    if crop_info is not None:
                        crop, offset = crop_info
                        estimate = estimator.estimate(
                            crop,
                            fighter_id=tracking.fighter_id,
                            offset_xy=offset,
                            timestamp_seconds=tracking.timestamp_seconds,
                        )
                        keypoints = estimate.keypoints
                        inference_ms = estimate.inference_ms
                    else:
                        _mark_missing(estimator, tracking.fighter_id)
                else:
                    _mark_missing(estimator, tracking.fighter_id)
                output.append(
                    PoseRecord(
                        frame_index=tracking.frame_index,
                        timestamp_seconds=tracking.timestamp_seconds,
                        fighter_id=tracking.fighter_id,
                        track_id=tracking.track_id,
                        bbox_xyxy=tracking.bbox_xyxy,
                        visible=tracking.visible,
                        pose_valid=_pose_is_valid(keypoints),
                        keypoints=keypoints,
                        inference_ms=inference_ms,
                    )
                )
            frame_index += 1
    finally:
        capture.release()
        estimator.close()

    if frame_index != expected_frames:
        raise RuntimeError(
            f"Tracking has {expected_frames} frames but video yielded {frame_index} frames."
        )
    return output


def _mark_missing(estimator: PoseEstimator, fighter_id: str) -> None:
    """Keep stateful backends from smoothing over a lost fighter track."""
    callback = getattr(estimator, "mark_missing", None)
    if callback is not None:
        callback(fighter_id)


def _fighter_metrics(records: list[PoseRecord]) -> dict[str, object]:
    visible = [record for record in records if record.visible]
    valid = [record for record in visible if record.pose_valid]
    inference = [record.inference_ms for record in visible if record.inference_ms is not None]
    keypoint_availability = {
        name: round(
            sum(record.keypoints[name] is not None for record in visible) / len(visible), 6
        )
        if visible
        else 0.0
        for name in REQUIRED_KEYPOINTS
    }
    complete = [
        record
        for record in visible
        if all(record.keypoints[name] is not None for name in REQUIRED_KEYPOINTS)
    ]

    continuity: dict[str, float] = {}
    motion: dict[str, float | None] = {}
    for name in REQUIRED_KEYPOINTS:
        pairs = 0
        both_present = 0
        distances: list[float] = []
        for previous, current in zip(records, records[1:]):
            if current.frame_index != previous.frame_index + 1:
                continue
            if not previous.visible or not current.visible:
                continue
            pairs += 1
            first, second = previous.keypoints[name], current.keypoints[name]
            if first is None or second is None or current.bbox_xyxy is None:
                continue
            both_present += 1
            x1, y1, x2, y2 = current.bbox_xyxy
            diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)
            distances.append(float(np.hypot(second[0] - first[0], second[1] - first[1])) / diagonal)
        continuity[name] = round(both_present / pairs, 6) if pairs else 0.0
        motion[name] = round(float(median(distances)), 6) if distances else None

    return {
        "tracking_visible_frames": len(visible),
        "pose_valid_frames": len(valid),
        "pose_coverage": round(len(valid) / len(visible), 6) if visible else 0.0,
        "pose_lost_frames": len(visible) - len(valid),
        "incomplete_required_keypoint_frames": len(visible) - len(complete),
        "required_keypoint_availability": keypoint_availability,
        "required_keypoint_temporal_continuity": continuity,
        "median_normalized_keypoint_motion": motion,
        "mean_inference_ms_per_visible_frame": round(float(np.mean(inference)), 4)
        if inference
        else None,
    }


def calculate_metrics(records: list[PoseRecord]) -> dict[str, object]:
    grouped: dict[str, list[PoseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.fighter_id].append(record)
    per_fighter = {fighter_id: _fighter_metrics(items) for fighter_id, items in grouped.items()}
    visible = sum(int(metrics["tracking_visible_frames"]) for metrics in per_fighter.values())
    valid = sum(int(metrics["pose_valid_frames"]) for metrics in per_fighter.values())
    inference = [
        record.inference_ms
        for record in records
        if record.inference_ms is not None and record.visible
    ]
    return {
        "fighters": per_fighter,
        "overall": {
            "tracking_visible_frames": visible,
            "pose_valid_frames": valid,
            "pose_coverage": round(valid / visible, 6) if visible else 0.0,
            "mean_inference_ms_per_visible_frame": round(float(np.mean(inference)), 4)
            if inference
            else None,
        },
    }


def _draw_pose(frame: np.ndarray, record: PoseRecord, color: tuple[int, int, int]) -> None:
    if record.bbox_xyxy is not None:
        x1, y1, x2, y2 = (int(value) for value in record.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            record.fighter_id,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    for first_name, second_name in SKELETON_EDGES:
        first, second = record.keypoints[first_name], record.keypoints[second_name]
        if first is not None and second is not None:
            cv2.line(frame, (int(first[0]), int(first[1])), (int(second[0]), int(second[1])), color, 2)
    for point in record.keypoints.values():
        if point is not None:
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)


def render_pose_preview(
    video_path: str | Path,
    records: list[PoseRecord],
    output_path: Path,
    *,
    frame_count: int | None = None,
) -> None:
    """Render stable track IDs and a skeleton, including frames with missing pose."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open preview writer: {output_path}")

    by_frame: dict[int, list[PoseRecord]] = defaultdict(list)
    fighter_order: dict[str, int] = {}
    for record in records:
        by_frame[record.frame_index].append(record)
        fighter_order.setdefault(record.fighter_id, len(fighter_order))
    frame_index = 0
    expected_frames = frame_count
    if expected_frames is None:
        expected_frames = max(by_frame, default=-1) + 1
    try:
        while True:
            if frame_index >= expected_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            for record in by_frame.get(frame_index, []):
                color = FIGHTER_COLORS[fighter_order[record.fighter_id] % len(FIGHTER_COLORS)]
                _draw_pose(frame, record, color)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def _default_factory() -> PoseEstimator:
    return MediaPipePoseEstimator()


def run_pose_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    tracking_confidence: float = 0.5,
    min_track_frames: int = 15,
    max_frames: int | None = None,
    merge_track_fragments: bool = False,
    max_merge_gap_frames: int = MAX_MERGE_GAP_FRAMES,
    max_merge_distance: float = MAX_MERGE_DISTANCE,
    estimator_factory: EstimatorFactory = _default_factory,
) -> PosePipelineResult:
    """Generate DVC-ready fighter tracks, MediaPipe poses, metrics and preview.

    With `merge_track_fragments` the ByteTrack fragments are chained into two
    stable identities instead of requiring the detector to select exactly two
    tracks, which makes `min_track_frames` irrelevant.
    """
    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Could not find input video: {source}")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if merge_track_fragments:
        merged = extract_merged_fighter_tracking(
            source,
            confidence=tracking_confidence,
            max_frames=max_frames,
            max_gap_frames=max_merge_gap_frames,
            max_distance=max_merge_distance,
        )
        tracking_records = merged.records
        fps = merged.fps
        frame_count = merged.frame_count
        tracking_metadata: dict[str, Any] = merged.metadata()
    else:
        tracking_records, fps, frame_count = extract_fighter_tracking(
            source,
            confidence=tracking_confidence,
            max_frames=max_frames,
            min_track_frames=min_track_frames,
        )
        tracking_metadata = {
            "component": "registered_person_detector",
            "model": "PersonDetector@production",
            "tracker": "bytetrack.yaml",
            "confidence": tracking_confidence,
            "configured_minimum_track_frames": min_track_frames,
            "effective_minimum_track_frames": min(
                min_track_frames,
                max(1, math.ceil(frame_count * 0.05)),
            ),
            "fighter_id_policy": "fighter_track_<bytetrack_fragment_id>",
            "track_fragment_count": len({record.track_id for record in tracking_records}),
        }
    tracking_path = destination / "tracking.jsonl"
    _write_jsonl(tracking_path, (record.to_dict() for record in tracking_records))

    estimator = estimator_factory()
    backend_name = estimator.name
    backend_metadata = {
        "name": backend_name,
        "weight_path": getattr(estimator, "weight_path", None),
        "implementation": type(estimator).__name__,
    }
    records = estimate_pose_records(
        source, tracking_records, estimator, frame_count=frame_count
    )
    pose_path = destination / "pose.jsonl"
    preview_path = destination / "pose_preview.mp4"
    _write_jsonl(pose_path, (record.to_dict() for record in records))
    render_pose_preview(source, records, preview_path, frame_count=frame_count)
    pose_metrics = calculate_metrics(records)

    root = project_root(source)
    try:
        relative_source = source.relative_to(root).as_posix()
    except ValueError:
        relative_source = str(source)
    metrics_path = destination / "pose_metrics.json"
    metrics_payload = {
        "video": relative_source,
        "fps": fps,
        "frame_count": frame_count,
        "backend": backend_name,
        "metrics": pose_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata_path = destination / "run_metadata.json"
    metadata_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": relative_source,
        "input_dvc_path": str(Path(relative_source).parent).replace("\\", "/"),
        "git_commit": _git_commit(root),
        "tracking": tracking_metadata,
        "pose_model": backend_metadata,
        "packages": {
            "ultralytics": _package_version("ultralytics"),
            "mediapipe": _package_version("mediapipe"),
            "opencv-contrib-python": _package_version("opencv-contrib-python"),
        },
        "keypoint_contract": list(KEYPOINT_NAMES),
        "required_keypoints": list(REQUIRED_KEYPOINTS),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return PosePipelineResult(
        output_dir=destination,
        tracking_path=tracking_path,
        pose_path=pose_path,
        preview_path=preview_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
    )
