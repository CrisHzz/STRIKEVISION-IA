
from __future__ import annotations

import math
from pathlib import Path

import cv2

from ufc_tracker.tracking.contracts import TrackingRecord


def _resolve_video_path(video_path: str | Path) -> Path:
    from ufc_tracker.detection import project_root

    path = Path(video_path)
    if not path.is_absolute():
        path = project_root() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Could not find video: {path}")
    return path


def extract_fighter_tracking(
    video_path: str | Path,
    *,
    confidence: float = 0.5,
    max_frames: int | None = None,
    min_track_frames: int = 15,
) -> tuple[list[TrackingRecord], float, int]:
    """Run the registered UFC detector/ByteTrack pipeline and export both fighters.

    Fighter identities intentionally retain their ByteTrack IDs. This prevents the
    left/right visual ordering used by the preview renderer from becoming ML labels.
    """
    from ufc_tracker.detection.personDetection import select_fighter_tracks, track_video

    path = _resolve_video_path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"Video has invalid FPS: {path}")

    per_frame, stats, frame_count = track_video(
        path, max_frames=max_frames, conf=confidence
    )
    effective_min_track_frames = min(
        min_track_frames,
        max(1, math.ceil(frame_count * 0.05)),
    )
    fighter_track_ids = sorted(
        select_fighter_tracks(
            stats,
            frame_count,
            min_track_frames=effective_min_track_frames,
        )
    )
    if len(fighter_track_ids) < 2:
        raise RuntimeError(
            "The UFC-specific detector must select at least two fighter track fragments; "
            f"selected {len(fighter_track_ids)}: {fighter_track_ids}."
        )

    records: list[TrackingRecord] = []
    for frame_index, frame_map in enumerate(per_frame):
        visible_fighter_ids = [
            track_id for track_id in fighter_track_ids if track_id in frame_map
        ]
        if len(visible_fighter_ids) > 2:
            raise RuntimeError(
                "The UFC-specific detector selected more than two fighter tracks in the same "
                f"frame ({frame_index}): {visible_fighter_ids}. Review the referee filter."
            )
        for track_id in visible_fighter_ids:
            observation = frame_map[track_id]
            bbox = tuple(float(value) for value in observation.bbox_xyxy.tolist())
            records.append(
                TrackingRecord(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    fighter_id=f"fighter_track_{track_id}",
                    track_id=track_id,
                    bbox_xyxy=bbox,
                    confidence=float(observation.confidence),
                    visible=True,
                )
            )
    return records, fps, frame_count
