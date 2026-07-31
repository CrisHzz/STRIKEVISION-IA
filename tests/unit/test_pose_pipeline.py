from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ufc_tracker.pose.estimator import KEYPOINT_NAMES, PoseEstimate
from ufc_tracker.pose.pipeline import calculate_metrics, estimate_pose_records, render_pose_preview
from ufc_tracker.tracking.contracts import TrackingRecord


class FakePoseEstimator:
    name = "fake_pose"

    def __init__(self) -> None:
        self.closed = False

    def estimate(
        self,
        crop_bgr,
        *,
        fighter_id: str,
        offset_xy: tuple[int, int],
        timestamp_seconds: float,
    ) -> PoseEstimate:
        del crop_bgr, fighter_id, timestamp_seconds
        keypoints = {name: None for name in KEYPOINT_NAMES}
        keypoints["left_elbow"] = (1.0 + offset_xy[0], 2.0 + offset_xy[1], 0.9)
        keypoints["right_elbow"] = (3.0 + offset_xy[0], 4.0 + offset_xy[1], 0.8)
        keypoints["left_wrist"] = (2.0 + offset_xy[0], 3.0 + offset_xy[1], 0.9)
        keypoints["right_wrist"] = (4.0 + offset_xy[0], 5.0 + offset_xy[1], 0.8)
        return PoseEstimate(keypoints=keypoints, inference_ms=1.25)

    def close(self) -> None:
        self.closed = True


def _make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()


def test_pose_records_keep_tracking_identity_and_full_frame_coordinates(tmp_path: Path) -> None:
    video_path = tmp_path / "round.mp4"
    _make_video(video_path)
    records = [
        TrackingRecord(0, 0.0, "fighter_track_7", 7, (10.0, 5.0, 20.0, 20.0), 0.9, True),
        TrackingRecord(1, 0.1, "fighter_track_7", 7, None, None, False),
    ]
    estimator = FakePoseEstimator()

    output = estimate_pose_records(video_path, records, estimator, frame_count=2)

    assert estimator.closed is True
    assert output[0].fighter_id == "fighter_track_7"
    assert output[0].track_id == 7
    assert output[0].keypoints["left_wrist"] == (12.0, 8.0, 0.9)
    assert output[1].keypoints == {name: None for name in KEYPOINT_NAMES}
    assert output[1].inference_ms is None


def test_metrics_and_preview_tolerate_missing_pose(tmp_path: Path) -> None:
    video_path = tmp_path / "round.mp4"
    preview_path = tmp_path / "preview.mp4"
    _make_video(video_path)
    records = [
        TrackingRecord(0, 0.0, "fighter_track_7", 7, (10.0, 5.0, 20.0, 20.0), 0.9, True),
        TrackingRecord(1, 0.1, "fighter_track_7", 7, None, None, False),
    ]
    output = estimate_pose_records(
        video_path, records, FakePoseEstimator(), frame_count=2
    )

    metrics = calculate_metrics(output)
    render_pose_preview(video_path, output, preview_path, frame_count=2)

    assert metrics["fighters"]["fighter_track_7"]["tracking_visible_frames"] == 1
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0
