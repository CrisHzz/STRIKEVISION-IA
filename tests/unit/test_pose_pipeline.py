from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ufc_tracker.pose.estimator import KEYPOINT_NAMES, PoseEstimate
from ufc_tracker.pose.pipeline import (
    calculate_metrics,
    display_fighter_id,
    estimate_pose_records,
    load_pose_records,
    preview_uses_legacy_fighter_labels,
    refresh_pose_preview_if_legacy_labels,
    render_pose_preview,
)
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


def test_display_fighter_id_maps_legacy_side_labels() -> None:
    assert display_fighter_id("fighter_left") == "1"
    assert display_fighter_id("fighter_right") == "2"
    assert display_fighter_id("1") == "1"
    assert display_fighter_id("fighter_track_7") == "fighter_track_7"


def test_load_pose_records_and_legacy_preview_refresh(tmp_path: Path) -> None:
    video_path = tmp_path / "round.mp4"
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    _make_video(video_path)
    pose_path = pose_dir / "pose.jsonl"
    pose_path.write_text(
        '{"frame_index": 0, "timestamp_seconds": 0.0, "fighter_id": "fighter_left",'
        ' "track_id": 1, "bbox_xyxy": [2, 2, 20, 20], "visible": true,'
        ' "pose_valid": false, "keypoints": {}, "inference_ms": null}\n'
        '{"frame_index": 0, "timestamp_seconds": 0.0, "fighter_id": "fighter_right",'
        ' "track_id": 2, "bbox_xyxy": [12, 2, 30, 20], "visible": true,'
        ' "pose_valid": false, "keypoints": {}, "inference_ms": null}\n',
        encoding="utf-8",
    )
    (pose_dir / "pose_metrics.json").write_text(
        '{"frame_count": 2}',
        encoding="utf-8",
    )
    (pose_dir / "pose_preview.mp4").write_bytes(b"old")

    records = load_pose_records(pose_path)
    assert [record.fighter_id for record in records] == ["fighter_left", "fighter_right"]
    assert preview_uses_legacy_fighter_labels(records) is True
    assert refresh_pose_preview_if_legacy_labels(video_path, pose_dir) is True
    assert (pose_dir / "pose_preview.mp4").stat().st_size > 3
    assert load_pose_records(pose_path)[0].fighter_id == "fighter_left"


def test_refresh_skips_preview_when_ids_are_already_numeric(tmp_path: Path) -> None:
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    pose_path = pose_dir / "pose.jsonl"
    pose_path.write_text(
        '{"frame_index": 0, "fighter_id": "1", "track_id": 1, "bbox_xyxy": null,'
        ' "visible": false, "pose_valid": false, "keypoints": {}, "inference_ms": null}\n',
        encoding="utf-8",
    )
    preview = pose_dir / "pose_preview.mp4"
    preview.write_bytes(b"keep")

    assert refresh_pose_preview_if_legacy_labels(tmp_path / "missing.mp4", pose_dir) is False
    assert preview.read_bytes() == b"keep"
