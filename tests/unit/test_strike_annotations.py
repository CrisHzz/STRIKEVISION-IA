from __future__ import annotations

from pathlib import Path

from ufc_tracker.annotations.contracts import (
    AnnotationConfig,
    QualityThresholds,
    validate_annotation_rows,
)
from ufc_tracker.annotations.windows import generate_annotation_windows

REQUIRED_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _config() -> AnnotationConfig:
    return AnnotationConfig(
        schema_version="strike_annotations_v1",
        pose_version="pose_dataset_v1",
        window_frames=24,
        stride_frames=6,
        expected_fighters=2,
        quality_thresholds=QualityThresholds(0.75, 0.75, 0.55),
        labels=("strike", "no_strike", "unknown_occluded"),
        required_keypoints=REQUIRED_KEYPOINTS,
    )


def _pose_record(frame: int, fighter: str, *, visible: bool = True) -> dict[str, object]:
    point = [1.0, 2.0, 0.9]
    return {
        "frame_index": frame,
        "fighter_id": fighter,
        "visible": visible,
        "pose_valid": visible,
        "keypoints": {
            name: point if visible else None for name in REQUIRED_KEYPOINTS
        },
    }


def test_generate_windows_marks_low_tracking_coverage_for_review() -> None:
    records = []
    for frame in range(30):
        records.append(_pose_record(frame, "fighter_left"))
        records.append(_pose_record(frame, "fighter_right", visible=frame >= 12))

    windows = generate_annotation_windows(
        records,
        config=_config(),
        video_id="round_1",
        video_path="data/round_1.mp4",
        pose_path="outputs/round_1/pose.jsonl",
        fps=24.0,
        frame_count=30,
    )

    assert len(windows) == 2
    assert windows[0]["window_id"] == "round_1__f000000-f000023"
    assert windows[0]["quality"]["two_fighter_frame_ratio"] == 0.5
    assert windows[0]["quality"]["status"] == "auto_unknown"
    assert windows[0]["suggested_label"] == "unknown_occluded"
    assert windows[1]["quality"]["two_fighter_frame_ratio"] == 0.75
    assert windows[1]["quality"]["status"] == "review"
    assert windows[1]["suggested_label"] is None


def test_validator_accepts_generated_unlabeled_windows() -> None:
    records = [
        _pose_record(frame, fighter)
        for frame in range(24)
        for fighter in ("fighter_left", "fighter_right")
    ]
    windows = generate_annotation_windows(
        records,
        config=_config(),
        video_id="round_1",
        video_path="data/round_1.mp4",
        pose_path="outputs/round_1/pose.jsonl",
        fps=24.0,
        frame_count=24,
    )

    assert validate_annotation_rows(windows, _config()) == []
    errors = validate_annotation_rows(windows, _config(), require_labeled=True)
    assert errors == ["line 1: label is required for a training-ready dataset"]


def test_validator_rejects_duplicates_and_invalid_labels() -> None:
    records = [
        _pose_record(frame, fighter)
        for frame in range(24)
        for fighter in ("fighter_left", "fighter_right")
    ]
    window = generate_annotation_windows(
        records,
        config=_config(),
        video_id="round_1",
        video_path="data/round_1.mp4",
        pose_path="outputs/round_1/pose.jsonl",
        fps=24.0,
        frame_count=24,
    )[0]
    duplicate = dict(window)
    duplicate["label"] = "kick"

    errors = validate_annotation_rows([window, duplicate], _config())

    assert "line 2: duplicate window_id 'round_1__f000000-f000023'" in errors
    assert "line 2: invalid label 'kick'" in errors


def test_repository_config_is_valid() -> None:
    from ufc_tracker.annotations.contracts import load_annotation_config

    root = Path(__file__).resolve().parents[2]
    config = load_annotation_config(root / "configs/data/strike_annotations_v1.yaml")

    assert config.window_frames == 24
    assert config.stride_frames == 6
