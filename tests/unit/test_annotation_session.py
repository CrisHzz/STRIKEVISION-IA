from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ufc_tracker.annotations.contracts import AnnotationConfig, QualityThresholds
from ufc_tracker.annotations.session import AnnotationMediaResolver, AnnotationStore
from ufc_tracker.annotations.windows import write_jsonl


def _config() -> AnnotationConfig:
    return AnnotationConfig(
        schema_version="strike_annotations_v1",
        pose_version="pose_dataset_v1",
        window_frames=4,
        stride_frames=1,
        expected_fighters=2,
        quality_thresholds=QualityThresholds(0.75, 0.75, 0.55),
        labels=("strike", "no_strike", "unknown_occluded"),
        required_keypoints=("left_wrist",),
    )


def _row() -> dict[str, object]:
    return {
        "schema_version": "strike_annotations_v1",
        "window_id": "round_1__f000000-f000003",
        "video_id": "round_1",
        "video_path": "data/round_1.mp4",
        "pose_path": "outputs/round_1/pose.jsonl",
        "pose_version": "pose_dataset_v1",
        "fps": 10.0,
        "start_frame": 0,
        "end_frame": 3,
        "start_seconds": 0.0,
        "end_seconds": 0.3,
        "frame_count": 4,
        "label": None,
        "suggested_label": None,
        "quality": {
            "status": "review",
            "two_fighter_frame_ratio": 1.0,
            "two_pose_valid_frame_ratio": 1.0,
            "required_keypoint_ratio": 1.0,
            "reasons": [],
        },
        "annotator": None,
        "annotated_at": None,
        "notes": "",
    }


def _write_video(path: Path, frame_count: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for frame_index in range(frame_count):
        writer.write(np.full((24, 32, 3), frame_index, dtype=np.uint8))
    writer.release()


def test_store_saves_a_label_and_advances_to_next_unlabeled(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.jsonl"
    first = _row()
    second = dict(_row(), window_id="round_1__f000001-f000004", start_frame=1, end_frame=4)
    second["start_seconds"] = 0.1
    second["end_seconds"] = 0.4
    write_jsonl(annotation_path, [first, second])
    store = AnnotationStore(annotation_path, _config())

    saved = store.save_label(0, "strike", annotator="Cristian", notes="jab claro")

    assert saved["label"] == "strike"
    assert saved["annotator"] == "Cristian"
    assert saved["notes"] == "jab claro"
    assert store.next_unlabeled() == 1
    assert store.summary().to_dict()["labeled"] == 1
    reloaded = AnnotationStore(annotation_path, _config())
    assert reloaded.row(0)["label"] == "strike"


def test_media_resolver_returns_full_original_and_pose_preview(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    original = project_root / "data" / "round_1.mp4"
    preview = project_root / "outputs" / "round_1" / "pose_preview.mp4"
    _write_video(original)
    _write_video(preview)
    media = AnnotationMediaResolver(project_root)
    row = _row()

    original_source = media.source_for(row, "original")
    preview_source = media.source_for(row, "pose_preview")

    assert original_source == original.resolve()
    assert preview_source == preview.resolve()
    assert not (project_root / "outputs" / "clips").exists()
