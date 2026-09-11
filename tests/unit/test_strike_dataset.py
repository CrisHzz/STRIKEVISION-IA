from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
import yaml

from ufc_tracker.annotations.dataset import (
    canonical_pose_records, inspect_dataset, load_pose_window, select_annotation_files, sha256,
)
from ufc_tracker.annotations.repair import atomic_jsonl, refresh_quality_file
from ufc_tracker.annotations.windows import read_jsonl


def example(fps=30.0):
    row = {
        "window_id": "example", "start_frame": 10, "end_frame": 33,
        "frame_count": 24, "fps": fps, "label": "strike",
    }
    records = [
        {
            "frame_index": frame, "fighter_id": slot, "visible": True,
            "pose_valid": True, "keypoints": {"left_wrist": [frame / fps * 100, 20, .9]},
        }
        for frame in range(10, 34) for slot in ("fighter_left", "fighter_right")
    ]
    return row, records


def test_slots_preserve_source_and_reject_collisions():
    _, records = example()
    before = deepcopy(records)
    assert canonical_pose_records(records)[0]["fighter_id"] == "fighter_0"
    assert canonical_pose_records([{**records[0], "fighter_id": 2}])[0]["fighter_id"] == "fighter_1"
    assert records == before
    with pytest.raises(ValueError, match="Duplicate"):
        canonical_pose_records([records[0], {**records[0], "fighter_id": "1"}])
    with pytest.raises(ValueError, match="Unknown"):
        canonical_pose_records([{**records[0], "fighter_id": "referee"}])


@pytest.mark.parametrize("fps", [24, 29.97002997, 30, 59.94005994, 60])
def test_real_time_velocity_and_interval_are_preserved(fps):
    row, records = example(fps)
    sample = load_pose_window(row, records, ("left_wrist",))
    assert sample["label"] == "strike"
    assert sample["duration_seconds"] == 24 / fps
    assert sample["sample_span_seconds"] == 23 / fps
    assert sample["timestamps_seconds"][0] == 10 / fps
    assert sample["timestamps_seconds"][-1] == 33 / fps
    np.testing.assert_allclose(sample["velocity_pixels_per_second"][1:, :, :, 0], 100)
    assert not sample["velocity_mask"][0].any()


def test_missing_points_not_zero_or_interpolated_across_gap():
    row, records = example()
    for record in records:
        if record["frame_index"] == 20:
            record["keypoints"]["left_wrist"] = None
    sample = load_pose_window(row, records, ("left_wrist",))
    assert np.isnan(sample["positions"][10]).all()
    assert not sample["valid_mask"][10].any()
    assert not sample["velocity_mask"][10:12].any()
    sparse = load_pose_window(row, records, ("left_wrist",), steps=2)
    assert not sparse["velocity_mask"].any()


def test_resampling_never_includes_unlabeled_context():
    row, records = example()
    records += [{**records[0], "frame_index": 9}, {**records[0], "frame_index": 34}]
    sample = load_pose_window(row, records, ("left_wrist",), steps=48)
    assert sample["frame_indices"].min() == 10
    assert sample["frame_indices"].max() == 33
    repeated = np.r_[False, np.diff(sample["frame_indices"]) == 0]
    assert not sample["velocity_mask"][repeated].any()


def test_track_reassignment_does_not_generate_velocity_spike():
    row, records = example()
    for record in records:
        record["track_id"] = 1 if record["frame_index"] < 20 else 10
    sample = load_pose_window(row, records, ("left_wrist",))
    assert not sample["velocity_mask"][10].any()
    assert sample["velocity_mask"][11].all()
    sparse = load_pose_window(row, records, ("left_wrist",), steps=2)
    assert not sparse["velocity_mask"].any()


def test_untrusted_identity_masks_pose_but_preserves_human_label():
    row, records = example()
    sample = load_pose_window({**row, "pose_usable": False}, records, ("left_wrist",))
    assert sample["label"] == "strike"
    assert sample["pose_usable"] is False
    assert not sample["valid_mask"].any()
    assert not sample["velocity_mask"].any()
    assert np.isnan(sample["positions"]).all()


def test_exclusions_are_applied_without_deleting_files(tmp_path):
    for name in ("complete.jsonl", "pilot.jsonl", "unlabeled.jsonl"):
        (tmp_path / name).touch()
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "schema_version": "strike_dataset_v1", "annotations_dir": ".",
        "retain_low_pose_quality": True,
        "excluded_files": {"pilot.jsonl": "legacy", "unlabeled.jsonl": "excluded"},
    }))
    files, _ = select_annotation_files(tmp_path, policy)
    assert [p.name for p in files] == ["complete.jsonl"]
    assert (tmp_path / "pilot.jsonl").exists()


def test_quality_refresh_preserves_all_human_fields_and_is_idempotent(tmp_path):
    path = tmp_path / "annotations.jsonl"
    rows = [{"window_id": "w", "label": "strike", "start_frame": 0,
             "notes": "human note", "annotated_at": "original", "quality": {"old": 1}}]
    atomic_jsonl(path, rows)
    updated = [{**rows[0], "quality": {"new": 2}, "suggested_label": None}]
    backup = tmp_path / "backup"
    assert refresh_quality_file(path, updated, expected_sha256=sha256(path), backup_dir=backup) == 1
    assert read_jsonl(backup / path.name) == rows
    assert read_jsonl(path) == updated
    assert refresh_quality_file(path, updated, expected_sha256=sha256(path), backup_dir=backup) == 0
    with pytest.raises(ValueError, match="human fields"):
        refresh_quality_file(path, [{**updated[0], "label": "no_strike"}],
                             expected_sha256=sha256(path), backup_dir=backup)
    with pytest.raises(RuntimeError, match="Concurrent"):
        refresh_quality_file(path, updated, expected_sha256="stale", backup_dir=backup)


def test_index_keeps_low_quality_binary_and_unknown_separate(tmp_path):
    from pathlib import Path

    from ufc_tracker.annotations.contracts import load_annotation_config
    from ufc_tracker.annotations.windows import generate_annotation_windows

    config_path = Path(__file__).resolve().parents[2] / "configs/data/strike_annotations_v1.yaml"
    (tmp_path / "config.yaml").write_bytes(config_path.read_bytes())
    config = load_annotation_config(config_path)
    pose_folder = tmp_path / "poses"
    pose_folder.mkdir()
    poses = [
        {"frame_index": frame, "fighter_id": slot, "visible": False,
         "pose_valid": False, "keypoints": {}}
        for frame in range(36) for slot in ("1", "2")
    ]
    atomic_jsonl(pose_folder / "pose.jsonl", poses)
    (pose_folder / "pose_metrics.json").write_text(json.dumps({"fps": 30, "frame_count": 36}))
    (pose_folder / "run_metadata.json").write_text("{}")
    (tmp_path / "round.mp4").touch()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    rows = generate_annotation_windows(
        poses, config=config, video_id="round", video_path="round.mp4",
        pose_path="poses/pose.jsonl", fps=30, frame_count=36,
    )
    rows[0]["label"] = "strike"
    rows[1]["label"] = "unknown_occluded"
    atomic_jsonl(annotations / "round.jsonl", rows)
    (tmp_path / "splits.csv").write_text(
        "relative_path,fight_id,category\nround.mp4,fight,aggressive_men\n"
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump({
        "schema_version": "strike_dataset_v1", "annotations_dir": "annotations",
        "splits_manifest": "splits.csv", "annotation_config": "config.yaml",
        "excluded_files": {}, "training_labels": ["strike", "no_strike"],
        "retain_low_pose_quality": True,
    }))
    manifest, refreshed, _ = inspect_dataset(tmp_path, policy)
    assert len(manifest) == 2
    assert manifest[0]["binary_eligible"] is True
    assert manifest[0]["quality"]["status"] == "auto_unknown"
    assert manifest[1]["binary_eligible"] is False
    assert refreshed[annotations / "round.jsonl"] == rows
    assert read_jsonl(annotations / "round.jsonl") == rows
