from __future__ import annotations

import json

import numpy as np

from ufc_tracker.action_recognition.features import (
    PoseFeatureSpec,
    build_pose_feature_cache,
    extract_pose_sequence,
    pooled_pose_features,
)


def _record(frame, fighter, track=1, valid=True):
    return {
        "frame_index": frame, "fighter_id": fighter, "track_id": track, "visible": valid,
        "pose_valid": valid,
        "keypoints": {
            "left_hip": [10 + frame, 20, 1], "right_hip": [20 + frame, 20, 1],
            "left_shoulder": [10 + frame, 10, 1], "right_shoulder": [20 + frame, 10, 1],
            "left_wrist": [12 + frame, 7, 0.8], "right_wrist": [24 + frame, 7, 0.8],
        },
    }


def test_pose_sequences_are_attacker_relative_and_keep_missing_observations_masked():
    sample = {"attacker": "1", "fps": 30, "start_frame": 2, "end_frame": 5, "pose_usable": True}
    records = [_record(frame, fighter) for frame in range(8) for fighter in ("fighter_left", "fighter_right")]
    records[7]["keypoints"]["left_wrist"] = None
    sequence = extract_pose_sequence(sample, records, PoseFeatureSpec(steps=4, context_seconds=0))
    assert sequence["frame_indices"].tolist() == [2, 3, 4, 5]
    assert sequence["frame_mask"].all()
    assert sequence["point_mask"][1, 1].sum() < sequence["point_mask"][1, 0].sum()
    assert sequence["pose_coverage"] > 0
    pooled = pooled_pose_features(sequence)
    assert np.isfinite(pooled).all()


def test_pose_sequences_do_not_create_motion_across_track_change_or_untrusted_pose():
    sample = {"attacker": "1", "fps": 30, "start_frame": 0, "end_frame": 3, "pose_usable": True}
    records = [_record(frame, fighter, track=1 if frame < 2 else 2) for frame in range(4) for fighter in ("1", "2")]
    sequence = extract_pose_sequence(sample, records, PoseFeatureSpec(steps=4, context_seconds=0))
    assert not sequence["velocity_mask"][2, 0].any()
    hidden = extract_pose_sequence({**sample, "pose_usable": False}, records)
    assert not hidden["point_mask"].any()
    assert hidden["pose_coverage"] == 0


def test_audited_velocity_respects_exclusions_in_skipped_source_frames():
    sample = {"attacker": "1", "fps": 30, "start_frame": 0, "end_frame": 4}
    records = [_record(frame, fighter) for frame in range(5) for fighter in ("1", "2")]
    edges = {frame: np.ones((2, 17), dtype=bool) for frame in range(1, 5)}
    edges[1][1] = False
    sequence = extract_pose_sequence(
        sample, records, PoseFeatureSpec(steps=3, context_seconds=0), audited_velocity_edges=edges,
    )
    assert sequence["frame_indices"].tolist() == [0, 2, 4]
    assert sequence["velocity_mask"][1, 0].any()
    assert not sequence["velocity_mask"][1, 1].any()
    assert sequence["velocity_mask"][2, 1].any()


def test_pose_feature_cache_preserves_requested_train_validation_order(tmp_path):
    for name in ("a", "z"):
        path = tmp_path / name / "pose.jsonl"
        path.parent.mkdir(parents=True)
        records = [_record(frame, fighter) for frame in range(4) for fighter in ("1", "2")]
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    def sample(sample_id, pose_path, label_index):
        target = [0] * 6
        target[label_index] = 1
        return {
            "sample_id": sample_id, "pose_path": pose_path, "attacker": "1", "fps": 30,
            "start_frame": 0, "end_frame": 3, "pose_usable": True,
            "target": target, "target_mask": [1] * 6,
        }
    samples = [
        sample("train-z", "z/pose.jsonl", 0),
        sample("validation-a", "a/pose.jsonl", 1),
        sample("train-z-2", "z/pose.jsonl", 2),
    ]
    destination = tmp_path / "cache.npz"
    metadata = build_pose_feature_cache(
        tmp_path, samples, destination, PoseFeatureSpec(steps=4, context_seconds=0),
    )
    assert metadata["sample_ids"] == [sample["sample_id"] for sample in samples]
    with np.load(destination, allow_pickle=False) as arrays:
        assert arrays["target"].argmax(axis=1).tolist() == [0, 1, 2]
