"""Pose-sequence features for action recognition without invented observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ufc_tracker.action_recognition.contracts import canonical_fighter_id
from ufc_tracker.pose.estimator import KEYPOINT_NAMES


@dataclass(frozen=True)
class PoseFeatureSpec:
    """Temporal context is expressed in seconds so mixed source FPS remains valid."""

    steps: int = 32
    context_seconds: float = 0.8

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError("Pose feature extraction requires at least two steps")
        if self.context_seconds < 0:
            raise ValueError("Context seconds cannot be negative")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _point(record: dict[str, Any], joint: str) -> tuple[float, float, float] | None:
    point = record.get("keypoints", {}).get(joint)
    if point is None or len(point) != 3:
        return None
    x, y, confidence = (float(value) for value in point)
    if not np.isfinite((x, y, confidence)).all() or not 0 < confidence <= 1:
        return None
    return x, y, confidence


def _origin_and_scale(record: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """Use the attacker's own body frame; missing joints never receive substitutes."""
    pairs = (("left_hip", "right_hip"), ("left_shoulder", "right_shoulder"))
    for left_name, right_name in pairs:
        left, right = _point(record, left_name), _point(record, right_name)
        if left is not None and right is not None:
            left_xy, right_xy = np.asarray(left[:2]), np.asarray(right[:2])
            scale = float(np.linalg.norm(right_xy - left_xy))
            if scale > 1e-6:
                return (left_xy + right_xy) / 2, scale
    return None


def _record_index(records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        fighter = canonical_fighter_id(record.get("fighter_id"))
        if fighter is None:
            continue
        key = int(record["frame_index"]), fighter
        if key in indexed:
            raise ValueError(f"Duplicate pose record after fighter-ID normalization: {key}")
        indexed[key] = record
    return indexed


def extract_pose_sequence(
    sample: dict[str, Any], records: list[dict[str, Any]], spec: PoseFeatureSpec = PoseFeatureSpec(),
) -> dict[str, np.ndarray]:
    """Create a fixed-length attacker-first sequence with explicit validity masks.

    No coordinates are interpolated.  A frame without a valid attacker body
    frame is represented with zero-valued features and zero masks, allowing
    the caller to abstain or use RGB instead.
    """
    fps = float(sample["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Sample FPS must be finite and positive")
    attacker = canonical_fighter_id(sample["attacker"])
    if attacker is None:
        raise ValueError("Pose sequence requires a known attacker ID")
    opponent = "2" if attacker == "1" else "1"
    start, end = int(sample["start_frame"]), int(sample["end_frame"])
    if start < 0 or end < start:
        raise ValueError("Invalid sample frame interval")
    margin = int(round(spec.context_seconds * fps / 2))
    window_start, window_end = max(0, start - margin), end + margin
    frame_indices = np.rint(np.linspace(window_start, window_end, spec.steps)).astype(np.int64)
    index = _record_index(records)
    joints = len(KEYPOINT_NAMES)
    positions = np.zeros((spec.steps, 2, joints, 2), dtype=np.float32)
    confidence = np.zeros((spec.steps, 2, joints), dtype=np.float32)
    point_mask = np.zeros((spec.steps, 2, joints), dtype=bool)
    frame_mask = np.zeros((spec.steps, 2), dtype=bool)
    track_ids = np.full((spec.steps, 2), -1, dtype=np.int64)

    for index_position, frame_index in enumerate(frame_indices):
        attacker_record = index.get((int(frame_index), attacker))
        if not sample.get("pose_usable", True) or attacker_record is None:
            continue
        if not attacker_record.get("visible") or not attacker_record.get("pose_valid"):
            continue
        body_frame = _origin_and_scale(attacker_record)
        if body_frame is None:
            continue
        origin, scale = body_frame
        for fighter_position, fighter in enumerate((attacker, opponent)):
            record = index.get((int(frame_index), fighter))
            if record is None or not record.get("visible") or not record.get("pose_valid"):
                continue
            frame_mask[index_position, fighter_position] = True
            track_ids[index_position, fighter_position] = int(record.get("track_id", -1))
            for joint_position, joint in enumerate(KEYPOINT_NAMES):
                point = _point(record, joint)
                if point is None:
                    continue
                positions[index_position, fighter_position, joint_position] = (
                    np.asarray(point[:2], dtype=np.float32) - origin
                ) / scale
                confidence[index_position, fighter_position, joint_position] = point[2]
                point_mask[index_position, fighter_position, joint_position] = True

    velocity = np.zeros_like(positions)
    velocity_mask = np.zeros_like(point_mask)
    for time_index in range(1, spec.steps):
        elapsed_frames = int(frame_indices[time_index] - frame_indices[time_index - 1])
        if elapsed_frames <= 0:
            continue
        continuous = track_ids[time_index] == track_ids[time_index - 1]
        valid = point_mask[time_index] & point_mask[time_index - 1] & continuous[:, None]
        velocity[time_index][valid] = (
            positions[time_index][valid] - positions[time_index - 1][valid]
        ) / (elapsed_frames / fps)
        velocity_mask[time_index] = valid

    values = np.concatenate(
        (
            positions.reshape(spec.steps, -1),
            velocity.reshape(spec.steps, -1),
            confidence.reshape(spec.steps, -1),
            point_mask.astype(np.float32).reshape(spec.steps, -1),
            frame_mask.astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    return {
        "features": values,
        "frame_indices": frame_indices,
        "point_mask": point_mask,
        "velocity_mask": velocity_mask,
        "frame_mask": frame_mask,
        "pose_coverage": np.asarray(point_mask[:, 0].mean(), dtype=np.float32),
        "attacker_id": np.asarray(int(attacker), dtype=np.int8),
    }


def pooled_pose_features(sequence: dict[str, np.ndarray]) -> np.ndarray:
    """A compact deterministic baseline representation from a pose sequence."""
    values = sequence["features"].astype(np.float32)
    valid = sequence["frame_mask"][:, 0]
    if not valid.any():
        return np.zeros(values.shape[1] * 3 + 1, dtype=np.float32)
    selected = values[valid]
    return np.concatenate((selected.mean(axis=0), selected.std(axis=0), selected.max(axis=0), sequence["pose_coverage"].reshape(1)))


def _samples_sha256(samples: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(sample, sort_keys=True, separators=(",", ":")) for sample in samples)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pose_feature_cache(
    root: Path,
    samples: list[dict[str, Any]],
    destination: Path,
    spec: PoseFeatureSpec = PoseFeatureSpec(),
) -> dict[str, Any]:
    """Extract a selected sample set once and save its hashes and masks beside it."""
    root, destination = Path(root), Path(destination)
    by_pose: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for position, sample in enumerate(samples):
        by_pose.setdefault(sample["pose_path"], []).append((position, sample))
    sequences: list[tuple[dict[str, Any], dict[str, np.ndarray]] | None] = [None] * len(samples)
    for pose_path, grouped_samples in sorted(by_pose.items()):
        records = _jsonl(root / pose_path)
        for position, sample in grouped_samples:
            sequences[position] = (sample, extract_pose_sequence(sample, records, spec))
    if not sequences:
        raise ValueError("No samples selected for pose feature cache")
    if any(sequence is None for sequence in sequences):
        raise RuntimeError("Pose feature cache did not preserve every requested sample")
    ordered_sequences = [sequence for sequence in sequences if sequence is not None]
    arrays = {
        "features": np.stack([sequence["features"] for _, sequence in ordered_sequences]),
        "frame_indices": np.stack([sequence["frame_indices"] for _, sequence in ordered_sequences]),
        "frame_mask": np.stack([sequence["frame_mask"] for _, sequence in ordered_sequences]),
        "point_mask": np.stack([sequence["point_mask"] for _, sequence in ordered_sequences]),
        "velocity_mask": np.stack([sequence["velocity_mask"] for _, sequence in ordered_sequences]),
        "pose_coverage": np.asarray([sequence["pose_coverage"] for _, sequence in ordered_sequences]),
        "target": np.asarray([sample["target"] for sample, _ in ordered_sequences], dtype=np.float32),
        "target_mask": np.asarray([sample["target_mask"] for sample, _ in ordered_sequences], dtype=bool),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "schema_version": "pose_feature_cache_v1", "samples_sha256": _samples_sha256(samples),
        "sample_ids": [sample["sample_id"] for sample, _ in ordered_sequences], "spec": asdict(spec),
        "shape": list(arrays["features"].shape), "feature_dim": int(arrays["features"].shape[-1]),
        "pose_coverage_mean": float(arrays["pose_coverage"].mean()),
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
