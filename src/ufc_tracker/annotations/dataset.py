"""Read annotated intervals without changing labels, time support or athlete identity.

Slots are technical channels, NOT guaranteed athlete identities. Missing poses
remain masked/NaN. Durations and derivatives use source FPS, not an assumed 30 FPS.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ufc_tracker.annotations.contracts import load_annotation_config, validate_annotation_rows
from ufc_tracker.annotations.windows import _quality_for_window, read_jsonl

SLOTS = ("fighter_0", "fighter_1")
SLOT_ALIASES = {
    "fighter_left": "fighter_0", "fighter_right": "fighter_1",
    "1": "fighter_0", "2": "fighter_1",
    "fighter_0": "fighter_0", "fighter_1": "fighter_1",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_pose_records(records: list[dict]) -> list[dict]:
    """Normalize channel names without mutating the stored JSONL or spatial order."""
    output = []
    seen = set()
    for row in records:
        raw_slot = str(row["fighter_id"])
        if raw_slot not in SLOT_ALIASES:
            raise ValueError(f"Unknown fighter slot: {raw_slot!r}")
        slot = SLOT_ALIASES[raw_slot]
        key = (int(row["frame_index"]), slot)
        if key in seen:
            raise ValueError(f"Duplicate fighter/frame after slot normalization: {key}")
        seen.add(key)
        output.append({**row, "fighter_id": slot})
    return output


def select_annotation_files(root: Path, policy_path: Path) -> tuple[list[Path], dict]:
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "strike_dataset_v1":
        raise ValueError("Unsupported dataset policy")
    if policy.get("retain_low_pose_quality") is not True:
        raise ValueError("This dataset must preserve low-quality human annotations")
    excluded = policy["excluded_files"]
    paths = sorted((root / policy["annotations_dir"]).glob("*.jsonl"))
    selected = [path for path in paths if path.name not in excluded]
    if not selected:
        raise ValueError("No included annotation files")
    return selected, policy


def inspect_dataset(root: Path, policy_path: Path) -> tuple[list[dict], dict, dict]:
    """Build provenance-rich manifest and refreshed annotation copies, without writes."""
    files, policy = select_annotation_files(root, policy_path)
    config = load_annotation_config(root / policy["annotation_config"])
    with (root / policy["splits_manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        videos = {Path(row["relative_path"]).stem: row for row in csv.DictReader(stream)}
    manifest: list[dict] = []
    refreshed: dict[Path, list[dict]] = {}
    sources = {}
    seen_ids = set()
    seen_intervals = set()
    for path in files:
        rows = read_jsonl(path)
        errors = validate_annotation_rows(rows, config, require_labeled=True)
        if errors:
            raise ValueError(f"{path.name}: {errors[:5]}")
        if not rows or len({row["video_id"] for row in rows}) != 1:
            raise ValueError(f"Expected one complete round in {path.name}")
        first = rows[0]
        video_id = first["video_id"]
        if video_id not in videos:
            raise ValueError(f"Unrecognized video/pilot must be explicitly excluded: {video_id}")
        video = root / first["video_path"]
        pose = root / first["pose_path"]
        if not video.is_file() or not pose.is_file():
            raise FileNotFoundError(f"Missing video or pose for {video_id}")
        if video.resolve() != (root / videos[video_id]["relative_path"]).resolve():
            raise ValueError(f"Video path disagrees with split manifest: {video_id}")
        pose_rows = canonical_pose_records(read_jsonl(pose))
        metrics = json.loads((pose.parent / "pose_metrics.json").read_text(encoding="utf-8"))
        if abs(float(metrics["fps"]) - float(first["fps"])) > 1e-6:
            raise ValueError(f"Annotation/pose FPS mismatch: {video_id}")
        by_frame: dict[int, list[dict]] = defaultdict(list)
        for record in pose_rows:
            by_frame[record["frame_index"]].append(record)
        frame_count = max(by_frame, default=-1) + 1
        if frame_count != metrics["frame_count"]:
            raise ValueError(f"Incomplete pose artifact: {video_id}")
        repaired = []
        for row in rows:
            if any(row[key] != first[key] for key in ("pose_path", "video_path", "fps")):
                raise ValueError(f"Mixed source contracts in {path.name}")
            interval = (str(video.resolve()), row["start_frame"], row["end_frame"])
            if row["window_id"] in seen_ids or interval in seen_intervals:
                raise ValueError(f"Duplicate annotated interval: {row['window_id']}")
            seen_ids.add(row["window_id"])
            seen_intervals.add(interval)
            if row["end_frame"] >= frame_count:
                raise ValueError(f"Annotation exceeds pose data: {row['window_id']}")
            quality = _quality_for_window(
                by_frame, row["start_frame"], row["end_frame"], config
            )
            updated = {
                **row, "quality": quality,
                "suggested_label": "unknown_occluded" if quality["reasons"] else None,
            }
            repaired.append(updated)
            manifest.append({
                **updated,
                "fight_id": videos[video_id]["fight_id"],
                "category": videos[video_id]["category"],
                "annotation_file": path.relative_to(root).as_posix(),
                "binary_eligible": row["label"] in policy["training_labels"],
                "pose_usable": video_id not in policy.get("untrusted_pose_videos", {}),
                "pose_unusable_reason": policy.get("untrusted_pose_videos", {}).get(video_id),
                "duration_seconds": row["frame_count"] / row["fps"],
                "sample_span_seconds": (row["end_frame"] - row["start_frame"]) / row["fps"],
                "frame_interval_seconds": 1.0 / row["fps"],
                "canonical_slots": list(SLOTS),
            })
        refreshed[path] = repaired
        metadata_path = pose.parent / "run_metadata.json"
        sources[video_id] = {
            "pose_sha256": sha256(pose),
            "pose_path": first["pose_path"],
            "annotation_sha256_before": sha256(path),
            "run_metadata_sha256": sha256(metadata_path),
        }
    return manifest, refreshed, {"sources": sources, "policy": policy}


def load_pose_window(
    row: dict, records: list[dict], keypoints: tuple[str, ...], *, steps: int | None = None,
) -> dict[str, Any]:
    """Load only the human-labeled interval. Optional sampling never adds context.

    Resampling selects existing frames (no invented keypoints). Physical sample
    times remain explicit; repeated frames have dt=0 and no valid derivative.
    """
    fps = float(row["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    start, end = int(row["start_frame"]), int(row["end_frame"])
    count = end - start + 1
    if count <= 0 or start < 0 or count != int(row["frame_count"]):
        raise ValueError("Invalid annotated interval")
    if steps is not None and steps < 2:
        raise ValueError("Resampling needs at least two steps")
    positions = np.full((count, 2, len(keypoints), 2), np.nan, dtype=np.float64)
    confidence = np.zeros((count, 2, len(keypoints)), dtype=np.float64)
    mask = np.zeros((count, 2, len(keypoints)), dtype=bool)
    track_ids = np.full((count, 2), -1, dtype=np.int64)
    pose_usable = row.get("pose_usable", True)
    selected = [r for r in records if start <= r["frame_index"] <= end] if pose_usable else []
    for record in canonical_pose_records(selected):
        frame = int(record["frame_index"]) - start
        slot = SLOTS.index(record["fighter_id"])
        track_ids[frame, slot] = int(record.get("track_id", -1))
        if "timestamp_seconds" in record and abs(
            float(record["timestamp_seconds"]) - record["frame_index"] / fps
        ) > 1e-4:
            raise ValueError("Pose timestamps disagree with annotation FPS")
        if not record.get("visible") or not record.get("pose_valid"):
            continue
        for joint, name in enumerate(keypoints):
            point = record.get("keypoints", {}).get(name)
            if point is None:
                continue
            values = np.asarray(point, dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                continue
            if not 0 < values[2] <= 1:
                continue
            positions[frame, slot, joint] = values[:2]
            confidence[frame, slot, joint] = values[2]
            mask[frame, slot, joint] = True
    indices = np.arange(count) if steps is None else np.rint(
        np.linspace(0, count - 1, steps)
    ).astype(int)
    timestamps = (start + indices) / fps
    velocity = np.full((len(indices), 2, len(keypoints), 2), np.nan)
    velocity_mask = np.zeros((len(indices), 2, len(keypoints)), dtype=bool)
    for i in range(1, len(indices)):
        a, b = indices[i - 1], indices[i]
        if b <= a:
            continue
        # Never differentiate across a missing observation, even after subsampling.
        valid = mask[a:b + 1].all(axis=0)
        # Do not derive a motion spike from a tracker reassignment/camera cut.
        continuous = (track_ids[a:b + 1] == track_ids[a]).all(axis=0)
        valid &= continuous[:, None]
        velocity_mask[i] = valid
        delta = (positions[b] - positions[a]) / ((b - a) / fps)
        velocity[i][valid] = delta[valid]
    return {
        "window_id": row["window_id"], "label": row["label"],
        "pose_usable": pose_usable,
        "frame_indices": start + indices, "timestamps_seconds": timestamps,
        "duration_seconds": count / fps, "sample_span_seconds": (count - 1) / fps,
        "positions": positions[indices], "confidence": confidence[indices],
        "valid_mask": mask[indices], "velocity_pixels_per_second": velocity,
        "velocity_mask": velocity_mask, "slots": SLOTS,
    }
