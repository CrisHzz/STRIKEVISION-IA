"""Audited human windows and exact sequential RGB decoding (no approximate seeks)."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import av
import cv2
import numpy as np
import torch
import yaml

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.annotations.windows import read_jsonl

CLASS_MAP = {"no_strike": 0, "strike": 1}
MANIFEST = Path("data/processed/datasets/strike_dataset_v1/binary_windows.jsonl")


def validate_window(row: dict) -> None:
    if row["label"] not in CLASS_MAP:
        raise ValueError("Only human binary labels are permitted")
    for key in ("start_frame", "end_frame", "frame_count"):
        if type(row[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    start, end, fps = row["start_frame"], row["end_frame"], row["fps"]
    if not math.isfinite(fps) or fps <= 0 or start < 0 or end < start:
        raise ValueError("Invalid interval or FPS")
    if end - start + 1 != row["frame_count"]:
        raise ValueError("Frame bounds must be inclusive")
    expected = {"start_seconds": start / fps, "end_seconds": end / fps,
                "duration_seconds": row["frame_count"] / fps,
                "sample_span_seconds": (end - start) / fps}
    for key, value in expected.items():
        if not math.isclose(row[key], value, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Inconsistent {key}: {row['window_id']}")


def load_manifest(root: Path) -> tuple[list[dict], str]:
    """Verify the existing audit and source labels without requiring pose files."""
    path = root / MANIFEST
    audit = json.loads((path.parent / "audit.json").read_text(encoding="utf-8"))
    digest = sha256(path)
    if digest != audit["binary_manifest_sha256"]:
        raise ValueError("Binary manifest changed since dataset audit")
    policy_path = root / "configs/data/strike_dataset_v1.yaml"
    if sha256(policy_path) != audit["policy_sha256"]:
        raise ValueError("Dataset policy changed since audit")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    rows = read_jsonl(path)
    seen, intervals, checked_sources, video_contracts = set(), set(), set(), {}
    for row in rows:
        validate_window(row)
        if Path(row["annotation_file"]).name in policy["excluded_files"]:
            raise ValueError("Excluded annotation leaked into RGB manifest")
        interval = (row["video_path"], row["start_frame"], row["end_frame"])
        if row["window_id"] in seen or interval in intervals:
            raise ValueError("Duplicate window or interval")
        seen.add(row["window_id"])
        intervals.add(interval)
        contract = (row["video_path"], row["fight_id"], row["fps"])
        if video_contracts.setdefault(row["video_id"], contract) != contract:
            raise ValueError("Inconsistent video contract")
        if not (root / row["video_path"]).is_file():
            raise FileNotFoundError(row["video_path"])
        if row["annotation_file"] not in checked_sources:
            source = root / row["annotation_file"]
            expected = audit["sources"][row["video_id"]]["annotation_sha256"]
            if sha256(source) != expected:
                raise ValueError(f"Annotation changed since audit: {source}")
            checked_sources.add(row["annotation_file"])
    if len(rows) != audit["binary_windows"]:
        raise ValueError("Audit count mismatch")
    review = path.parent / "review_windows.jsonl"
    if sha256(review) != audit["review_manifest_sha256"]:
        raise ValueError("Review manifest changed since audit")
    return rows, digest


def sample_indices(row: dict, steps: int = 13) -> np.ndarray:
    validate_window(row)
    if steps < 2 or row["frame_count"] < steps:
        raise ValueError("Need at least steps distinct annotated frames")
    return np.rint(np.linspace(row["start_frame"], row["end_frame"], steps)).astype(int)


def letterbox(rgb: np.ndarray, size: int = 182) -> np.ndarray:
    """Preserve the entire frame; black padding is explicit, before normalization."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB HWC")
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    result = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    result[top:top + nh, left:left + nw] = resized
    return result


def clip_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255
    tensor = (tensor - 0.45) / 0.225
    if not torch.isfinite(tensor).all():
        raise ValueError("Non-finite RGB tensor")
    return tensor.contiguous()


def _timed_decode(container, stream):
    decoder = iter(container.decode(stream))
    index = 0
    while True:
        started = perf_counter()
        try:
            frame = next(decoder)
        except StopIteration:
            return
        elapsed = perf_counter() - started
        yield index, frame, elapsed
        index += 1


def iter_rgb_windows(root: Path, rows: list[dict], *, size: int = 182, steps: int = 13):
    """Open each video once; decode from frame zero, checking presentation timestamps.

    The buffer holds at most the largest annotated interval in resized uint8 RGB.
    Sampled frame indices are mapped explicitly to their decoded PTS. Variable
    or missing time intervals are reported rather than silently replacing the
    human frame bounds. Non-monotonic PTS still fail closed.
    Rows are yielded in video/frame order, not arbitrary input order.
    """
    groups = defaultdict(list)
    for row in rows:
        validate_window(row)
        groups[row["video_path"]].append(row)
    for relative, group in sorted(groups.items()):
        group.sort(key=lambda r: (r["end_frame"], r["start_frame"]))
        fps = group[0]["fps"]
        if any(r["fps"] != fps for r in group):
            raise ValueError("Mixed FPS for the same source")
        capacity = max(r["frame_count"] for r in group)
        buffer, stamps = {}, {}
        pending = 0
        decode_seconds = transform_seconds = 0.0
        timestamp_discrepancies = []
        with av.open(str(root / relative)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            actual_fps = float(stream.average_rate or 0)
            if not math.isclose(actual_fps, fps, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"Video/annotation FPS mismatch: {relative}")
            origin = None
            previous_stamp = None
            for index, frame, decode_elapsed in _timed_decode(container, stream):
                decode_seconds += decode_elapsed
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Missing PTS; explicit index/timestamp mapping required")
                stamp = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = stamp
                expected_stamp = origin + index / fps
                drift = stamp - expected_stamp
                if index and stamp <= previous_stamp:
                    raise ValueError(
                        f"Non-monotonic PTS at frame {index}: {relative}; "
                        "an explicit external frame mapping is required"
                    )
                if abs(drift) > 0.001:
                    timestamp_discrepancies.append({
                        "frame_index": index,
                        "pts_seconds": stamp,
                        "expected_cfr_seconds": expected_stamp,
                        "drift_seconds": drift,
                        "previous_delta_seconds": stamp - previous_stamp,
                    })
                previous_stamp = stamp
                # Resize only frames that can belong to a pending annotated window.
                if index >= group[pending]["start_frame"]:
                    transform_started = perf_counter()
                    buffer[index] = letterbox(frame.to_ndarray(format="rgb24"), size)
                    transform_seconds += perf_counter() - transform_started
                    stamps[index] = stamp
                for old in list(buffer):
                    if old <= index - capacity:
                        del buffer[old]
                        del stamps[old]
                while pending < len(group) and group[pending]["end_frame"] == index:
                    row = group[pending]
                    indices = sample_indices(row, steps).tolist()
                    transform_started = perf_counter()
                    tensor = clip_tensor([buffer[i] for i in indices])
                    transform_seconds += perf_counter() - transform_started
                    yield {
                        "row": row, "tensor": tensor,
                        "sampled_indices": indices, "sampled_pts_seconds": [stamps[i] for i in indices],
                        "pts_origin_seconds": origin, "source_fps": actual_fps,
                        "timestamp_mapping": "decoded_frame_index_to_pts",
                        "timestamp_discrepancy_count": len(timestamp_discrepancies),
                        "max_abs_timestamp_drift_seconds": max(
                            (abs(item["drift_seconds"]) for item in timestamp_discrepancies),
                            default=0.0,
                        ),
                        "latest_timestamp_discrepancy": (
                            timestamp_discrepancies[-1] if timestamp_discrepancies else None
                        ),
                        "buffer_frames": len(buffer), "decoded_frames_so_far": index + 1,
                        "decode_seconds_cumulative": decode_seconds,
                        "transform_seconds_cumulative": transform_seconds,
                    }
                    pending += 1
                if pending == len(group):
                    break
            if pending != len(group):
                raise ValueError(f"Premature EOF: {relative}")


def split_summary(rows: list[dict], assignments: dict[str, list[str]]) -> dict:
    flat = [fight for group in assignments.values() for fight in group]
    if set(assignments) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test")
    if len(flat) != len(set(flat)) or set(flat) != {r["fight_id"] for r in rows}:
        raise ValueError("Fights must occur in exactly one partition")
    result = {}
    for split, fights in assignments.items():
        selected = [r for r in rows if r["fight_id"] in fights]
        if set(r["label"] for r in selected) != set(CLASS_MAP):
            raise ValueError(f"Both classes required in {split}")
        result[split] = {"fights": fights, "windows": len(selected),
                         "labels": dict(Counter(r["label"] for r in selected)),
                         "categories": sorted({r["category"] for r in selected}),
                         "fps": sorted({r["fps"] for r in selected})}
    return result
