"""Frozen X3D-S embeddings from attacker crops for the RGB tournament candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import yaml

from ufc_tracker.action_recognition.contracts import canonical_fighter_id
from ufc_tracker.action_recognition.features import PoseFeatureSpec
from ufc_tracker.strike.dataset import clip_tensor, letterbox
from ufc_tracker.strike.model import build_model


def _records(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    output = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line); fighter = canonical_fighter_id(row.get("fighter_id"))
        if fighter is not None:
            output[(int(row["frame_index"]), fighter)] = row
    return output


def _crop(frame: np.ndarray, bbox: list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if bbox is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    padding_x, padding_y = (x2 - x1) * .2, (y2 - y1) * .15
    left, top = max(0, int(np.floor(x1 - padding_x))), max(0, int(np.floor(y1 - padding_y)))
    right, bottom = min(width, int(np.ceil(x2 + padding_x))), min(height, int(np.ceil(y2 + padding_y)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _sampled_indices(
    sample: dict[str, Any], spec: PoseFeatureSpec, steps: int = 13, max_frame_index: int | None = None,
) -> np.ndarray:
    fps = float(sample["fps"]); margin = int(round(spec.context_seconds * fps / 2))
    start, end = max(0, int(sample["start_frame"]) - margin), int(sample["end_frame"]) + margin
    if max_frame_index is not None and max_frame_index >= 0:
        start, end = min(start, max_frame_index), min(end, max_frame_index)
    return np.rint(np.linspace(start, end, steps)).astype(int)


def iter_attacker_clips(root: Path, samples: list[dict[str, Any]], spec: PoseFeatureSpec) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
    """Decode each source video sequentially once and retain only requested crops."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["video_path"], []).append(sample)
    for relative_video, tasks in sorted(grouped.items()):
        pose_index = _records(root / tasks[0]["pose_path"])
        capture = cv2.VideoCapture(str(root / relative_video))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {relative_video}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frame_index = frame_count - 1 if frame_count > 0 else None
            bounded_tasks = [
                (sample, _sampled_indices(sample, spec, max_frame_index=max_frame_index))
                for sample in tasks
            ]
            by_frame: dict[int, list[tuple[int, int]]] = {}
            crops: list[list[np.ndarray | None]] = [[None] * len(indices) for _, indices in bounded_tasks]
            for task_index, (_, indices) in enumerate(bounded_tasks):
                for position, frame_index in enumerate(indices):
                    by_frame.setdefault(int(frame_index), []).append((task_index, position))
            max_index = max(by_frame)
            for frame_index in range(max_index + 1):
                ok, frame = capture.read()
                if not ok:
                    # Do not fabricate pixels after a genuine decoder failure. Any
                    # affected sample remains incomplete and is excluded below.
                    break
                for task_index, position in by_frame.get(frame_index, []):
                    sample, _ = bounded_tasks[task_index]
                    pose = pose_index.get((frame_index, str(sample["attacker"])))
                    crop = None if pose is None or not pose.get("visible") else _crop(frame, pose.get("bbox_xyxy"))
                    if crop is not None:
                        crops[task_index][position] = letterbox(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        for (sample, _), frames in zip(bounded_tasks, crops, strict=True):
            if any(frame is None for frame in frames):
                continue
            yield sample, clip_tensor(frames)  # type: ignore[arg-type]


def build_x3d_embeddings(root: Path, samples: list[dict[str, Any]], destination: Path, spec: PoseFeatureSpec, batch_size: int = 2) -> dict[str, Any]:
    """Cache frozen embeddings; this is extraction only, never X3D fine-tuning."""
    root, destination = Path(root), Path(destination)
    config = yaml.safe_load((root / "configs/models/strike_x3d_s.yaml").read_text(encoding="utf-8"))
    extractor = config["extractor"]
    model = build_model(root / extractor["weights_path"], extractor["weights_sha256"]).eval()
    selected, vectors, pending_samples, pending_tensors = [], [], [], []
    def flush() -> None:
        if not pending_tensors:
            return
        with torch.no_grad():
            vectors.extend(model.embeddings(torch.stack(pending_tensors)).numpy())
        selected.extend(pending_samples); pending_samples.clear(); pending_tensors.clear()
    for sample, tensor in iter_attacker_clips(root, samples, spec):
        pending_samples.append(sample); pending_tensors.append(tensor)
        if len(pending_tensors) >= batch_size:
            flush()
    flush()
    if not selected:
        raise ValueError("No RGB clips had a complete attacker crop")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, embeddings=np.asarray(vectors, dtype=np.float32),
                                target=np.asarray([sample["target"] for sample in selected], dtype=np.float32),
                                target_mask=np.asarray([sample["target_mask"] for sample in selected], dtype=bool))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {"schema_version": "action_x3d_embeddings_v1", "sample_ids": [sample["sample_id"] for sample in selected],
                "requested_samples": len(samples), "embedded_samples": len(selected), "feature_spec": spec.__dict__,
                "embedding_dim": int(np.asarray(vectors).shape[1]), "batch_size": batch_size}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
