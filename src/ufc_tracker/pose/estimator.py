
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.request import urlopen

import numpy as np

from ufc_tracker.detection.weights import project_root, weights_dir

KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
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
MIN_VALID_REQUIRED_KEYPOINTS = 4
MEDIAPIPE_POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
MEDIAPIPE_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class PoseEstimate:
    keypoints: dict[str, tuple[float, float, float] | None]
    inference_ms: float


class PoseEstimator(Protocol):
    name: str

    def estimate(
        self,
        crop_bgr: np.ndarray,
        *,
        fighter_id: str,
        offset_xy: tuple[int, int],
        timestamp_seconds: float,
    ) -> PoseEstimate:
        """Estimate canonical keypoints in full-frame coordinates."""

    def close(self) -> None:
        """Release optional backend resources."""

    def mark_missing(self, fighter_id: str) -> None:
        """Forget temporal state after a fighter is no longer visible."""


def empty_keypoints() -> dict[str, tuple[float, float, float] | None]:
    return {name: None for name in KEYPOINT_NAMES}


def _with_offset(
    values: Mapping[str, tuple[float, float, float] | None], offset_xy: tuple[int, int]
) -> dict[str, tuple[float, float, float] | None]:
    offset_x, offset_y = offset_xy
    return {
        name: None
        if point is None
        else (float(point[0] + offset_x), float(point[1] + offset_y), float(point[2]))
        for name, point in values.items()
    }


def resolve_mediapipe_pose_model(
    model_path: str | Path | None = None,
) -> Path:
    """Resolve the official MediaPipe Tasks pose bundle under ``models/weights``."""
    if model_path is not None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model not found: {path}")
        return path

    root = project_root(Path(__file__).resolve())
    destination = weights_dir(root) / MEDIAPIPE_POSE_MODEL_FILENAME
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(MEDIAPIPE_POSE_MODEL_URL, timeout=120) as response:
            with partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if partial.stat().st_size == 0:
            raise RuntimeError("MediaPipe pose model download produced an empty file")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


class MediaPipePoseEstimator:
    """MediaPipe Tasks Pose Landmarker with one video task per fighter track."""

    name = "mediapipe_pose"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        min_visibility: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self._min_visibility = min_visibility
        self.weight_path = str(resolve_mediapipe_pose_model(model_path))
        self._models: dict[str, Any] = {}

    def _model_for(self, fighter_id: str) -> Any:
        if fighter_id not in self._models:
            options = self._mp.tasks.vision.PoseLandmarkerOptions(
                base_options=self._mp.tasks.BaseOptions(
                    model_asset_path=self.weight_path
                ),
                running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self._models[fighter_id] = (
                self._mp.tasks.vision.PoseLandmarker.create_from_options(options)
            )
        return self._models[fighter_id]

    def estimate(
        self,
        crop_bgr: np.ndarray,
        *,
        fighter_id: str,
        offset_xy: tuple[int, int],
        timestamp_seconds: float,
    ) -> PoseEstimate:
        import cv2

        started = perf_counter()
        rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(round(timestamp_seconds * 1000))
        result = self._model_for(fighter_id).detect_for_video(image, timestamp_ms)
        elapsed_ms = (perf_counter() - started) * 1000
        values = empty_keypoints()
        if not result.pose_landmarks:
            return PoseEstimate(values, elapsed_ms)

        height, width = crop_bgr.shape[:2]
        mapping = {
            "nose": 0,
            "left_eye": 2,
            "right_eye": 5,
            "left_ear": 7,
            "right_ear": 8,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        }
        landmarks = result.pose_landmarks[0]
        for name, landmark_index in mapping.items():
            point = landmarks[landmark_index]
            visibility = float(point.visibility or 0.0)
            if visibility < self._min_visibility:
                continue
            values[name] = (float(point.x * width), float(point.y * height), visibility)
        return PoseEstimate(_with_offset(values, offset_xy), elapsed_ms)

    def close(self) -> None:
        for model in self._models.values():
            model.close()
        self._models.clear()

    def mark_missing(self, fighter_id: str) -> None:
        model = self._models.pop(fighter_id, None)
        if model is not None:
            model.close()
