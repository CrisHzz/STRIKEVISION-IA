"""Stable tracking records shared by pose and temporal-action pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    """One ByteTrack observation for a person in a video frame."""

    bbox_xyxy: np.ndarray
    silhouette: np.ndarray | None
    confidence: float


@dataclass(frozen=True)
class TrackingRecord:
    """Serializable fighter state, including explicit missing-frame records."""

    frame_index: int
    timestamp_seconds: float
    fighter_id: str
    track_id: int
    bbox_xyxy: tuple[float, float, float, float] | None
    confidence: float | None
    visible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "fighter_id": self.fighter_id,
            "track_id": self.track_id,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "confidence": round(self.confidence, 6) if self.confidence is not None else None,
            "visible": self.visible,
        }
