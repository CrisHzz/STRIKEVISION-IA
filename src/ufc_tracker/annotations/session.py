"""Local persistence and media helpers for strike-window annotation."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from ufc_tracker.annotations.contracts import ALLOWED_LABELS, AnnotationConfig, validate_annotation_rows
from ufc_tracker.annotations.windows import read_jsonl, write_jsonl


@dataclass(frozen=True)
class AnnotationSummary:
    total: int
    labeled: int
    unlabeled: int
    strike: int
    no_strike: int
    unknown_occluded: int
    suggested_unknown: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "labeled": self.labeled,
            "unlabeled": self.unlabeled,
            "strike": self.strike,
            "no_strike": self.no_strike,
            "unknown_occluded": self.unknown_occluded,
            "suggested_unknown": self.suggested_unknown,
        }


class AnnotationStore:
    """In-memory annotation rows with validated, atomic local persistence."""

    def __init__(self, path: str | Path, config: AnnotationConfig) -> None:
        self.path = Path(path).resolve()
        self.config = config
        self._lock = threading.Lock()
        self.rows = read_jsonl(self.path)
        errors = validate_annotation_rows(self.rows, config)
        if errors:
            raise ValueError(
                f"Annotation file does not satisfy {config.schema_version}:\n"
                + "\n".join(f"- {error}" for error in errors)
            )

    def __len__(self) -> int:
        return len(self.rows)

    def row(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.rows):
            raise IndexError(f"Annotation index {index} is outside 0..{len(self.rows) - 1}")
        return self.rows[index]

    def next_unlabeled(self, start_index: int = 0) -> int:
        """Return the next unlabeled row, wrapping around when necessary."""
        if not self.rows:
            raise ValueError("Annotation file has no windows")
        start = start_index % len(self.rows)
        for offset in range(len(self.rows)):
            index = (start + offset) % len(self.rows)
            if self.rows[index].get("label") is None:
                return index
        return start

    def summary(self) -> AnnotationSummary:
        labels = [row.get("label") for row in self.rows]
        return AnnotationSummary(
            total=len(self.rows),
            labeled=sum(label is not None for label in labels),
            unlabeled=sum(label is None for label in labels),
            strike=labels.count("strike"),
            no_strike=labels.count("no_strike"),
            unknown_occluded=labels.count("unknown_occluded"),
            suggested_unknown=sum(
                row.get("suggested_label") == "unknown_occluded" for row in self.rows
            ),
        )

    def save_label(
        self,
        index: int,
        label: str,
        *,
        annotator: str | None,
        notes: str | None,
    ) -> dict[str, Any]:
        """Persist one human label and return the updated row."""
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported label {label!r}; expected one of {sorted(ALLOWED_LABELS)}")
        with self._lock:
            row = self.row(index)
            row["label"] = label
            row["annotator"] = annotator.strip() if annotator and annotator.strip() else None
            row["annotated_at"] = datetime.now(timezone.utc).isoformat()
            row["notes"] = notes.strip() if notes else ""
            errors = validate_annotation_rows(self.rows, self.config)
            if errors:
                raise ValueError("Cannot save invalid annotations:\n" + "\n".join(errors))
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                write_jsonl(temporary, self.rows)
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return row


class AnnotationMediaCache:
    """Create small, reusable video clips for exact annotation windows."""

    def __init__(self, project_root: str | Path, cache_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.cache_root = Path(cache_root).resolve()

    def clip_for(self, row: dict[str, Any], source: str) -> Path:
        if source not in {"original", "pose_preview"}:
            raise ValueError(f"Unsupported clip source: {source}")
        source_path = self._source_path(row, source)
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        video_id = str(row["video_id"])
        destination = (
            self.cache_root
            / video_id
            / f"{start_frame:06d}-{end_frame:06d}_{source}.mp4"
        )
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._extract_clip(source_path, destination, start_frame, end_frame, float(row["fps"]))
        return destination

    def _source_path(self, row: dict[str, Any], source: str) -> Path:
        if source == "original":
            raw_path = Path(str(row["video_path"]))
        else:
            raw_path = Path(str(row["pose_path"])).parent / "pose_preview.mp4"
        path = raw_path if raw_path.is_absolute() else self.project_root / raw_path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Could not find annotation media: {path}")
        return path

    @staticmethod
    def _extract_clip(
        source: Path,
        destination: Path,
        start_frame: int,
        end_frame: int,
        fallback_fps: float,
    ) -> None:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not open annotation media: {source}")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or fallback_fps
            if width <= 0 or height <= 0 or fps <= 0:
                raise ValueError(f"Invalid annotation media metadata: {source}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer = cv2.VideoWriter(
                str(destination),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise ValueError(f"Could not create annotation clip: {destination}")
            try:
                for frame_index in range(start_frame, end_frame + 1):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(
                            f"Could not read frame {frame_index} from annotation media: {source}"
                        )
                    writer.write(frame)
            finally:
                writer.release()
        finally:
            capture.release()

