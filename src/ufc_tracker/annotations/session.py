"""Local persistence and media helpers for strike-window annotation."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


class AnnotationMediaResolver:
    """Resolve the two full videos used for browser-side annotation playback."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def source_for(self, row: dict[str, Any], source: str) -> Path:
        if source not in {"original", "pose_preview"}:
            raise ValueError(f"Unsupported media source: {source}")
        if source == "original":
            raw_path = Path(str(row["video_path"]))
        else:
            raw_path = Path(str(row["pose_path"])).parent / "pose_preview.mp4"
        path = raw_path if raw_path.is_absolute() else self.project_root / raw_path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Could not find annotation media: {path}")
        return path
