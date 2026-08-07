"""Group ByteTrack fragments into two stable fighter identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ufc_tracker.tracking.contracts import TrackObservation, TrackingRecord

# Appearance gates inherited from person detection: keep fragments that look like
# a shirtless fighter and drop crowd, corner staff and camera operators.
FIGHTER_SKIN_MIN = 0.50
FIGHTER_MIN_AREA = 0.01

# Below this length a fragment is detection noise rather than a camera cut.
MIN_FRAGMENT_FRAMES = 5

# Two fragments may only belong to the same fighter when the gap between them is
# short (60 frames is 2 s at 30 fps) and the box barely moved. The jump is
# measured as a fraction of the box diagonal so the threshold survives zoom.
MAX_MERGE_GAP_FRAMES = 60
MAX_MERGE_DISTANCE = 0.5

# Stable identities, ordered left to right so preview colors stay consistent.
FIGHTER_LABELS = ("fighter_left", "fighter_right")

# Track id written on frames where a fighter has no box.
MISSING_TRACK_ID = -1


@dataclass(frozen=True)
class TrackFragment:
    """One uninterrupted ByteTrack identity, summarised for merging."""

    track_id: int
    frames: frozenset[int]
    first_frame: int
    last_frame: int
    first_centroid: tuple[float, float]
    last_centroid: tuple[float, float]
    diagonal: float
    mean_cx: float
    skin: float
    area: float


@dataclass
class FighterSlot:
    """A chain of fragments believed to belong to the same fighter."""

    track_ids: list[int]
    frames: set[int]
    first_frame: int
    first_centroid: tuple[float, float]
    last_frame: int
    last_centroid: tuple[float, float]
    diagonal: float
    mean_cx: float

    @classmethod
    def from_fragment(cls, fragment: TrackFragment) -> FighterSlot:
        return cls(
            track_ids=[fragment.track_id],
            frames=set(fragment.frames),
            first_frame=fragment.first_frame,
            first_centroid=fragment.first_centroid,
            last_frame=fragment.last_frame,
            last_centroid=fragment.last_centroid,
            diagonal=fragment.diagonal,
            mean_cx=fragment.mean_cx,
        )

    def absorb(self, fragment: TrackFragment) -> None:
        """Extend the chain with a fragment that starts after the current tail."""
        previous_frames = len(self.frames)
        self.track_ids.append(fragment.track_id)
        self.frames |= fragment.frames
        self.last_frame = fragment.last_frame
        self.last_centroid = fragment.last_centroid
        self.diagonal = fragment.diagonal
        self.mean_cx = float(
            (
                (self.mean_cx * previous_frames)
                + (fragment.mean_cx * len(fragment.frames))
            )
            / len(self.frames)
        )

    def absorb_slot(self, other: FighterSlot) -> None:
        """Combine a disjoint slot while preserving chronological endpoints."""
        if self.frames & other.frames:
            raise ValueError("Cannot combine fighter slots that overlap in time.")

        own_frames = len(self.frames)
        other_frames = len(other.frames)
        if other.first_frame < self.first_frame:
            self.track_ids = [*other.track_ids, *self.track_ids]
            self.first_frame = other.first_frame
            self.first_centroid = other.first_centroid
        else:
            self.track_ids.extend(other.track_ids)

        if other.last_frame > self.last_frame:
            self.last_frame = other.last_frame
            self.last_centroid = other.last_centroid
            self.diagonal = other.diagonal

        self.frames |= other.frames
        self.mean_cx = float(
            ((self.mean_cx * own_frames) + (other.mean_cx * other_frames))
            / len(self.frames)
        )


@dataclass(frozen=True)
class MergedTracking:
    """Fighter tracks rebuilt from merged fragments, plus the settings used."""

    records: list[TrackingRecord]
    fps: float
    frame_count: int
    slots: list[FighterSlot]
    confidence: float
    min_fragment_frames: int
    max_gap_frames: int
    max_distance: float
    candidate_track_ids: tuple[int, ...]
    unassigned_track_ids: tuple[int, ...]

    def metadata(self) -> dict[str, object]:
        """Tracking block for run_metadata.json, including the merge audit trail."""
        return {
            "component": "merged_person_detector",
            "model": "PersonDetector@production",
            "tracker": "bytetrack.yaml",
            "confidence": self.confidence,
            "fragment_merge": {
                "min_fragment_frames": self.min_fragment_frames,
                "max_gap_frames": self.max_gap_frames,
                "max_normalized_distance": self.max_distance,
            },
            "fighter_id_policy": "merged_slot_<left|right>",
            "track_fragment_count": sum(len(slot.track_ids) for slot in self.slots),
            "candidate_fragment_count": len(self.candidate_track_ids),
            "merged_fragments": {
                label: list(slot.track_ids)
                for label, slot in zip(FIGHTER_LABELS, self.slots)
            },
            "unassigned_fragments": list(self.unassigned_track_ids),
        }


def describe_fragments(
    per_frame: list[dict[int, TrackObservation]],
    stats: dict[int, dict[str, float]],
) -> dict[int, TrackFragment]:
    """Summarise each ByteTrack id: span, entry/exit centroids and appearance."""
    frames: dict[int, list[int]] = defaultdict(list)
    centroids: dict[int, list[tuple[float, float]]] = defaultdict(list)
    diagonals: dict[int, list[float]] = defaultdict(list)
    for frame_index, frame_map in enumerate(per_frame):
        for track_id, observation in frame_map.items():
            x1, y1, x2, y2 = (float(value) for value in observation.bbox_xyxy)
            frames[track_id].append(frame_index)
            centroids[track_id].append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
            diagonals[track_id].append(float(np.hypot(x2 - x1, y2 - y1)))

    fragments: dict[int, TrackFragment] = {}
    for track_id, frame_indices in frames.items():
        count = stats[track_id]["n"]
        if count <= 0:
            continue
        fragments[track_id] = TrackFragment(
            track_id=track_id,
            frames=frozenset(frame_indices),
            first_frame=frame_indices[0],
            last_frame=frame_indices[-1],
            first_centroid=centroids[track_id][0],
            last_centroid=centroids[track_id][-1],
            diagonal=float(np.median(diagonals[track_id])),
            mean_cx=float(np.mean([point[0] for point in centroids[track_id]])),
            skin=stats[track_id]["skin"] / count,
            area=stats[track_id]["area"] / count,
        )
    return fragments


def select_fragment_candidates(
    fragments: dict[int, TrackFragment],
    *,
    skin_min: float = FIGHTER_SKIN_MIN,
    min_area: float = FIGHTER_MIN_AREA,
    min_frames: int = MIN_FRAGMENT_FRAMES,
) -> dict[int, TrackFragment]:
    """Drop fragments that cannot be a fighter before any chaining happens."""
    return {
        track_id: fragment
        for track_id, fragment in fragments.items()
        if fragment.skin >= skin_min
        and fragment.area >= min_area
        and len(fragment.frames) >= min_frames
    }


def _normalized_distance(slot: FighterSlot, fragment: TrackFragment) -> float:
    """On-screen jump between the slot tail and the fragment head, scale free."""
    ax, ay = slot.last_centroid
    bx, by = fragment.first_centroid
    scale = max(1.0, (slot.diagonal + fragment.diagonal) * 0.5)
    return float(np.hypot(bx - ax, by - ay)) / scale


def merge_fragments(
    fragments: dict[int, TrackFragment],
    *,
    max_gap_frames: int = MAX_MERGE_GAP_FRAMES,
    max_distance: float = MAX_MERGE_DISTANCE,
) -> list[FighterSlot]:
    """Chain fragments that are disjoint in time, close in time and close on screen.

    Fragments that share a frame are different people, so a slot can never hold
    two boxes in the same frame. Keeping two slots therefore guarantees at most
    two fighters per frame without any further validation downstream.
    """
    slots: list[FighterSlot] = []
    for fragment in sorted(fragments.values(), key=lambda item: item.first_frame):
        best: FighterSlot | None = None
        best_distance = 0.0
        for slot in slots:
            if slot.frames & fragment.frames:
                continue
            gap = fragment.first_frame - slot.last_frame
            if gap < 0 or gap > max_gap_frames:
                continue
            distance = _normalized_distance(slot, fragment)
            if distance > max_distance:
                continue
            if best is None or distance < best_distance:
                best, best_distance = slot, distance
        if best is None:
            slots.append(FighterSlot.from_fragment(fragment))
        else:
            best.absorb(fragment)
    return slots


def _select_anchor_slots(slots: list[FighterSlot]) -> list[FighterSlot]:
    """Choose two strong identities that are proven to be different people.

    Prefer a pair that appears simultaneously. Two slots sharing a frame cannot
    describe the same fighter, which is stronger evidence than screen position.
    """
    overlapping_pairs = [
        (first, second)
        for index, first in enumerate(slots)
        for second in slots[index + 1 :]
        if first.frames & second.frames
    ]
    if overlapping_pairs:
        first, second = max(
            overlapping_pairs,
            key=lambda pair: (
                len(pair[0].frames) + len(pair[1].frames),
                len(pair[0].frames & pair[1].frames),
            ),
        )
        return [first, second]
    return sorted(slots, key=lambda slot: -len(slot.frames))[:2]


def resolve_fighter_slots(
    slots: list[FighterSlot],
) -> tuple[list[FighterSlot], list[FighterSlot]]:
    """Resolve all compatible fragment chains into two fighter identities.

    The previous implementation kept only the two longest chains. A camera cut
    could therefore split one real fighter into a third chain and make that
    fighter disappear for an entire shot. Here, temporal overlap acts as a hard
    identity constraint: a leftover chain overlapping exactly one fighter must
    belong to the other one.

    Chains overlapping both identities are left unassigned because they are
    likely a referee/false positive. Chains overlapping neither identity remain
    unassigned until appearance-based re-identification is available; guessing
    from horizontal position alone can swap fighters after a camera cut.
    """
    if len(slots) <= 2:
        fighters = list(slots)
        fighters.sort(key=lambda slot: slot.mean_cx)
        return fighters, []

    fighters = _select_anchor_slots(slots)
    anchor_ids = {id(slot) for slot in fighters}
    remaining = sorted(
        (slot for slot in slots if id(slot) not in anchor_ids),
        key=lambda slot: -len(slot.frames),
    )
    unassigned: list[FighterSlot] = []

    for slot in remaining:
        overlaps_first = bool(slot.frames & fighters[0].frames)
        overlaps_second = bool(slot.frames & fighters[1].frames)
        if overlaps_first and not overlaps_second:
            fighters[1].absorb_slot(slot)
        elif overlaps_second and not overlaps_first:
            fighters[0].absorb_slot(slot)
        else:
            unassigned.append(slot)

    fighters.sort(key=lambda slot: slot.mean_cx)
    return fighters, unassigned


def select_fighter_slots(slots: list[FighterSlot]) -> list[FighterSlot]:
    """Compatibility wrapper returning the two resolved fighter identities."""
    fighters, _ = resolve_fighter_slots(slots)
    return fighters


def build_tracking_records(
    per_frame: list[dict[int, TrackObservation]],
    slots: list[FighterSlot],
    fps: float,
) -> list[TrackingRecord]:
    """Emit one record per frame and fighter, including frames with no box.

    The stable identity lives in `fighter_id`, while `track_id` keeps the
    ByteTrack fragment that supplied the box so the merge stays auditable.
    """
    records: list[TrackingRecord] = []
    for frame_index, frame_map in enumerate(per_frame):
        timestamp = frame_index / fps
        for label, slot in zip(FIGHTER_LABELS, slots):
            visible_id = next(
                (track_id for track_id in slot.track_ids if track_id in frame_map), None
            )
            if visible_id is None:
                records.append(
                    TrackingRecord(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        fighter_id=label,
                        track_id=MISSING_TRACK_ID,
                        bbox_xyxy=None,
                        confidence=None,
                        visible=False,
                    )
                )
                continue
            observation = frame_map[visible_id]
            records.append(
                TrackingRecord(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    fighter_id=label,
                    track_id=visible_id,
                    bbox_xyxy=tuple(
                        float(value) for value in observation.bbox_xyxy.tolist()
                    ),
                    confidence=float(observation.confidence),
                    visible=True,
                )
            )
    return records


def extract_merged_fighter_tracking(
    video_path: str | Path,
    *,
    confidence: float = 0.5,
    max_frames: int | None = None,
    min_fragment_frames: int = MIN_FRAGMENT_FRAMES,
    max_gap_frames: int = MAX_MERGE_GAP_FRAMES,
    max_distance: float = MAX_MERGE_DISTANCE,
) -> MergedTracking:
    """Run detection and ByteTrack, then rebuild two fighters from the fragments.

    Unlike `extract_fighter_tracking`, this never asks the detector to produce
    exactly two tracks: fragments left by camera cuts and occlusions are chained
    back together, so no `min_track_frames` tuning is needed per video.
    """
    from ufc_tracker.detection.personDetection import track_video
    from ufc_tracker.tracking.export import _resolve_video_path

    path = _resolve_video_path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"Video has invalid FPS: {path}")

    per_frame, stats, frame_count = track_video(
        path, max_frames=max_frames, conf=confidence
    )
    fragments = describe_fragments(per_frame, stats)
    candidates = select_fragment_candidates(
        fragments, min_frames=min_fragment_frames
    )
    slots = merge_fragments(
        candidates, max_gap_frames=max_gap_frames, max_distance=max_distance
    )
    fighters, unassigned = resolve_fighter_slots(slots)
    if len(fighters) < 2:
        raise RuntimeError(
            "Fragment merging must yield at least two fighter slots; produced "
            f"{len(fighters)} from {len(candidates)} candidate fragments. "
            "Loosen max_gap_frames/max_distance or lower the appearance gates."
        )

    records = build_tracking_records(per_frame, fighters, fps)
    return MergedTracking(
        records=records,
        fps=fps,
        frame_count=frame_count,
        slots=fighters,
        confidence=confidence,
        min_fragment_frames=min_fragment_frames,
        max_gap_frames=max_gap_frames,
        max_distance=max_distance,
        candidate_track_ids=tuple(sorted(candidates)),
        unassigned_track_ids=tuple(
            track_id for slot in unassigned for track_id in slot.track_ids
        ),
    )
