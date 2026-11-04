"""Stable contracts shared by action-recognition training and inference."""

from __future__ import annotations

from typing import Iterable

ACTION_SCHEMA_VERSION = "strikevision_action_recognition_v1"
ACTION_CLASSES = (
    "punch_head",
    "punch_body",
    "kick_leg",
    "kick_body",
    "kick_head",
    "other_strike",
)
PRIMARY_ACTION_CLASSES = ACTION_CLASSES[:-1]
AMBIGUOUS_LABEL = "ambiguous"
FIGHTER_IDS = ("1", "2")
LEGACY_FIGHTER_IDS = {"fighter_left": "1", "fighter_right": "2"}


def canonical_fighter_id(value: object) -> str | None:
    """Map persisted tracking labels to Gradio's red/purple fighter IDs."""
    fighter_id = str(value)
    if fighter_id in FIGHTER_IDS:
        return fighter_id
    return LEGACY_FIGHTER_IDS.get(fighter_id)


def intervals_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int,
) -> bool:
    """Inclusive frame intervals overlap when they share at least one frame."""
    return max(int(first_start), int(second_start)) <= min(int(first_end), int(second_end))


def multi_hot(labels: Iterable[str]) -> list[int]:
    """Encode known positive labels while retaining class order in artifacts."""
    labels = set(labels)
    unknown = labels - set(ACTION_CLASSES)
    if unknown:
        raise ValueError(f"Unknown action labels: {sorted(unknown)}")
    return [int(label in labels) for label in ACTION_CLASSES]
