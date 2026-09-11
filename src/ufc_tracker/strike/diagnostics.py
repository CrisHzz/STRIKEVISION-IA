"""Validation-only temporal and error diagnostics for a fixed candidate."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ufc_tracker.strike.metrics import choose_threshold, metrics


def _centered_mean(values: np.ndarray, width: int) -> np.ndarray:
    radius = width // 2
    return np.array([
        values[max(0, index - radius):min(len(values), index + radius + 1)].mean()
        for index in range(len(values))
    ])


def analyze_validation(rows: list[dict], allowed_fights: set[str]) -> dict:
    if not rows or {row["fight_id"] for row in rows} != allowed_fights:
        raise ValueError("Expected exactly the configured validation fights")
    thresholds = np.arange(0.05, 0.951, 0.01)
    by_video = defaultdict(list)
    for row in rows:
        if row["fight_id"] not in allowed_fights:
            raise ValueError("Test/train prediction found in validation diagnostics")
        by_video[row["video_id"]].append(row)

    ordered, smoothed = [], []
    transition_distances = {}
    for video, group in sorted(by_video.items()):
        group.sort(key=lambda row: (row["start_frame"], row["end_frame"]))
        probabilities = np.array([row["p_strike"] for row in group], dtype=float)
        labels = np.array([int(row["label"] == "strike") for row in group])
        changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        for index, row in enumerate(group):
            transition_distances[row["window_id"]] = (
                int(np.min(np.abs(changes - index))) if len(changes) else None
            )
        ordered.extend(group)
        smoothed.extend(_centered_mean(probabilities, 3).tolist())

    labels = np.array([int(row["label"] == "strike") for row in ordered])
    raw = np.array([row["p_strike"] for row in ordered], dtype=float)
    fixed_threshold = float(ordered[0]["threshold"])
    if any(float(row["threshold"]) != fixed_threshold for row in ordered):
        raise ValueError("Mixed fixed thresholds")
    smooth = np.asarray(smoothed)
    smooth_threshold = choose_threshold(labels, smooth, thresholds)
    predicted = raw >= fixed_threshold
    error_type = np.where((labels == 0) & predicted, "false_positive",
                          np.where((labels == 1) & ~predicted, "false_negative", "correct"))

    def describe(kind: str) -> dict:
        indices = np.flatnonzero(error_type == kind)
        probabilities = raw[indices]
        distances = [transition_distances[ordered[index]["window_id"]] for index in indices]
        distances = [value for value in distances if value is not None]
        most_confident = sorted(
            (ordered[index] for index in indices),
            key=lambda row: row["p_strike"],
            reverse=(kind == "false_positive"),
        )[:20]
        return {
            "count": int(len(indices)),
            "probability_quantiles": {
                str(q): float(np.quantile(probabilities, q)) for q in (0, .25, .5, .75, 1)
            },
            "within_windows_of_label_transition": {
                str(radius): int(sum(value <= radius for value in distances))
                for radius in (1, 2, 3)
            },
            "most_confident_window_ids": [row["window_id"] for row in most_confident],
        }

    fights = {}
    for fight in sorted(allowed_fights):
        mask = np.array([row["fight_id"] == fight for row in ordered])
        fight_errors = error_type[mask]
        fights[fight] = {
            "windows": int(mask.sum()),
            "false_positives": int(np.sum(fight_errors == "false_positive")),
            "false_negatives": int(np.sum(fight_errors == "false_negative")),
        }
    return {
        "scope": "configured validation fights only; test predictions are forbidden",
        "raw_fixed": metrics(labels, raw, fixed_threshold),
        "centered_mean_3_selected_on_validation": metrics(labels, smooth, smooth_threshold),
        "smoothing_notes": [
            "Centered smoothing uses one future neighboring window and is suitable for offline review.",
            "It changes window-level display stability, not the definition or count of strikes.",
        ],
        "errors": {
            "false_positive": describe("false_positive"),
            "false_negative": describe("false_negative"),
            "by_fight": fights,
        },
        "test_accessed": False,
    }
