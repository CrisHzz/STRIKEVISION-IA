"""Masked multilabel metrics that never score unknown action labels as negatives."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES, PRIMARY_ACTION_CLASSES


def optimize_thresholds(
    targets: np.ndarray, probabilities: np.ndarray, masks: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Tune each class independently using only its known validation targets."""
    thresholds = np.arange(0.05, 0.951, 0.01) if thresholds is None else thresholds
    output = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for index in range(targets.shape[1]):
        usable = masks[:, index].astype(bool)
        if usable.sum() == 0 or np.unique(targets[usable, index]).size < 2:
            continue
        scores = []
        for threshold in thresholds:
            _, _, f1, _ = precision_recall_fscore_support(
                targets[usable, index], probabilities[usable, index] >= threshold,
                average="binary", zero_division=0,
            )
            scores.append(float(f1))
        output[index] = float(thresholds[int(np.argmax(scores))])
    return output


def multilabel_metrics(
    targets: np.ndarray, probabilities: np.ndarray, masks: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report each class separately, then macro-average only available classes."""
    targets, probabilities, masks = (np.asarray(value) for value in (targets, probabilities, masks))
    if targets.shape != probabilities.shape or targets.shape != masks.shape:
        raise ValueError("Targets, probabilities and masks must share shape")
    if targets.ndim != 2 or targets.shape[1] != len(ACTION_CLASSES):
        raise ValueError("Expected [samples, action_classes] multilabel arrays")
    thresholds = np.full(len(ACTION_CLASSES), 0.5) if thresholds is None else np.asarray(thresholds)
    if thresholds.shape != (len(ACTION_CLASSES),):
        raise ValueError("Expected one threshold for every action class")
    classes = {}
    for index, label in enumerate(ACTION_CLASSES):
        usable = masks[:, index].astype(bool)
        if not usable.any():
            classes[label] = {"support": 0, "precision": None, "recall": None, "f1": None, "average_precision": None}
            continue
        truth = targets[usable, index].astype(int)
        predicted = probabilities[usable, index] >= thresholds[index]
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
        ap = None if np.unique(truth).size < 2 else float(average_precision_score(truth, probabilities[usable, index]))
        classes[label] = {
            "support": int(truth.sum()), "known_examples": int(usable.sum()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1), "average_precision": ap,
        }
    def macro(labels: tuple[str, ...], metric: str) -> float | None:
        values = [classes[label][metric] for label in labels if classes[label][metric] is not None]
        return None if not values else float(np.mean(values))
    return {
        "thresholds": {label: float(thresholds[index]) for index, label in enumerate(ACTION_CLASSES)},
        "classes": classes,
        "primary_macro_f1": macro(PRIMARY_ACTION_CLASSES, "f1"),
        "all_class_macro_f1": macro(ACTION_CLASSES, "f1"),
        "primary_macro_average_precision": macro(PRIMARY_ACTION_CLASSES, "average_precision"),
    }
