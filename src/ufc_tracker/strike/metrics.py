"""Window metrics; average precision is the reported PR-AUC convention."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)


def choose_threshold(labels, probabilities, thresholds):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    predictions = p[:, None] >= thresholds[None, :]
    tp = ((y[:, None] == 1) & predictions).sum(axis=0)
    fp = ((y[:, None] == 0) & predictions).sum(axis=0)
    fn = ((y[:, None] == 1) & ~predictions).sum(axis=0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    # Prefer a threshold near 0.5 on tied F1, then the smaller threshold.
    index = min(range(len(thresholds)), key=lambda i: (-f1[i], abs(thresholds[i] - .5), thresholds[i]))
    return float(thresholds[index])


def metrics(labels, probabilities, threshold):
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid labels/probabilities")
    if not set(y).issubset({0, 1}) or not 0 <= threshold <= 1:
        raise ValueError("Binary labels and threshold required")
    predicted = (p >= threshold).astype(int)
    both = len(set(y)) == 2
    return {"windows": len(y), "strike_support": int(y.sum()), "threshold": threshold,
            "precision_strike": float(precision_score(y, predicted, zero_division=0)),
            "recall_strike": float(recall_score(y, predicted, zero_division=0)),
            "f1_strike": float(f1_score(y, predicted, zero_division=0)),
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
            "pr_auc_average_precision": float(average_precision_score(y, p)) if both else None,
            "accuracy": float(accuracy_score(y, predicted)),
            "confusion_matrix_true_rows_pred_columns": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def evaluation(rows, probabilities, threshold):
    y = np.array([int(r["label"] == "strike") for r in rows])
    p = np.asarray(probabilities)
    result = {"model": metrics(y, p, threshold),
              "always_no_strike": metrics(y, np.zeros(len(y)), .5),
              "always_strike": metrics(y, np.ones(len(y)), .5),
              "scope": "Overlapping windows, not independent strikes; no calibrated confidence claim",
              "groups": {}}
    groupers = {"fight": lambda r: r["fight_id"], "category": lambda r: r["category"],
                "fps": lambda r: str(round(r["fps"], 3)),
                "duration": lambda r: str(round(r["duration_seconds"], 4)),
                "pose_usable_diagnostic_only": lambda r: str(r.get("pose_usable", True))}
    for name, key in groupers.items():
        values = [key(r) for r in rows]
        result["groups"][name] = {}
        for value in sorted(set(values)):
            mask = np.array([v == value for v in values])
            result["groups"][name][value] = {
                "model": metrics(y[mask], p[mask], threshold),
                "always_no_strike": metrics(y[mask], np.zeros(mask.sum()), .5)}
    return result
