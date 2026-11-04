"""Evaluate a hierarchical candidate after applying its action gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ufc_tracker.action_recognition.metrics import multilabel_metrics, optimize_thresholds
from ufc_tracker.action_recognition.models import build_torch_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--reference-candidate", type=Path)
    args = parser.parse_args()
    run_dir = args.candidate_dir.resolve()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    candidate = torch.load(run_dir / "candidate.pt", map_location="cpu", weights_only=False)
    if candidate.get("schema") != "action_temporal_candidate_v2":
        raise ValueError("Expected a hierarchical V2 candidate")
    with np.load(run_dir / "pose_features.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    split = int(report["train_samples"])
    model = build_torch_model(
        candidate["name"], int(candidate["feature_dim"]), 7, **candidate["parameters"],
    )
    model.load_state_dict(candidate["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        raw = torch.sigmoid(model(
            torch.from_numpy(arrays["features"][split:]),
            torch.from_numpy(arrays["frame_mask"][split:, :, 0]),
        )).numpy()
    action_probabilities, technique_probabilities = raw[:, 0], raw[:, 1:]
    targets, masks = arrays["target"][split:], arrays["target_mask"][split:]
    action_targets = targets.max(axis=1).astype(bool)
    rows = []
    for threshold in np.arange(0.35, 0.951, 0.01):
        action_predictions = action_probabilities >= threshold
        gated = technique_probabilities.copy()
        gated[~action_predictions] = 0.0
        metrics = multilabel_metrics(targets, gated, masks, np.asarray(candidate["thresholds"]))
        true_positive = float(np.logical_and(action_targets, action_predictions).sum())
        false_positive = float(np.logical_and(~action_targets, action_predictions).sum())
        false_negative = float(np.logical_and(action_targets, ~action_predictions).sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        action_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({
            "action_threshold": round(float(threshold), 2),
            "gated_primary_macro_f1": metrics["primary_macro_f1"],
            "gated_all_class_macro_f1": metrics["all_class_macro_f1"],
            "action_precision": precision,
            "action_recall": recall,
            "action_f1": action_f1,
        })
    rows.sort(key=lambda row: (-(row["gated_primary_macro_f1"] or -1), -row["action_precision"], row["action_threshold"]))
    result = {"candidate": str(run_dir), "saved_action_threshold": candidate["action_threshold"], "best": rows[0], "top": rows[:10]}
    if args.reference_candidate:
        reference_dir = args.reference_candidate.resolve()
        reference = torch.load(reference_dir / "candidate.pt", map_location="cpu", weights_only=False)
        if reference.get("schema") != "action_temporal_candidate_v1":
            raise ValueError("Reference must be a temporal V1 candidate")
        if reference["feature_spec"] != candidate["feature_spec"]:
            raise ValueError("Reference and V2 candidate use different feature specifications")
        reference_model = build_torch_model(
            reference["name"], int(reference["feature_dim"]), 6, **reference["parameters"],
        )
        reference_model.load_state_dict(reference["state_dict"], strict=True)
        reference_model.eval()
        with torch.no_grad():
            reference_probabilities = torch.sigmoid(reference_model(
                torch.from_numpy(arrays["features"][split:]),
                torch.from_numpy(arrays["frame_mask"][split:, :, 0]),
            )).numpy()
        recalibrated_thresholds = optimize_thresholds(targets, reference_probabilities, masks)
        result["reference"] = {
            "candidate": str(reference_dir),
            "saved_threshold_metrics_on_v2_labels": multilabel_metrics(
                targets, reference_probabilities, masks, np.asarray(reference["thresholds"]),
            ),
            "recalibrated_thresholds": recalibrated_thresholds.tolist(),
            "recalibrated_metrics_on_v2_labels": multilabel_metrics(
                targets, reference_probabilities, masks, recalibrated_thresholds,
            ),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
