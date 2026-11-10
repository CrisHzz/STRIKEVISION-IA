"""Compose a v4 checkpoint with punch_head supplied by a second trained TCN."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ufc_tracker.action_recognition.comparison import _prediction_bundle
from ufc_tracker.action_recognition.contracts import ACTION_CLASSES
from ufc_tracker.action_recognition.dataset import sha256
from ufc_tracker.action_recognition.inference import ActionInferenceEngine
from ufc_tracker.action_recognition.metrics import multilabel_metrics
from ufc_tracker.action_recognition.models import build_torch_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    base_path, donor_path = (directory / "candidate.pt" for directory in (args.reference, args.donor))
    base, donor = (torch.load(path, map_location="cpu", weights_only=True) for path in (base_path, donor_path))
    if any(checkpoint["name"] != "tcn" or tuple(checkpoint["classes"]) != ACTION_CLASSES
           for checkpoint in (base, donor)) or any(
               base[key] != donor[key] for key in ("schema", "feature_dim", "feature_spec")
           ):
        raise ValueError("Expected two compatible v4 TCN checkpoints")
    ids, base_scores, targets, masks, base_thresholds = _prediction_bundle(args.reference)
    donor_ids, donor_scores, donor_targets, donor_masks, donor_thresholds = _prediction_bundle(args.donor)
    if (ids != donor_ids or not np.array_equal(targets, donor_targets) or not np.array_equal(masks, donor_masks)):
        raise ValueError("Checkpoints were evaluated on different validation samples")

    head = ACTION_CLASSES.index("punch_head")
    other = [index for index in range(len(ACTION_CLASSES)) if index != head]
    thresholds = base_thresholds.copy()
    thresholds[head] = donor_thresholds[head]
    parameters = {"reference_parameters": base["parameters"], "donor_parameters": donor["parameters"]}
    state = {f"{prefix}.{name}": value for prefix, checkpoint in (("reference", base), ("donor", donor))
             for name, value in checkpoint["state_dict"].items()}
    payload = {**base, "name": "tcn_punch_head_hybrid", "parameters": parameters,
               "state_dict": state, "thresholds": thresholds.tolist()}

    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "candidate.pt"
    torch.save(payload, checkpoint_path)
    _, loaded, _, loaded_thresholds = ActionInferenceEngine(Path.cwd())._load_candidate(args.output)
    model = build_torch_model(loaded["name"], loaded["feature_dim"], len(ACTION_CLASSES), **loaded["parameters"])
    model.load_state_dict(loaded["state_dict"], strict=True)
    model.eval()
    with np.load(args.reference / "pose_features.npz", allow_pickle=False) as arrays:
        features = torch.from_numpy(arrays["features"][-len(ids):])
        frames = torch.from_numpy(arrays["frame_mask"][-len(ids):, :, 0])
    with torch.inference_mode():
        scores = model(features, frames).sigmoid().numpy()
    np.testing.assert_array_equal(scores[:, other], base_scores[:, other])
    np.testing.assert_array_equal(scores[:, head], donor_scores[:, head])
    np.testing.assert_array_equal(loaded_thresholds[other], base_thresholds[other])
    np.testing.assert_equal(loaded_thresholds[head], donor_thresholds[head])

    report = {
        "reference_sha256": sha256(base_path), "donor_sha256": sha256(donor_path),
        "validation_samples": len(ids),
        "reference_metrics": multilabel_metrics(targets, base_scores, masks, base_thresholds),
        "donor_metrics": multilabel_metrics(targets, donor_scores, masks, donor_thresholds),
        "candidate_metrics": multilabel_metrics(targets, scores, masks, loaded_thresholds),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["candidate_metrics"], indent=2))


if __name__ == "__main__":
    main()
