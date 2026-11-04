"""Compare saved action candidates on exactly the same validation samples."""

from __future__ import annotations

import json
import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES
from ufc_tracker.action_recognition.metrics import multilabel_metrics
from ufc_tracker.action_recognition.models import build_torch_model


def _report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def _metadata_ids(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8"))["sample_ids"])


def _prediction_bundle(run_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ordered validation IDs, probabilities, targets, masks and thresholds."""
    report = _report(run_dir)
    validation_ids = list(report["sample_ids"]["validation"])
    pickle_path, torch_path = run_dir / "candidate.pkl", run_dir / "candidate.pt"
    if pickle_path.is_file():
        with pickle_path.open("rb") as stream:
            candidate = pickle.load(stream)
        with np.load(run_dir / "x3d_embeddings.npz", allow_pickle=False) as arrays:
            embeddings, target, mask = arrays["embeddings"], arrays["target"], arrays["target_mask"]
        ids = _metadata_ids(run_dir / "x3d_embeddings.json")
        index = {sample_id: position for position, sample_id in enumerate(ids)}
        positions = [index[sample_id] for sample_id in validation_ids if sample_id in index]
        selected_ids = [sample_id for sample_id in validation_ids if sample_id in index]
        probabilities = np.column_stack([model.predict_proba(embeddings[positions])[:, 1] for model in candidate["models"]])
        return selected_ids, probabilities, target[positions], mask[positions], np.asarray(candidate["thresholds"])
    if torch_path.is_file():
        candidate = torch.load(torch_path, map_location="cpu", weights_only=False)
        with np.load(run_dir / "pose_features.npz", allow_pickle=False) as arrays:
            features, frame_mask = arrays["features"], arrays["frame_mask"]
            target, mask = arrays["target"], arrays["target_mask"]
        ids = _metadata_ids(run_dir / "pose_features.json")
        expected_ids = list(report["sample_ids"]["train"]) + validation_ids
        if ids != expected_ids:
            raise ValueError(
                f"{run_dir.name} uses a legacy reordered pose cache that mixed train/validation; retrain it"
            )
        index = {sample_id: position for position, sample_id in enumerate(ids)}
        positions = [index[sample_id] for sample_id in validation_ids]
        model = build_torch_model(
            candidate["name"], int(candidate["feature_dim"]), len(ACTION_CLASSES), **candidate["parameters"],
        )
        model.load_state_dict(candidate["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(
                model(torch.from_numpy(features[positions]), torch.from_numpy(frame_mask[positions, :, 0]))
            ).numpy()
        return validation_ids, probabilities, target[positions], mask[positions], np.asarray(candidate["thresholds"])
    raise FileNotFoundError(f"No candidate artifact in {run_dir}")


def compare_on_common_validation_samples(candidate_dir: Path, reference_dir: Path) -> dict[str, Any]:
    """Score two candidates with saved thresholds on their common validation IDs."""
    candidate = _prediction_bundle(Path(candidate_dir))
    reference = _prediction_bundle(Path(reference_dir))
    candidate_index = {sample_id: position for position, sample_id in enumerate(candidate[0])}
    reference_index = {sample_id: position for position, sample_id in enumerate(reference[0])}
    common_ids = [sample_id for sample_id in candidate[0] if sample_id in reference_index]
    if not common_ids:
        raise ValueError("Candidates do not share validation samples")
    candidate_positions = [candidate_index[sample_id] for sample_id in common_ids]
    reference_positions = [reference_index[sample_id] for sample_id in common_ids]
    candidate_target, candidate_mask = candidate[2][candidate_positions], candidate[3][candidate_positions]
    reference_target, reference_mask = reference[2][reference_positions], reference[3][reference_positions]
    if not np.array_equal(candidate_target, reference_target) or not np.array_equal(candidate_mask, reference_mask):
        raise ValueError("Candidates disagree on targets or supervision masks for common samples")
    return {
        "schema_version": "action_common_subset_comparison_v1",
        "candidate": Path(candidate_dir).name,
        "reference": Path(reference_dir).name,
        "candidate_validation_samples": len(candidate[0]),
        "reference_validation_samples": len(reference[0]),
        "common_validation_samples": len(common_ids),
        "common_sample_ids_sha256": sha256("\n".join(common_ids).encode("utf-8")).hexdigest(),
        "candidate_metrics": multilabel_metrics(
            candidate_target, candidate[1][candidate_positions], candidate_mask, candidate[4],
        ),
        "reference_metrics": multilabel_metrics(
            reference_target, reference[1][reference_positions], reference_mask, reference[4],
        ),
    }
