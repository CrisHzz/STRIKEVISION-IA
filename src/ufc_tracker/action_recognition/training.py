"""Manual training entry points for the action-recognition tournament.

Nothing in this module runs at import time.  Every training invocation receives
an explicit output directory and records its fold, sampled IDs and thresholds.
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES, ACTION_SCHEMA_VERSION
from ufc_tracker.action_recognition.features import PoseFeatureSpec, build_pose_feature_cache, samples_sha256
from ufc_tracker.action_recognition.metrics import multilabel_metrics, optimize_thresholds
from ufc_tracker.action_recognition.models import build_torch_model, masked_bce_with_logits
from ufc_tracker.action_recognition.rgb import build_x3d_embeddings


def read_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_pose_samples(
    samples: list[dict[str, Any]], *, split: str, negative_ratio: float, seed: int,
    validation_fold: int | None = None,
) -> list[dict[str, Any]]:
    """Keep all positives and deterministically downsample only confirmed negatives."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    filtered = [sample for sample in samples if sample.get("pose_usable", True)]
    if split == "test":
        filtered = [sample for sample in filtered if sample["split"] == "test"]
    elif validation_fold is None:
        raise ValueError("Development selections require validation_fold")
    elif split == "validation":
        filtered = [sample for sample in filtered if sample["split"] == "development" and sample["development_fold"] == validation_fold]
    else:
        filtered = [
            sample for sample in filtered
            if sample["split"] == "development"
            and (sample["development_fold"] is None or sample["development_fold"] != validation_fold)
        ]
    positives = [sample for sample in filtered if sample["sample_type"] == "positive_action"]
    negatives_by_fight: dict[str, list[dict[str, Any]]] = {}
    positives_by_fight: dict[str, int] = {}
    for sample in filtered:
        if sample["sample_type"] == "positive_action":
            positives_by_fight[sample["fight_id"]] = positives_by_fight.get(sample["fight_id"], 0) + 1
        elif sample["sample_type"] == "confirmed_no_strike":
            negatives_by_fight.setdefault(sample["fight_id"], []).append(sample)
    rng = random.Random(seed)
    selected_negatives = []
    for fight_id, candidates in sorted(negatives_by_fight.items()):
        count = min(len(candidates), int(np.ceil(positives_by_fight.get(fight_id, 0) * negative_ratio)))
        selected_negatives.extend(rng.sample(candidates, count))
    output = sorted(positives + selected_negatives, key=lambda sample: sample["sample_id"])
    if not output or not positives:
        raise ValueError(f"No positive pose samples selected for {split}")
    return output


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _pooled_batch(features: np.ndarray, attacker_mask: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    output = []
    for values, mask, score in zip(features, attacker_mask, coverage, strict=True):
        selected = values[mask.astype(bool)]
        if len(selected) == 0:
            output.append(np.zeros(values.shape[1] * 3 + 1, dtype=np.float32))
        else:
            output.append(np.concatenate((selected.mean(0), selected.std(0), selected.max(0), [score])))
    return np.asarray(output, dtype=np.float32)


def _feature_cache(
    root: Path, samples: list[dict[str, Any]], cache_path: Path, spec: PoseFeatureSpec,
) -> dict[str, np.ndarray]:
    metadata_path = cache_path.with_suffix(".json")
    expected_hash = samples_sha256(samples)
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("samples_sha256") == expected_hash and metadata.get("spec") == asdict(spec):
            return _load_cache(cache_path)
    build_pose_feature_cache(root, samples, cache_path, spec)
    return _load_cache(cache_path)


def pose_training_loader(arrays: dict[str, np.ndarray], count: int, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays["features"][:count]), torch.from_numpy(arrays["frame_mask"][:count, :, 0]),
        torch.from_numpy(arrays["target"][:count]), torch.from_numpy(arrays["target_mask"][:count]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _fit_classical(
    name: str, values: np.ndarray, targets: np.ndarray, masks: np.ndarray, seed: int, *,
    logistic_c: float = 1.0, logistic_class_weight: str | None = "balanced",
) -> list[Any]:
    models = []
    for class_index, label in enumerate(ACTION_CLASSES):
        known = masks[:, class_index].astype(bool)
        truth = targets[known, class_index].astype(int)
        if known.sum() == 0 or np.unique(truth).size < 2:
            raise ValueError(f"{name}: class {label} lacks both positive and negative training labels")
        if name == "logistic_regression":
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=logistic_c, class_weight=logistic_class_weight, max_iter=2000, random_state=seed,
            ))
        elif name == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed,
            )
        else:
            raise ValueError(f"Unsupported classical baseline: {name}")
        model.fit(values[known], truth)
        models.append(model)
    return models


def _classical_probabilities(models: list[Any], values: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict_proba(values)[:, 1] for model in models])


def run_classical_baseline(
    root: Path, samples: list[dict[str, Any]], output_dir: Path, *, name: str,
    validation_fold: int, negative_ratio: float, seed: int, spec: PoseFeatureSpec,
) -> dict[str, Any]:
    """Train one auditable baseline on a held-out development fight pair."""
    root, output_dir = Path(root), Path(output_dir)
    train = select_pose_samples(samples, split="train", validation_fold=validation_fold, negative_ratio=negative_ratio, seed=seed)
    validation = select_pose_samples(samples, split="validation", validation_fold=validation_fold, negative_ratio=negative_ratio, seed=seed + 1)
    combined = train + validation
    arrays = _feature_cache(root, combined, output_dir / "pose_features.npz", spec)
    values = _pooled_batch(arrays["features"], arrays["frame_mask"][:, :, 0], arrays["pose_coverage"])
    split = len(train)
    models = _fit_classical(name, values[:split], arrays["target"][:split], arrays["target_mask"][:split], seed)
    probabilities = _classical_probabilities(models, values[split:])
    thresholds = optimize_thresholds(arrays["target"][split:], probabilities, arrays["target_mask"][split:])
    metrics = multilabel_metrics(arrays["target"][split:], probabilities, arrays["target_mask"][split:], thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate.pkl").open("wb") as stream:
        pickle.dump({"schema": "action_classical_candidate_v1", "name": name, "classes": ACTION_CLASSES,
                     "models": models, "thresholds": thresholds.tolist(), "feature_spec": asdict(spec)}, stream)
    report = {"schema_version": ACTION_SCHEMA_VERSION, "family": name, "validation_fold": validation_fold,
              "train_samples": len(train), "validation_samples": len(validation), "metrics": metrics,
              "sample_ids": {"train": [sample["sample_id"] for sample in train], "validation": [sample["sample_id"] for sample in validation]}}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_x3d_linear_probe(
    root: Path, samples: list[dict[str, Any]], output_dir: Path, *, validation_fold: int,
    negative_ratio: float, seed: int, spec: PoseFeatureSpec, logistic_c: float = 1.0,
    logistic_class_weight: str | None = "balanced", embeddings_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit only six classical heads over frozen X3D-S attacker-crop embeddings."""
    root, output_dir = Path(root), Path(output_dir)
    train = select_pose_samples(samples, split="train", validation_fold=validation_fold, negative_ratio=negative_ratio, seed=seed)
    validation = select_pose_samples(samples, split="validation", validation_fold=validation_fold, negative_ratio=negative_ratio, seed=seed + 1)
    combined = train + validation
    cache_dir = Path(embeddings_cache_dir) if embeddings_cache_dir is not None else output_dir
    embeddings_path = cache_dir / "x3d_embeddings.npz"
    metadata_path = cache_dir / "x3d_embeddings.json"
    if not embeddings_path.is_file() or not metadata_path.is_file():
        metadata = build_x3d_embeddings(root, combined, embeddings_path, spec)
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("feature_spec") != asdict(spec):
            raise ValueError(f"X3D embedding cache uses a different feature specification: {cache_dir}")
    with np.load(embeddings_path, allow_pickle=False) as arrays:
        values, targets, masks = arrays["embeddings"], arrays["target"], arrays["target_mask"]
    ids = metadata["sample_ids"]; index = {sample_id: position for position, sample_id in enumerate(ids)}
    train_indices = [index[sample["sample_id"]] for sample in train if sample["sample_id"] in index]
    validation_indices = [index[sample["sample_id"]] for sample in validation if sample["sample_id"] in index]
    if not train_indices or not validation_indices:
        raise ValueError("X3D cache does not contain both training and validation samples")
    models = _fit_classical(
        "logistic_regression", values[train_indices], targets[train_indices], masks[train_indices], seed,
        logistic_c=logistic_c, logistic_class_weight=logistic_class_weight,
    )
    probabilities = _classical_probabilities(models, values[validation_indices])
    thresholds = optimize_thresholds(targets[validation_indices], probabilities, masks[validation_indices])
    metrics = multilabel_metrics(targets[validation_indices], probabilities, masks[validation_indices], thresholds)
    with (output_dir / "candidate.pkl").open("wb") as stream:
        pickle.dump({"schema": "action_x3d_linear_probe_v1", "classes": ACTION_CLASSES, "models": models,
                     "thresholds": thresholds.tolist(), "feature_spec": asdict(spec),
                     "parameters": {"logistic_c": logistic_c, "logistic_class_weight": logistic_class_weight}}, stream)
    report = {"schema_version": ACTION_SCHEMA_VERSION, "family": "x3d_linear_probe", "validation_fold": validation_fold,
              "train_samples": len(train_indices), "validation_samples": len(validation_indices),
              "rgb_crop_excluded": len(combined) - len(ids), "metrics": metrics, "sample_ids": {"train": [ids[i] for i in train_indices], "validation": [ids[i] for i in validation_indices]}}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _pos_weight(targets: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    weights = []
    for index in range(targets.shape[1]):
        known = masks[:, index].astype(bool)
        positives = float(targets[known, index].sum())
        negatives = float(known.sum() - positives)
        weights.append(min(20.0, negatives / max(positives, 1.0)))
    return torch.tensor(weights, dtype=torch.float32)


def _binary_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    """Tune an action/no-action gate with deterministic threshold tie-breaking."""
    targets = np.asarray(targets, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    best = (-1.0, -1.0, -1.0, 0.5)
    for threshold in np.arange(0.05, 0.951, 0.01):
        predicted = probabilities >= threshold
        tp = float(np.logical_and(targets, predicted).sum())
        fp = float(np.logical_and(~targets, predicted).sum())
        fn = float(np.logical_and(targets, ~predicted).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        key = (f1, -abs(float(threshold) - 0.5), -float(threshold), float(threshold))
        if key > best:
            best = key
            selected = (float(threshold), {"precision": precision, "recall": recall, "f1": f1})
    return selected


def _evaluate_hierarchical(
    targets: np.ndarray, probabilities: np.ndarray, masks: np.ndarray,
    action_probabilities: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float], dict[str, Any], dict[str, Any]]:
    """Select the action gate using final gated technique performance."""
    thresholds = optimize_thresholds(targets, probabilities, masks)
    ungated_metrics = multilabel_metrics(targets, probabilities, masks, thresholds)
    action_targets = targets.max(axis=1).astype(bool)
    choices = []
    for action_threshold in np.arange(0.05, 0.951, 0.01):
        action_predictions = action_probabilities >= action_threshold
        gated = probabilities.copy()
        gated[~action_predictions] = 0.0
        metrics = multilabel_metrics(targets, gated, masks, thresholds)
        true_positive = float(np.logical_and(action_targets, action_predictions).sum())
        false_positive = float(np.logical_and(~action_targets, action_predictions).sum())
        false_negative = float(np.logical_and(action_targets, ~action_predictions).sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        action_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        score = float(metrics["primary_macro_f1"] or 0.0)
        choices.append((score, precision, -abs(float(action_threshold) - 0.5), float(action_threshold),
                        {"precision": precision, "recall": recall, "f1": action_f1}, metrics))
    best = max(choices, key=lambda row: row[:3])
    return thresholds, best[3], best[4], best[5], ungated_metrics


def _hierarchical_loss(
    logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor,
    technique_weights: torch.Tensor, action_pos_weight: torch.Tensor,
) -> torch.Tensor:
    action_targets = targets.amax(dim=1)
    action_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], action_targets, pos_weight=action_pos_weight,
    )
    technique_loss = masked_bce_with_logits(
        logits[:, 1:], targets, target_mask, technique_weights,
    )
    return action_loss + technique_loss


def run_temporal_model(
    root: Path, samples: list[dict[str, Any]], output_dir: Path, *, name: str,
    validation_fold: int, negative_ratio: float, seed: int, spec: PoseFeatureSpec,
    learning_rate: float = 1e-3, weight_decay: float = 1e-4, channels: int = 64,
    blocks: int = 3, dropout: float = 0.2, batch_size: int = 32, max_epochs: int = 50,
    patience: int = 6, epoch_callback: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    """Train a local temporal pose candidate. Invoked manually from the CLI."""
    torch.manual_seed(seed); np.random.seed(seed)
    root, output_dir = Path(root), Path(output_dir)
    train = select_pose_samples(samples, split="train", validation_fold=validation_fold, negative_ratio=negative_ratio, seed=seed)
    validation = select_pose_samples(samples, split="validation", validation_fold=validation_fold, negative_ratio=negative_ratio, seed=seed + 1)
    combined = train + validation
    arrays = _feature_cache(root, combined, output_dir / "pose_features.npz", spec)
    split = len(train)
    feature_dim = int(arrays["features"].shape[-1])
    hierarchical = name == "hierarchical_tcn"
    model = build_torch_model(
        name, feature_dim, len(ACTION_CLASSES) + int(hierarchical),
        channels=channels, blocks=blocks, dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = _pos_weight(arrays["target"][:split], arrays["target_mask"][:split])
    action_targets = arrays["target"][:split].max(axis=1)
    action_positives = float(action_targets.sum())
    action_pos_weight = torch.tensor(
        min(20.0, float(len(action_targets) - action_positives) / max(action_positives, 1.0)),
        dtype=torch.float32,
    )
    loader = pose_training_loader(arrays, split, batch_size, shuffle=True)
    best, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for features, frame_mask, targets, target_mask in loader:
            optimizer.zero_grad()
            logits = model(features, frame_mask)
            loss = (
                _hierarchical_loss(logits, targets, target_mask, weights, action_pos_weight)
                if hierarchical else masked_bce_with_logits(logits, targets, target_mask, weights)
            )
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(torch.from_numpy(arrays["features"][split:]), torch.from_numpy(arrays["frame_mask"][split:, :, 0]))
            raw_probabilities = torch.sigmoid(validation_logits).numpy()
            action_probabilities = raw_probabilities[:, 0] if hierarchical else None
            probabilities = raw_probabilities[:, 1:] if hierarchical else raw_probabilities
        if hierarchical:
            thresholds, action_threshold, action_metrics, metrics, ungated_metrics = _evaluate_hierarchical(
                arrays["target"][split:], probabilities, arrays["target_mask"][split:], action_probabilities,
            )
            technique_score = metrics["primary_macro_f1"] if metrics["primary_macro_f1"] is not None else -1.0
            score = technique_score
        else:
            thresholds = optimize_thresholds(arrays["target"][split:], probabilities, arrays["target_mask"][split:])
            metrics = multilabel_metrics(arrays["target"][split:], probabilities, arrays["target_mask"][split:], thresholds)
            technique_score = metrics["primary_macro_f1"] if metrics["primary_macro_f1"] is not None else -1.0
            action_threshold, action_metrics, ungated_metrics, score = None, None, None, technique_score
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "primary_macro_f1": technique_score,
                        "action_f1": None if action_metrics is None else action_metrics["f1"], "selection_score": score})
        if epoch_callback is not None:
            epoch_callback(epoch, score)
        if score > best:
            best, best_state, stale = score, {key: value.detach().clone() for key, value in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Temporal training did not produce a candidate")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        raw_probabilities = torch.sigmoid(model(torch.from_numpy(arrays["features"][split:]), torch.from_numpy(arrays["frame_mask"][split:, :, 0]))).numpy()
    action_probabilities = raw_probabilities[:, 0] if hierarchical else None
    probabilities = raw_probabilities[:, 1:] if hierarchical else raw_probabilities
    if hierarchical:
        thresholds, action_threshold, action_metrics, metrics, ungated_metrics = _evaluate_hierarchical(
            arrays["target"][split:], probabilities, arrays["target_mask"][split:], action_probabilities,
        )
        selection_score = float(metrics["primary_macro_f1"] or 0.0)
    else:
        thresholds = optimize_thresholds(arrays["target"][split:], probabilities, arrays["target_mask"][split:])
        metrics = multilabel_metrics(arrays["target"][split:], probabilities, arrays["target_mask"][split:], thresholds)
        action_threshold, action_metrics = None, None
        ungated_metrics = None
        selection_score = float(metrics["primary_macro_f1"] or 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = {"channels": channels, "blocks": blocks, "dropout": dropout}
    torch.save({"schema": "action_temporal_candidate_v2" if hierarchical else "action_temporal_candidate_v1", "name": name, "classes": ACTION_CLASSES,
                "feature_dim": feature_dim, "feature_spec": asdict(spec), "parameters": parameters,
                "thresholds": thresholds.tolist(), "action_threshold": action_threshold,
                "state_dict": model.state_dict()}, output_dir / "candidate.pt")
    report = {"schema_version": ACTION_SCHEMA_VERSION, "family": name, "validation_fold": validation_fold,
              "train_samples": len(train), "validation_samples": len(validation), "epochs_completed": len(history),
              "history": history, "metrics": metrics, "ungated_metrics": ungated_metrics,
              "action_metrics": action_metrics,
              "action_threshold": action_threshold, "selection_score": selection_score,
              "sample_ids": {"train": [sample["sample_id"] for sample in train], "validation": [sample["sample_id"] for sample in validation]}}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
