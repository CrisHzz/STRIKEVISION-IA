"""Optuna tuning for manual temporal-model experiments, with resumable SQLite studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ufc_tracker.action_recognition.features import PoseFeatureSpec
from ufc_tracker.action_recognition.training import run_temporal_model, run_x3d_linear_probe


def tune_temporal_model(
    root: Path, samples: list[dict[str, Any]], output_root: Path, *, name: str,
    validation_fold: int, negative_ratio: float, seed: int, spec: PoseFeatureSpec,
    trials: int, timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run Optuna only when explicitly invoked; no dependency is needed for import."""
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Instala requirements-action-recognition.txt para usar Optuna.") from error
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_root / 'optuna.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=f"{name}-fold{validation_fold}", storage=storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=4),
    )
    def objective(trial):
        parameters = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "channels": trial.suggest_categorical("channels", [32, 64, 96]),
            "blocks": trial.suggest_categorical("blocks", [2, 3, 4]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        run_dir = output_root / "trials" / f"trial-{trial.number:03d}"
        report = run_temporal_model(
            root, samples, run_dir, name=name, validation_fold=validation_fold,
            negative_ratio=negative_ratio, seed=seed + trial.number, spec=spec,
            max_epochs=50, patience=6,
            epoch_callback=lambda epoch, score: _prune_if_needed(trial, epoch, score, optuna),
            **parameters,
        )
        score = report.get("selection_score", report["metrics"]["primary_macro_f1"])
        if score is None:
            raise RuntimeError("Validation produced no primary macro F1")
        return score
    study.optimize(objective, n_trials=trials, timeout=timeout_seconds)
    result = {
        "schema_version": "action_tuning_v1",
        "family": name,
        "validation_fold": validation_fold,
        "study": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "storage": storage,
        "completed_trials": len(study.trials),
    }
    report_path = output_root / "tuning_report.json"
    temporary_path = report_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(report_path)
    return result


def _prune_if_needed(trial, epoch: int, score: float, optuna) -> None:
    trial.report(score, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()


def tune_x3d_linear_probe(
    root: Path, samples: list[dict[str, Any]], output_root: Path, *, validation_fold: int,
    negative_ratio: float, seed: int, spec: PoseFeatureSpec, trials: int,
) -> dict[str, Any]:
    """Tune only X3D's lightweight heads, reusing the completed fold embedding cache."""
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Instala requirements-action-recognition.txt para usar Optuna.") from error
    output_root = Path(output_root)
    cache_dir = Path(root) / "outputs/action-recognition" / f"x3d-fold{validation_fold}"
    if not (cache_dir / "x3d_embeddings.npz").is_file():
        raise FileNotFoundError(f"No existe el caché X3D del fold {validation_fold}: {cache_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(output_root / 'optuna.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=f"x3d-linear-probe-fold{validation_fold}", storage=storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial):
        parameters = {
            "logistic_c": trial.suggest_float("logistic_c", 0.03, 30.0, log=True),
            "logistic_class_weight": trial.suggest_categorical("logistic_class_weight", ["balanced", None]),
        }
        report = run_x3d_linear_probe(
            root, samples, output_root / "trials" / f"trial-{trial.number:03d}", validation_fold=validation_fold,
            negative_ratio=negative_ratio, seed=seed + trial.number, spec=spec,
            embeddings_cache_dir=cache_dir, **parameters,
        )
        score = report["metrics"]["primary_macro_f1"]
        if score is None:
            raise RuntimeError("Validation produced no primary macro F1")
        return score

    study.optimize(objective, n_trials=trials)
    result = {
        "schema_version": "action_tuning_v1", "family": "x3d_linear_probe", "validation_fold": validation_fold,
        "study": study.study_name, "best_value": study.best_value, "best_params": study.best_params,
        "storage": storage, "completed_trials": len(study.trials), "embedding_cache": str(cache_dir),
    }
    report_path = output_root / "tuning_report.json"
    temporary_path = report_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(report_path)
    return result
