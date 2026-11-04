"""Manual commands for preparing and training action-recognition candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ufc_tracker.action_recognition.dataset import write_action_corpus
from ufc_tracker.action_recognition.features import PoseFeatureSpec
from ufc_tracker.action_recognition.training import read_samples, run_classical_baseline, run_temporal_model, run_x3d_linear_probe
from ufc_tracker.action_recognition.tuning import tune_temporal_model, tune_x3d_linear_probe

ROOT = Path(__file__).resolve().parents[2]


def _settings(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "action_training_v1":
        raise ValueError("Unsupported action training configuration")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "baseline", "temporal", "rgb", "tune"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/models/action_training_v1.yaml")
    parser.add_argument("--run", default=None, help="Required for every training or tuning command.")
    parser.add_argument("--model", choices=("logistic_regression", "extra_trees", "tcn", "hierarchical_tcn", "stgcn_compact", "x3d_linear_probe"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trials", type=int, default=20, help="Only used by tune.")
    parser.add_argument("--parameters-from", type=Path, help="Reuse best_params from a completed tuning report for one temporal run.")
    args = parser.parse_args()
    config = _settings(args.config)
    dataset_root = ROOT / config["dataset_root"]
    if args.command == "prepare":
        report = write_action_corpus(ROOT, ROOT / config["dataset_config"], dataset_root)
        print(json.dumps({key: value for key, value in report.items() if key != "sources"}, indent=2))
        return
    if not args.run or not args.model:
        parser.error("--run and --model are required for training commands")
    samples = read_samples(dataset_root / "samples.jsonl")
    spec = PoseFeatureSpec(**config["feature_extraction"])
    output = ROOT / config["output_root"] / args.run
    common = {"root": ROOT, "samples": samples, "output_dir": output, "validation_fold": args.fold,
              "negative_ratio": float(config["sampling"]["negative_to_positive_ratio"]), "seed": int(config["seed"]), "spec": spec}
    if args.command == "baseline":
        if args.model not in config["models"]["baseline"]:
            parser.error("--model must be a configured baseline")
        report = run_classical_baseline(name=args.model, **common)
    elif args.command == "temporal":
        if args.model not in config["models"]["temporal"]:
            parser.error("--model must be a configured temporal model")
        tune = config["tuning"]
        parameters = {}
        if args.parameters_from:
            source = json.loads(args.parameters_from.read_text(encoding="utf-8"))
            parameters = dict(source.get("best_params", {}))
            allowed = {"learning_rate", "weight_decay", "channels", "blocks", "dropout", "batch_size"}
            unknown = set(parameters) - allowed
            if unknown or not parameters:
                raise ValueError(f"Invalid best_params in {args.parameters_from}: unknown={sorted(unknown)}")
        report = run_temporal_model(
            name=args.model, max_epochs=int(tune["max_epochs"]), patience=int(tune["patience"]),
            **common, **parameters,
        )
    elif args.command == "rgb":
        if args.model not in config["models"]["video"]:
            parser.error("--model must be a configured RGB model")
        report = run_x3d_linear_probe(**common)
    else:
        if args.model in config["models"]["temporal"]:
            report = tune_temporal_model(
                ROOT, samples, output, name=args.model, validation_fold=args.fold,
                negative_ratio=float(config["sampling"]["negative_to_positive_ratio"]), seed=int(config["seed"]),
                spec=spec, trials=args.trials,
            )
        elif args.model == "x3d_linear_probe":
            report = tune_x3d_linear_probe(
                ROOT, samples, output, validation_fold=args.fold,
                negative_ratio=float(config["sampling"]["negative_to_positive_ratio"]), seed=int(config["seed"]),
                spec=spec, trials=args.trials,
            )
        else:
            parser.error("--model must be a configured temporal or video finalist")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
