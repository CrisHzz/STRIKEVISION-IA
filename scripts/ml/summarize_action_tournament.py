"""Compare completed action-recognition runs without running or selecting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]


def _summary(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    if not metrics:
        raise ValueError(f"Missing metrics: {report_path}")
    return {
        "run": report_path.parent.name, "family": report.get("family"),
        "validation_fold": report.get("validation_fold"),
        "primary_macro_f1": metrics.get("primary_macro_f1"),
        "all_class_macro_f1": metrics.get("all_class_macro_f1"),
        "primary_macro_average_precision": metrics.get("primary_macro_average_precision"),
        "train_samples": report.get("train_samples"), "validation_samples": report.get("validation_samples"),
        "classes": metrics.get("classes", {}), "report_path": report_path.relative_to(ROOT).as_posix(),
    }


def _mean_and_std(values: list[float | None]) -> dict[str, float | int | None]:
    """Summarize a metric only over reports that actually provide it."""
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": mean(usable),
        "std": pstdev(usable) if len(usable) > 1 else 0.0,
        "count": len(usable),
    }


def _aggregate_by_family(results: list[dict]) -> list[dict]:
    """Aggregate comparable development folds before naming a tournament winner."""
    grouped: dict[str, list[dict]] = {}
    for result in results:
        family = result["family"] or "unknown"
        grouped.setdefault(family, []).append(result)

    aggregates = []
    for family, family_results in grouped.items():
        ordered = sorted(family_results, key=lambda item: (item["validation_fold"] is None, item["validation_fold"]))
        aggregates.append({
            "family": family,
            "runs": [item["run"] for item in ordered],
            "validation_folds": [item["validation_fold"] for item in ordered],
            "primary_macro_f1": _mean_and_std([item["primary_macro_f1"] for item in ordered]),
            "all_class_macro_f1": _mean_and_std([item["all_class_macro_f1"] for item in ordered]),
            "primary_macro_average_precision": _mean_and_std(
                [item["primary_macro_average_precision"] for item in ordered]
            ),
        })
    aggregates.sort(
        key=lambda item: (
            item["primary_macro_f1"]["mean"] is None,
            -(item["primary_macro_f1"]["mean"] or -1),
            item["family"],
        )
    )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run names under the selected output root.")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs/action-recognition")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/action-recognition/tournament.json")
    args = parser.parse_args()
    runs_root = args.runs_root if args.runs_root.is_absolute() else ROOT / args.runs_root
    results = [_summary(runs_root / run / "report.json") for run in args.runs]
    results.sort(key=lambda item: (item["family"] or "", item["validation_fold"] is None, item["validation_fold"], item["run"]))
    aggregates = _aggregate_by_family(results)
    payload = {
        "schema_version": "action_tournament_v2",
        "selection_rule": "rank families by mean development primary_macro_f1 across matching validation folds; do not use held-out test to choose a winner",
        "runs": results,
        "model_aggregates": aggregates,
        "winner_on_provided_development_aggregates": aggregates[0]["family"] if aggregates else None,
        "note": "A family is comparable only when it was evaluated on the same folds as its competitors. Add latency and memory measurements before choosing a production candidate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
