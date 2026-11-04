"""Evaluate two completed action-recognition candidates on common validation clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufc_tracker.action_recognition.comparison import compare_on_common_validation_samples


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-prefix", default="x3d")
    parser.add_argument("--reference-prefix", default="tcn")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--candidate-dir", type=Path, help="Compare one explicit completed run.")
    parser.add_argument("--reference-dir", type=Path, help="Reference for --candidate-dir.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/action-recognition/common-subset-x3d-vs-tcn.json")
    args = parser.parse_args()
    comparisons = []
    if args.candidate_dir or args.reference_dir:
        if not args.candidate_dir or not args.reference_dir:
            parser.error("--candidate-dir and --reference-dir must be provided together")
        candidate = args.candidate_dir if args.candidate_dir.is_absolute() else ROOT / args.candidate_dir
        reference = args.reference_dir if args.reference_dir.is_absolute() else ROOT / args.reference_dir
        comparisons.append(compare_on_common_validation_samples(candidate, reference))
    else:
        for fold in args.folds:
            candidate = ROOT / "outputs/action-recognition" / f"{args.candidate_prefix}-fold{fold}"
            reference = ROOT / "outputs/action-recognition" / f"{args.reference_prefix}-fold{fold}"
            comparisons.append(compare_on_common_validation_samples(candidate, reference))
    def average(side: str, metric: str) -> float:
        return sum(item[f"{side}_metrics"][metric] for item in comparisons) / len(comparisons)
    direct = args.candidate_dir is not None
    payload = {
        "schema_version": "action_common_subset_summary_v1",
        "candidate_prefix": None if direct else args.candidate_prefix,
        "reference_prefix": None if direct else args.reference_prefix,
        "candidate_dir": str(args.candidate_dir) if args.candidate_dir else None,
        "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        "folds": comparisons,
        "mean_primary_macro_f1": {
            "candidate": average("candidate", "primary_macro_f1"),
            "reference": average("reference", "primary_macro_f1"),
        },
        "mean_primary_macro_average_precision": {
            "candidate": average("candidate", "primary_macro_average_precision"),
            "reference": average("reference", "primary_macro_average_precision"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    display = {
        "candidate_prefix": None if direct else args.candidate_prefix,
        "reference_prefix": None if direct else args.reference_prefix,
        "candidate_dir": str(args.candidate_dir) if args.candidate_dir else None,
        "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        "folds": [
            {
                "candidate": item["candidate"], "reference": item["reference"],
                "common_validation_samples": item["common_validation_samples"],
                "candidate_primary_macro_f1": item["candidate_metrics"]["primary_macro_f1"],
                "reference_primary_macro_f1": item["reference_metrics"]["primary_macro_f1"],
            }
            for item in comparisons
        ],
        "mean_primary_macro_f1": payload["mean_primary_macro_f1"],
        "saved_to": str(args.output),
    }
    print(json.dumps(display, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
