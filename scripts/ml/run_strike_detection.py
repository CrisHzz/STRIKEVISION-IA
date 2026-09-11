"""Manual phase 3 entry point. No extraction/training without an explicit subcommand."""
import argparse
import json
from pathlib import Path

import yaml

from ufc_tracker.strike.cache import context, cache_path, extract
from ufc_tracker.strike.final_evaluation import run_final_test
from ufc_tracker.strike.training import settings_check, train

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/models/strike_training_v1.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Verify contracts and show selection; no decoding/training")
    extraction = commands.add_parser("extract", help="Manually start/resume frozen feature extraction")
    extraction.add_argument("--split", choices=["train", "validation", "both"], default="both")
    training = commands.add_parser("train", help="Manually train/resume heads from complete caches")
    training.add_argument("--run", required=True)
    testing = commands.add_parser("test", help="Manually open the reserved test once for a fixed candidate")
    testing.add_argument("--run", required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.settings.read_text(encoding="utf-8"))
    settings_check(settings)
    ctx = context(ROOT)
    print(json.dumps({"cache": str(cache_path(ROOT, ctx)), "summary": ctx["summary"],
                      "test": "closed", "command": args.command}, indent=2), flush=True)
    if args.command == "extract":
        splits = ["train", "validation"] if args.split == "both" else [args.split]
        print(extract(ROOT, ctx, settings, splits))
    elif args.command == "train":
        print(train(ROOT, ctx, settings, args.run))
    elif args.command == "test":
        print(run_final_test(ROOT, ctx, settings, args.run))


if __name__ == "__main__":
    main()
