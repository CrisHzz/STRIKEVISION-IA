"""Create derived, read-only action-recognition manifests; does not train a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufc_tracker.action_recognition.dataset import write_action_corpus

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/models/action_recognition_v1.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/datasets/action_recognition_v1",
    )
    args = parser.parse_args()
    report = write_action_corpus(ROOT, args.config, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
