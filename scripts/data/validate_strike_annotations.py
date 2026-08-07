from __future__ import annotations

import argparse
from pathlib import Path

from ufc_tracker.annotations.contracts import load_annotation_config, validate_annotation_rows
from ufc_tracker.annotations.windows import read_jsonl
from ufc_tracker.detection.weights import project_root

DEFAULT_CONFIG = (
    project_root(Path(__file__).resolve())
    / "configs"
    / "data"
    / "strike_annotations_v1.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strike_annotations_v1 JSONL files.")
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--require-labeled",
        action="store_true",
        help="Fail when a window still has label=null.",
    )
    args = parser.parse_args()

    root = project_root(Path(__file__).resolve())
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_annotation_config(config_path.resolve())
    total_rows = 0
    total_errors = 0
    for raw_path in args.annotations:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        rows = read_jsonl(path.resolve())
        errors = validate_annotation_rows(
            rows,
            config,
            require_labeled=args.require_labeled,
        )
        total_rows += len(rows)
        total_errors += len(errors)
        if errors:
            print(f"INVALID {path.resolve()} ({len(errors)} errors)")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK {path.resolve()} ({len(rows)} windows)")

    print(f"Validated windows: {total_rows}")
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

