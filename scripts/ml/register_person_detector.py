"""Register the person-detection YOLO weights in the local MLflow Model Registry.

Usage (from project root, with strikevision env active):

    python scripts/ml/register_person_detector.py
    python scripts/ml/register_person_detector.py --stage Staging
    python scripts/ml/register_person_detector.py --weight models/weights/yolo11n-seg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ufc_tracker.detection.weights import project_root, resolve_pretrained_weight
from ufc_tracker.ml.registry import (
    DEFAULT_ALIAS,
    DEFAULT_WEIGHT_FILENAME,
    PERSON_DETECTOR_DESCRIPTION,
    PERSON_DETECTOR_NAME,
    configure_mlflow,
    register_person_detector,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Register {PERSON_DETECTOR_NAME} in the local MLflow Model Registry."
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=None,
        help=f"Path to .pt weights (default: models/weights/{DEFAULT_WEIGHT_FILENAME})",
    )
    parser.add_argument(
        "--stage",
        default="Production",
        choices=("None", "Staging", "Production", "Archived"),
        help=f"Alias/stage to assign (maps to MLflow alias, default: {DEFAULT_ALIAS}).",
    )
    args = parser.parse_args()

    root = project_root()
    tracking = configure_mlflow(root)
    weight = args.weight
    if weight is None:
        weight = resolve_pretrained_weight(DEFAULT_WEIGHT_FILENAME, root=root)
    elif not weight.is_absolute():
        weight = (root / weight).resolve()

    version = register_person_detector(weight, stage=args.stage, root=root)
    alias = args.stage.lower()
    print(f"Registered model: {PERSON_DETECTOR_NAME}")
    print(f"  version: {version.version}")
    print(f"  alias:   {alias}")
    print(f"  stage:   {version.current_stage}")
    print(f"  run_id:  {version.run_id}")
    print(f"  source:  {weight}")
    print(f"  URI:     models:/{PERSON_DETECTOR_NAME}@{alias}")
    print(f"  tracking URI: {tracking.as_uri()}")
    print()
    print("Description:")
    print(PERSON_DETECTOR_DESCRIPTION)
    print()
    print("Inputs:")
    print("  video_path  Path to the fight video to analyze")
    print("  frames      Number of frames to analyze from the start of the video")
    print("Output:")
    print("  output_video_path  Annotated MP4 under outputs/predictions/")
    print()
    print(
        "Open UI with: mlflow ui --backend-store-uri ./mlruns "
        "--default-artifact-root ./mlartifacts"
    )


if __name__ == "__main__":
    main()
