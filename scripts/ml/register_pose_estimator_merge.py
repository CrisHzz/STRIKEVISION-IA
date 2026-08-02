"""Register the fragment-merge pose pipeline in the local MLflow Model Registry.

Usage (from project root, with strikevision env active):

    python scripts/ml/register_pose_estimator_merge.py
    python scripts/ml/register_pose_estimator_merge.py --stage Staging
    python scripts/ml/register_pose_estimator_merge.py --weight models/weights/pose_landmarker_lite.task
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ufc_tracker.detection.weights import project_root
from ufc_tracker.ml.registry import (
    DEFAULT_ALIAS,
    DEFAULT_POSE_WEIGHT_FILENAME,
    POSE_ESTIMATOR_MERGE_DESCRIPTION,
    POSE_ESTIMATOR_MERGE_NAME,
    POSE_EXPERIMENT_NAME,
    configure_mlflow,
    register_pose_estimator_merge,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Register {POSE_ESTIMATOR_MERGE_NAME} in the local MLflow Model Registry."
        )
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=None,
        help=(
            f"Path to MediaPipe .task weights "
            f"(default: models/weights/{DEFAULT_POSE_WEIGHT_FILENAME})"
        ),
    )
    parser.add_argument(
        "--stage",
        default="Production",
        choices=("None", "Staging", "Production", "Archived"),
        help=f"Alias/stage to assign (maps to MLflow alias, default: {DEFAULT_ALIAS}).",
    )
    args = parser.parse_args()

    root = project_root()
    tracking = configure_mlflow(root, experiment=POSE_EXPERIMENT_NAME)
    weight = args.weight
    if weight is not None and not weight.is_absolute():
        weight = (root / weight).resolve()

    version = register_pose_estimator_merge(weight, stage=args.stage, root=root)
    alias = args.stage.lower()
    print(f"Registered model: {POSE_ESTIMATOR_MERGE_NAME}")
    print(f"  version: {version.version}")
    print(f"  alias:   {alias}")
    print(f"  stage:   {version.current_stage}")
    print(f"  run_id:  {version.run_id}")
    print(f"  source:  {weight or f'models/weights/{DEFAULT_POSE_WEIGHT_FILENAME}'}")
    print(f"  URI:     models:/{POSE_ESTIMATOR_MERGE_NAME}@{alias}")
    print(f"  tracking URI: {tracking.as_uri()}")
    print()
    print("Description:")
    print(POSE_ESTIMATOR_MERGE_DESCRIPTION)
    print()
    print("Inputs:")
    print("  video_path  Path to the fight video to analyze")
    print("  frames      Number of frames to analyze from the start of the video")
    print("Output:")
    print("  output_dir  Artifact directory under outputs/predictions/")
    print("              tracking.jsonl, pose.jsonl, pose_preview.mp4,")
    print("              pose_metrics.json, run_metadata.json")
    print()
    print(
        "Open UI with: mlflow ui --backend-store-uri ./mlruns "
        "--default-artifact-root ./mlartifacts"
    )


if __name__ == "__main__":
    main()
