from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.pipeline import run_pose_pipeline

DEFAULT_CONFIG = project_root(Path(__file__).resolve()) / "configs" / "app" / "pose_pipeline.yaml"


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pose pipeline config does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Pose pipeline config must be a mapping: {path}")
    for field in ("output_root", "tracking_confidence", "min_track_frames"):
        if field not in data:
            raise ValueError(f"Pose pipeline config missing required field: {field}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MediaPipe pose on one fight round.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Round video path, absolute or relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory. Default: <output_root>/<video_stem>.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--dvc-add",
        action="store_true",
        help="Add the generated pose artifact directory to DVC.",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")

    root = project_root(Path(__file__).resolve())
    config_path = args.config if args.config.is_absolute() else (root / args.config)
    config = load_config(config_path.resolve())
    video_path = args.video if args.video.is_absolute() else root / args.video
    if not video_path.is_file():
        parser.error(f"Video does not exist: {video_path}")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(config["output_root"]) / video_path.stem
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    max_frames = args.max_frames if args.max_frames is not None else config.get("max_frames")
    result = run_pose_pipeline(
        video_path,
        output_dir,
        tracking_confidence=float(config.get("tracking_confidence", 0.5)),
        min_track_frames=int(config.get("min_track_frames", 15)),
        max_frames=max_frames,
    )
    print(f"Pose artifacts:  {result.output_dir}")
    print(f"Tracking:        {result.tracking_path.name}")
    print(f"Pose:            {result.pose_path.name}")
    print(f"Preview:         {result.preview_path.name}")
    print(f"Metrics:         {result.metrics_path.name}")

    if args.dvc_add:
        relative_output = result.output_dir.relative_to(root)
        pointer = Path("data/metadata") / f"pose_{video_path.stem}.dvc"
        subprocess.run(
            ["dvc", "add", "--file", str(pointer), str(relative_output)],
            cwd=root,
            check=True,
        )
        print(f"DVC pointer created: {pointer}")


if __name__ == "__main__":
    main()
