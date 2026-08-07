from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufc_tracker.annotations.contracts import load_annotation_config
from ufc_tracker.annotations.windows import (
    generate_annotation_windows,
    read_jsonl,
    write_jsonl,
)
from ufc_tracker.detection.weights import project_root

DEFAULT_CONFIG = (
    project_root(Path(__file__).resolve())
    / "configs"
    / "data"
    / "strike_annotations_v1.yaml"
)


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unlabeled strike/no_strike windows from pose artifacts."
    )
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--video-id", default=None)
    args = parser.parse_args()

    root = project_root(Path(__file__).resolve())
    pose_dir = args.pose_dir if args.pose_dir.is_absolute() else root / args.pose_dir
    pose_dir = pose_dir.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_annotation_config(config_path.resolve())

    pose_path = pose_dir / "pose.jsonl"
    metrics_path = pose_dir / "pose_metrics.json"
    metadata_path = pose_dir / "run_metadata.json"
    for required in (pose_path, metrics_path, metadata_path):
        if not required.is_file():
            parser.error(f"Missing pose artifact: {required}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_path = str(metadata.get("video", ""))
    if not video_path:
        parser.error(f"run_metadata.json does not contain video: {metadata_path}")
    fps = float(metrics.get("fps", 0.0))
    frame_count = int(metrics.get("frame_count", 0))
    video_id = args.video_id or Path(video_path).stem
    output = args.output
    if output is None:
        output = (
            root
            / "data"
            / "annotations"
            / config.schema_version
            / f"{video_id}.jsonl"
        )
    elif not output.is_absolute():
        output = root / output

    windows = generate_annotation_windows(
        read_jsonl(pose_path),
        config=config,
        video_id=video_id,
        video_path=video_path.replace("\\", "/"),
        pose_path=_relative_or_absolute(pose_path, root),
        fps=fps,
        frame_count=frame_count,
    )
    write_jsonl(output, windows)
    auto_unknown = sum(
        window["quality"]["status"] == "auto_unknown" for window in windows
    )
    print(f"Generated windows: {len(windows)}")
    print(f"Review candidates: {len(windows) - auto_unknown}")
    print(f"Suggested unknown: {auto_unknown}")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()

