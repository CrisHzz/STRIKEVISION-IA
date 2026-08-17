"""Build one resumable strike-annotation dataset from every round in a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ufc_tracker.annotations.contracts import AnnotationConfig, load_annotation_config
from ufc_tracker.annotations.windows import generate_annotation_windows, read_jsonl, write_jsonl
from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.pipeline import run_pose_pipeline

DEFAULT_ANNOTATION_CONFIG = (
    project_root(Path(__file__).resolve())
    / "configs"
    / "data"
    / "strike_annotations_v1.yaml"
)
DEFAULT_POSE_CONFIG = (
    project_root(Path(__file__).resolve()) / "configs" / "app" / "pose_pipeline.yaml"
)


def _load_pose_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Pose pipeline config must be a mapping: {path}")
    return data


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _pose_artifacts_are_ready(pose_dir: Path) -> bool:
    required = ("pose.jsonl", "pose_metrics.json", "run_metadata.json", "pose_preview.mp4")
    return all((pose_dir / filename).is_file() for filename in required)


def _windows_for_pose_dir(
    pose_dir: Path,
    *,
    config: AnnotationConfig,
    root: Path,
    video_path: Path,
) -> list[dict[str, Any]]:
    pose_path = pose_dir / "pose.jsonl"
    metrics = json.loads((pose_dir / "pose_metrics.json").read_text(encoding="utf-8"))
    fps = float(metrics.get("fps", 0.0))
    frame_count = int(metrics.get("frame_count", 0))
    return generate_annotation_windows(
        read_jsonl(pose_path),
        config=config,
        video_id=video_path.stem,
        video_path=_relative_or_absolute(video_path, root),
        pose_path=_relative_or_absolute(pose_path, root),
        fps=fps,
        frame_count=frame_count,
    )


def _launch_annotator(
    *,
    root: Path,
    output: Path,
    config: AnnotationConfig,
    port: int,
    inbrowser: bool,
) -> None:
    from ufc_tracker.annotations.session import AnnotationMediaResolver, AnnotationStore
    from ufc_tracker.ui.strike_annotation_app import build_app

    store = AnnotationStore(output, config)
    media = AnnotationMediaResolver(root)
    print("Opening Gradio annotator for the combined split...")
    build_app(store, media).queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=inbrowser,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run merged pose estimation for every MP4 in a split and create one "
            "combined strike/no_strike JSONL for continuous annotation."
        )
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_ANNOTATION_CONFIG)
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG)
    parser.add_argument(
        "--pose-output-root",
        type=Path,
        default=Path("data/processed/poses"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--launch-annotator",
        action="store_true",
        help="Open the Gradio annotator for the combined JSONL when preparation finishes.",
    )
    parser.add_argument("--annotator-port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--overwrite-annotations",
        action="store_true",
        help="Allow replacing an existing combined JSONL. Never use after labeling starts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected rounds and output paths without running pose inference.",
    )
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")

    root = project_root(Path(__file__).resolve())
    split_dir = args.split_dir if args.split_dir.is_absolute() else root / args.split_dir
    split_dir = split_dir.resolve()
    if not split_dir.is_dir():
        parser.error(f"Split directory does not exist: {split_dir}")
    videos = sorted(split_dir.glob("*.mp4"))
    if not videos:
        parser.error(f"No MP4 rounds found in: {split_dir}")

    annotation_config_path = args.config if args.config.is_absolute() else root / args.config
    annotation_config = load_annotation_config(annotation_config_path.resolve())
    pose_config_path = args.pose_config if args.pose_config.is_absolute() else root / args.pose_config
    pose_config = _load_pose_config(pose_config_path.resolve())
    pose_root = args.pose_output_root if args.pose_output_root.is_absolute() else root / args.pose_output_root
    pose_root = pose_root.resolve() / split_dir.name
    output = args.output or (
        root / "data" / "annotations" / annotation_config.schema_version / f"{split_dir.name}.jsonl"
    )
    if not output.is_absolute():
        output = root / output
    output = output.resolve()

    if output.exists() and not args.overwrite_annotations:
        if args.launch_annotator and not args.dry_run:
            _launch_annotator(
                root=root,
                output=output,
                config=annotation_config,
                port=args.annotator_port,
                inbrowser=not args.no_browser,
            )
            return
        parser.error(
            f"Annotation output already exists: {output}. "
            "Refusing to replace possible human labels; pass --overwrite-annotations "
            "only before annotation starts."
        )

    print(f"Split: {split_dir}")
    print(f"Rounds: {len(videos)}")
    print(f"Pose artifacts: {pose_root}")
    print(f"Combined annotations: {output}")
    if args.dry_run:
        for video in videos:
            print(f"- {video.name} -> {pose_root / video.stem}")
        return

    combined_windows: list[dict[str, Any]] = []
    for number, video in enumerate(videos, start=1):
        pose_dir = pose_root / video.stem
        if _pose_artifacts_are_ready(pose_dir):
            print(f"[{number}/{len(videos)}] Reusing pose artifacts: {video.name}")
        else:
            print(f"[{number}/{len(videos)}] Processing with Merge: {video.name}")
            run_pose_pipeline(
                video,
                pose_dir,
                tracking_confidence=float(pose_config.get("tracking_confidence", 0.5)),
                min_track_frames=int(pose_config.get("min_track_frames", 15)),
                max_frames=args.max_frames,
                merge_track_fragments=True,
            )
        windows = _windows_for_pose_dir(
            pose_dir,
            config=annotation_config,
            root=root,
            video_path=video,
        )
        combined_windows.extend(windows)
        print(f"  Windows: {len(windows)}")

    window_ids = [str(row["window_id"]) for row in combined_windows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("Duplicate window_id values generated across the split")
    write_jsonl(output, combined_windows)
    print(f"Generated windows: {len(combined_windows)}")
    print(f"Output: {output}")

    if args.launch_annotator:
        _launch_annotator(
            root=root,
            output=output,
            config=annotation_config,
            port=args.annotator_port,
            inbrowser=not args.no_browser,
        )


if __name__ == "__main__":
    main()
