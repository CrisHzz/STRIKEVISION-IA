from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ufc_tracker.annotations.contracts import load_annotation_config
from ufc_tracker.annotations.session import AnnotationMediaResolver, AnnotationStore
from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.pipeline import ensure_browser_compatible_preview
from ufc_tracker.ui.split_annotation_app import SplitAnnotationApp, build_split_app
from ufc_tracker.ui.strike_annotation_app import build_app

DEFAULT_CONFIG = (
    project_root(Path(__file__).resolve())
    / "configs"
    / "data"
    / "strike_annotations_v1.yaml"
)
DEFAULT_POSE_CONFIG = (
    project_root(Path(__file__).resolve()) / "configs" / "app" / "pose_pipeline.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the local StrikeVision strike/no_strike annotator."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--annotations", type=Path)
    source.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help=(
            "Splits root or one category folder. Defaults to data/splits so "
            "you can pick the subfolder and then the video."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG)
    parser.add_argument(
        "--pose-output-root",
        type=Path,
        default=Path("data/processed/poses"),
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("data/annotations/strike_annotations_v1"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    root = project_root(Path(__file__).resolve())
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_annotation_config(config_path)
    media = AnnotationMediaResolver(root)
    if args.annotations is not None:
        annotation_path = (
            args.annotations if args.annotations.is_absolute() else root / args.annotations
        )
        store = AnnotationStore(annotation_path, config)
        preview_path = media.source_for(store.row(0), "pose_preview")
        if ensure_browser_compatible_preview(preview_path):
            print(f"Optimized browser preview: {preview_path}")
        app = build_app(store, media)
    else:
        split_dir = args.split_dir or Path("data/splits")
        split_dir = split_dir if split_dir.is_absolute() else root / split_dir
        pose_config_path = (
            args.pose_config if args.pose_config.is_absolute() else root / args.pose_config
        )
        pose_config = yaml.safe_load(pose_config_path.read_text(encoding="utf-8"))
        if not isinstance(pose_config, dict):
            parser.error(f"Pose config must be a mapping: {pose_config_path}")
        pose_output_root = (
            args.pose_output_root
            if args.pose_output_root.is_absolute()
            else root / args.pose_output_root
        )
        annotation_root = (
            args.annotation_root
            if args.annotation_root.is_absolute()
            else root / args.annotation_root
        )
        controller = SplitAnnotationApp(
            root=root,
            split_dir=split_dir,
            config=config,
            pose_config=pose_config,
            pose_root=pose_output_root,
            annotation_root=annotation_root,
            media=media,
        )
        app = build_split_app(controller)
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
