from __future__ import annotations

import argparse
from pathlib import Path

from ufc_tracker.annotations.contracts import load_annotation_config
from ufc_tracker.annotations.session import AnnotationMediaCache, AnnotationStore
from ufc_tracker.detection.weights import project_root
from ufc_tracker.ui.strike_annotation_app import build_app

DEFAULT_CONFIG = (
    project_root(Path(__file__).resolve())
    / "configs"
    / "data"
    / "strike_annotations_v1.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the local StrikeVision strike/no_strike annotator."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--clip-root",
        type=Path,
        default=Path("outputs/annotation_clips"),
        help="Cache directory for one-second original/pose clips.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    root = project_root(Path(__file__).resolve())
    annotation_path = args.annotations if args.annotations.is_absolute() else root / args.annotations
    config_path = args.config if args.config.is_absolute() else root / args.config
    clip_root = args.clip_root if args.clip_root.is_absolute() else root / args.clip_root
    store = AnnotationStore(annotation_path, load_annotation_config(config_path))
    cache = AnnotationMediaCache(root, clip_root)
    build_app(store, cache).queue().launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()

