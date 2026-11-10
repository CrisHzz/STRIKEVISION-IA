from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.pipeline import run_pose_pipeline

DEFAULT_CONFIG = project_root(Path(__file__).resolve()) / "configs" / "app" / "pose_pipeline.yaml"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preparation_key(root: Path, video: Path, options: dict[str, Any]) -> str:
    """Conservative cache identity, including video, weights, code and dependencies."""
    from ufc_tracker.ml.registry import resolve_person_detector_weight
    from ufc_tracker.pose.estimator import resolve_mediapipe_pose_model

    code = {}
    for folder in ("pose", "tracking", "detection", "ml"):
        for path in sorted((root / "src/ufc_tracker" / folder).rglob("*.py")):
            code[path.relative_to(root).as_posix()] = file_hash(path)
    payload = {
        "video_sha256": file_hash(video), "video_path": str(video.resolve()),
        "options": options, "code": code,
        "detector": file_hash(Path(resolve_person_detector_weight())),
        "pose_model": file_hash(Path(resolve_mediapipe_pose_model())),
        "packages": {name: version(name) for name in ("torch", "ultralytics", "mediapipe", "numpy", "opencv-contrib-python")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
    parser.add_argument("--device", default=None, help="auto, cpu, cuda:N or GPU index")
    parser.add_argument("--no-preview", action="store_true", help="Skip full video rendering")
    parser.add_argument("--merge-tracks", action="store_true", help="Merge fragments into two fighter slots")
    parser.add_argument("--reuse", action="store_true", help="Reuse only artifacts with matching source, options, weights and hashes")
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
    options = {
        "tracking_confidence": float(config.get("tracking_confidence", 0.5)),
        "min_track_frames": int(config.get("min_track_frames", 15)),
        "max_frames": max_frames,
        "tracking_device": str(args.device or config.get("tracking_device", "auto")),
        "generate_preview": not args.no_preview and bool(config.get("generate_preview", True)),
        "merge_track_fragments": args.merge_tracks or bool(config.get("merge_track_fragments", False)),
    }
    # Resolve auto before hashing so installing CUDA invalidates a prior CPU cache.
    from ufc_tracker.tracking.device import resolve_tracking_device

    options["tracking_device"] = resolve_tracking_device(options["tracking_device"])
    cache_path = output_dir / "preparation_cache.json"
    key = preparation_key(root, video_path, options) if args.reuse else None
    if args.reuse and output_dir.exists() and any(output_dir.iterdir()):
        if not cache_path.is_file():
            parser.error("Existing artifacts have no verified cache manifest. Use a fresh output directory.")
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        required = {"tracking.jsonl", "pose.jsonl", "pose_metrics.json", "run_metadata.json"}
        if options["generate_preview"]:
            required.add("pose_preview.mp4")
        if saved.get("key") != key or set(saved.get("artifacts", {})) != required:
            parser.error("Preparation inputs changed. Use a fresh output directory.")
        for name, digest in saved["artifacts"].items():
            path = output_dir / name
            if not path.is_file() or file_hash(path) != digest:
                parser.error(f"Cached artifact is missing or changed: {name}. Use a fresh output directory.")
        print(f"Reused verified preparation: {output_dir}")
        if args.dvc_add:
            parser.error("Use --dvc-add on the initial run, or version the reused artifacts separately.")
        return
    result = run_pose_pipeline(
        video_path,
        output_dir,
        **options,
    )
    if args.reuse:
        artifacts = [result.tracking_path, result.pose_path, result.metrics_path, result.metadata_path]
        if result.preview_path:
            artifacts.append(result.preview_path)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"key": key, "artifacts": {p.name: file_hash(p) for p in artifacts}}, indent=2), encoding="utf-8")
        temporary.replace(cache_path)
    print(f"Pose artifacts:  {result.output_dir}")
    print(f"Tracking:        {result.tracking_path.name}")
    print(f"Pose:            {result.pose_path.name}")
    print(f"Preview:         {result.preview_path.name if result.preview_path else 'omitted'}")
    print(f"Metrics:         {result.metrics_path.name}")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    print(f"Performance:     {json.dumps(metadata.get('performance', {}))}")

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
