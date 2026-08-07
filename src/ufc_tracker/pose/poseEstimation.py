from pathlib import Path

from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.pipeline import PosePipelineResult, run_pose_pipeline

# Detection score required before a box enters ByteTrack association
TRACKING_CONFIDENCE = 0.5

# Minimum frames a ByteTrack fragment must last to be treated as a fighter.
# The dev notebook needed 50: with a lower threshold, short fragments add a
# third "fighter" to the same frame and the pipeline aborts.
MIN_TRACK_FRAMES = 50

# Prediction artifacts live next to the person-detection outputs
PREDICTIONS_DIRNAME = "predictions"


# ------------------------------------- #
# Helper/util functions
# ------------------------------------- #

# Resolve a video path that may be absolute, project-relative, or under data/
def _resolve_video_path(path: Path | str, root: Path) -> Path:
    raw = str(path).strip().replace("\\", "/")
    candidates = [Path(raw)]
    if not candidates[0].is_absolute():
        candidates.append(root / raw)
        # Allow "splits/..." without repeating "data/"
        if not raw.startswith("data/"):
            candidates.append(root / "data" / raw)
        # Allow "data/splits/..." written without the project root
        if raw.startswith("data/"):
            candidates.append(root / raw)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        "Could not find video. Tried:\n  - "
        + "\n  - ".join(str(path.resolve()) for path in seen)
        + "\nPass a path relative to data/, e.g. "
        "'splits/category/fight_round.mp4'."
    )


# Build the artifact directory for one prediction run
def _resolve_output_dir(
    video_path: Path,
    frames: int,
    root: Path,
    output_dir: Path | str | None = None,
) -> Path:
    if output_dir is not None:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    else:
        out_dir = root / "outputs" / PREDICTIONS_DIRNAME
        out_dir = out_dir / f"{video_path.stem}__first_{frames}_pose"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ------------------------------------- #
# Main process for external invocation
# ------------------------------------- #
def send_prediction(
    path: Path,
    frames: int,
    tracking_confidence: float = TRACKING_CONFIDENCE,
    min_track_frames: int = MIN_TRACK_FRAMES,
    output_dir: Path | str | None = None,
) -> PosePipelineResult:
    """
    Run the full pose pipeline on the first `frames` frames of a video,
    writing tracking, keypoints, preview, metrics and metadata to the
    predictions/ folder.

    Args:
        path (Path): path to video file (can be relative to project root)
        frames (int): number of frames to process
        tracking_confidence (float): detection score threshold before ByteTrack
        min_track_frames (int): minimum frames a track must last to be a fighter
        output_dir (Path | str | None): optional artifact directory; defaults to
            outputs/predictions/{video_stem}__first_{frames}_pose

    Returns:
        PosePipelineResult with the paths of the five generated artifacts.
    """
    if frames <= 0:
        raise ValueError(f"frames must be greater than 0, received: {frames}")
    if min_track_frames <= 0:
        raise ValueError(
            f"min_track_frames must be greater than 0, received: {min_track_frames}"
        )

    print(f"Starting pose estimation for {frames} frames...")

    root = project_root()
    video_path = _resolve_video_path(path, root)
    out_dir = _resolve_output_dir(video_path, frames, root, output_dir=output_dir)

    print(f"Tracking video: {video_path}")
    print(
        f"Parameters: confidence={tracking_confidence}, "
        f"min_track_frames={min_track_frames}"
    )
    print("Running detection, tracking and MediaPipe pose...")

    result = run_pose_pipeline(
        video_path,
        out_dir,
        tracking_confidence=tracking_confidence,
        min_track_frames=min_track_frames,
        max_frames=frames,
    )

    print(f"Reached the desired frame count: {frames} frames processed and saved.")
    print(f"Processing complete. Artifacts saved to: {result.output_dir}")
    print(f"  Tracking: {result.tracking_path.name}")
    print(f"  Pose:     {result.pose_path.name}")
    print(f"  Preview:  {result.preview_path.name}")
    print(f"  Metrics:  {result.metrics_path.name}")
    print(f"  Metadata: {result.metadata_path.name}")
    return result


# Try function (run only as a script, not on package import).
# Edit only VIDEO (relative to data/) and optionally FRAMES / SAVE_DIR.
# if __name__ == "__main__":
#     PROJECT_ROOT = project_root(Path.cwd())

#     VIDEO = "splits/striking_light_grappling_men/topuria_gaethje__topuria_vs_gaethje__striking_light_grappling_men_round2.mp4"
#     FRAMES = 1000
#     SAVE_DIR = PROJECT_ROOT / "outputs/generic-pose-smoke/TEST-POSES"

#     result = send_prediction(VIDEO, FRAMES, output_dir=SAVE_DIR)
#     print(result)
