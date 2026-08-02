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

# Resolve a video path that may be absolute or relative to the project root
def _resolve_video_path(path: Path | str, root: Path) -> Path:
    video_path = Path(path)
    if not video_path.is_absolute():
        video_path = root / video_path
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Could not find video: {video_path}")
    return video_path


# Build the artifact directory for one prediction run
def _resolve_output_dir(video_path: Path, frames: int, root: Path) -> Path:
    out_dir = root / "outputs" / PREDICTIONS_DIRNAME
    out_dir = out_dir / f"{video_path.stem}__first_{frames}_pose"
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
    out_dir = _resolve_output_dir(video_path, frames, root)

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


# -------------
# Try function (uncomment to run example)
# -------------
# fiziev_bahamondes = send_prediction(
#     Path("data/splits/normal_men/fiziev_bahamondes__rafael_fiziev_vs_ignacio_bahamondes__normal_men_round1.mp4"),
#     1000
# )
# print(fiziev_bahamondes)
