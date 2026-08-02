from pathlib import Path

from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.pipeline import PosePipelineResult, run_pose_pipeline
from ufc_tracker.tracking.merge import MAX_MERGE_DISTANCE, MAX_MERGE_GAP_FRAMES

# Detection score required before a box enters ByteTrack association
TRACKING_CONFIDENCE = 0.5

# How long a fighter may stay out of view and still count as the same person.
# 60 frames is 2 s at 30 fps, enough to survive a camera cut or a clinch.
MERGE_GAP_FRAMES = MAX_MERGE_GAP_FRAMES

# Largest on-screen jump allowed between two fragments, as a fraction of the
# box diagonal so the threshold keeps working under camera zoom.
MERGE_DISTANCE = MAX_MERGE_DISTANCE

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
    out_dir = out_dir / f"{video_path.stem}__first_{frames}_pose_merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ------------------------------------- #
# Main process for external invocation
# ------------------------------------- #
def send_prediction(
    path: Path,
    frames: int,
    tracking_confidence: float = TRACKING_CONFIDENCE,
    merge_gap_frames: int = MERGE_GAP_FRAMES,
    merge_distance: float = MERGE_DISTANCE,
) -> PosePipelineResult:
    """
    Run the pose pipeline with fragment merging on the first `frames` frames of
    a video, writing tracking, keypoints, preview, metrics and metadata to the
    predictions/ folder.

    ByteTrack splits a fighter into several track ids on camera cuts and
    occlusions. This variant chains those fragments into two stable identities
    (fighter_left / fighter_right) before estimating pose, so no per-video
    min_track_frames tuning is required.

    Args:
        path (Path): path to video file (can be relative to project root)
        frames (int): number of frames to process
        tracking_confidence (float): detection score threshold before ByteTrack
        merge_gap_frames (int): longest gap two fragments may span and still merge
        merge_distance (float): largest on-screen jump, in box diagonals

    Returns:
        PosePipelineResult with the paths of the five generated artifacts.
    """
    if frames <= 0:
        raise ValueError(f"frames must be greater than 0, received: {frames}")
    if merge_gap_frames <= 0:
        raise ValueError(
            f"merge_gap_frames must be greater than 0, received: {merge_gap_frames}"
        )
    if merge_distance <= 0:
        raise ValueError(
            f"merge_distance must be greater than 0, received: {merge_distance}"
        )

    print(f"Starting merged pose estimation for {frames} frames...")

    root = project_root()
    video_path = _resolve_video_path(path, root)
    out_dir = _resolve_output_dir(video_path, frames, root)

    print(f"Tracking video: {video_path}")
    print(
        f"Parameters: confidence={tracking_confidence}, "
        f"merge_gap_frames={merge_gap_frames}, merge_distance={merge_distance}"
    )
    print("Running detection, tracking, fragment merge and MediaPipe pose...")

    result = run_pose_pipeline(
        video_path,
        out_dir,
        tracking_confidence=tracking_confidence,
        max_frames=frames,
        merge_track_fragments=True,
        max_merge_gap_frames=merge_gap_frames,
        max_merge_distance=merge_distance,
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
