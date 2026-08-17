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
        out_dir = out_dir / f"{video_path.stem}__first_{frames}_pose_merged"
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
    merge_gap_frames: int = MERGE_GAP_FRAMES,
    merge_distance: float = MERGE_DISTANCE,
    output_dir: Path | str | None = None,
) -> PosePipelineResult:
    """
    Run the pose pipeline with fragment merging on the first `frames` frames of
    a video, writing tracking, keypoints, preview, metrics and metadata to the
    predictions/ folder.

    ByteTrack splits a fighter into several track ids on camera cuts and
    occlusions. This variant chains those fragments into two stable identities
    (1 / 2) before estimating pose, so no per-video
    min_track_frames tuning is required.

    Args:
        path (Path): path to video file (can be relative to project root)
        frames (int): number of frames to process
        tracking_confidence (float): detection score threshold before ByteTrack
        merge_gap_frames (int): longest gap two fragments may span and still merge
        merge_distance (float): largest on-screen jump, in box diagonals
        output_dir (Path | str | None): optional artifact directory; defaults to
            outputs/predictions/{video_stem}__first_{frames}_pose_merged

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
    out_dir = _resolve_output_dir(video_path, frames, root, output_dir=output_dir)

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


# Try function (run only as a script, not on package import).
# Edit only VIDEO (relative to data/) and optionally FRAMES / SAVE_DIR.
# if __name__ == "__main__":
#     PROJECT_ROOT = project_root(Path.cwd())

#     VIDEO = "splits/striking_light_grappling_men/topuria_gaethje__topuria_vs_gaethje__striking_light_grappling_men_round2.mp4"
#     FRAMES = 1000
#     SAVE_DIR = PROJECT_ROOT / "outputs/generic-pose-smoke/TEST-POSES-MERGE"

#     result = send_prediction(VIDEO, FRAMES, output_dir=SAVE_DIR)
#     print(result)
