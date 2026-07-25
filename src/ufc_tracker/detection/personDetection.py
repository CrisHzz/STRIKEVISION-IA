from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ufc_tracker.detection import project_root, resolve_pretrained_weight

# Person class id (COCO dataset)
PERSON_CLASS_ID = 0

# Colors for visualization of persons
COLOR_BLUE = (69, 3, 252)
COLOR_RED = (255, 51, 109)
PERSON_COLORS = (COLOR_BLUE, COLOR_RED)

# Default configuration constants
KEEP_TOP_N = 2
MIN_AREA_FRAC = 0.002
ROI_POLYGON = None

SELECT_SHIRTLESS = True
SKIN_FRAC_MIN = 0.35
SKIN_FALLBACK_MIN = 0.15

FIGHTER_SKIN_MIN = 0.50
FIGHTER_MIN_PERSISTENCE = 0.05
FIGHTER_MIN_AREA = 0.01
TRACKER_CFG = "bytetrack.yaml"

# Load the YOLOv8 segmentation model (detects persons)
MODEL_PATH = resolve_pretrained_weight("yolo11n-seg.pt")
model_person = YOLO(str(MODEL_PATH))

# ------------------- #
# Helper/util functions
# ------------------- #

# Return an empty detection output (for consistency)
def _empty_detections():
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    empty_scores = np.empty((0,), dtype=np.float32)
    return empty_boxes, empty_scores, []

# Return a list of visualization colors for n persons
def person_colors(n: int) -> list[tuple[int, int, int]]:
    if n <= 0:
        return []
    if n == 1:
        return [COLOR_BLUE]
    return [PERSON_COLORS[i % 2] for i in range(n)]

# Estimate the region in the box that corresponds to the torso
def _torso_region(box):
    x1, y1, x2, y2 = map(int, box)
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    return (
        x1 + int(0.18 * w),
        y1 + int(0.18 * h),
        x1 + int(0.82 * w),
        y1 + int(0.55 * h),
    )

# Compute appearance metrics of the torso region: fraction black, fraction skin
def torso_appearance(frame, box, silhouette=None):
    tx1, ty1, tx2, ty2 = _torso_region(box)
    ty1, tx1 = max(0, ty1), max(0, tx1)
    ty2 = min(frame.shape[0], ty2)
    tx2 = min(frame.shape[1], tx2)
    if ty2 <= ty1 or tx2 <= tx1:
        return 0.0, 0.0

    crop = frame[ty1:ty2, tx1:tx2]
    mask = np.ones(crop.shape[:2], dtype=bool)

    if silhouette is not None and len(silhouette) >= 3:
        local = np.asarray(silhouette, dtype=np.float32).copy()
        local[:, 0] -= tx1
        local[:, 1] -= ty1
        m = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [local.astype(np.int32)], 1)
        if int(m.sum()) >= 20:
            mask = m.astype(bool)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hch = hsv[:, :, 0][mask]
    sch = hsv[:, :, 1][mask]
    vch = hsv[:, :, 2][mask]
    if len(vch) == 0:
        return 0.0, 0.0

    black = (vch < 60) & (sch < 90)
    skin = ((hch <= 25) | (hch >= 160)) & (sch >= 30) & (vch >= 50)
    return float(black.mean()), float(skin.mean())

# Compute only the skin fraction from torso appearance
def skin_fraction(frame, box, silhouette=None):
    return torso_appearance(frame, box, silhouette)[1]


# ----------------------------- #
# Filtering and detection process
# ----------------------------- #

# Filter detected persons by area, ROI, shirtless/skin fraction etc.
def filter_relevant_people(
    boxes,
    scores,
    silhouettes,
    frame,
    keep_top_n=None,
    min_area_frac=None,
    roi_polygon=None,
    select_shirtless=None,
    skin_min=None,
):
    keep_top_n = KEEP_TOP_N if keep_top_n is None else keep_top_n
    min_area_frac = MIN_AREA_FRAC if min_area_frac is None else min_area_frac
    if roi_polygon is None:
        roi_polygon = ROI_POLYGON
    if select_shirtless is None:
        select_shirtless = SELECT_SHIRTLESS
    skin_min = SKIN_FRAC_MIN if skin_min is None else skin_min

    n = len(boxes)
    if n == 0:
        return boxes, scores, silhouettes

    h, w = frame.shape[:2]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = areas >= (min_area_frac * h * w)

    if roi_polygon is not None:
        poly = np.asarray(roi_polygon, dtype=np.int32)
        cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5).astype(np.float32)
        cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5).astype(np.float32)
        inside = np.array(
            [
                cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                for x, y in zip(cx, cy)
            ]
        )
        keep &= inside

    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return _empty_detections()

    # Optionally, filter for "shirtless" (skin fraction)
    if select_shirtless:
        skin = np.array(
            [skin_fraction(frame, boxes[i], silhouettes[i]) for i in idx],
            dtype=np.float32,
        )
        gated = idx[skin >= skin_min]
        gated_skin = skin[skin >= skin_min]
        if len(gated) == 0:
            best = int(np.argmax(skin))
            if skin[best] >= SKIN_FALLBACK_MIN:
                gated = idx[[best]]
                gated_skin = skin[[best]]
            else:
                return _empty_detections()
        order = np.argsort(gated_skin)[::-1][:keep_top_n]
        idx = gated[order]
    else:
        idx = idx[np.argsort(areas[idx])[::-1][:keep_top_n]]

    idx = np.sort(idx)
    return boxes[idx], scores[idx], [silhouettes[i] for i in idx]

# Main detection process to get person boxes and silhouettes from frame
def detect_person_silhouettes(
    frame,
    conf: float = 0.5,
    keep_top_n=None,
    min_area_frac=None,
    roi_polygon=None,
    select_shirtless=None,
    skin_min=None,
):
    result = model_person(frame, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return _empty_detections()

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(np.int64)
    keep = (classes == PERSON_CLASS_ID) & (scores >= conf)
    indices = np.flatnonzero(keep)
    if len(indices) == 0:
        return _empty_detections()

    boxes = boxes[keep]
    scores = scores[keep]

    silhouettes: list[np.ndarray] = []
    if result.masks is not None:
        for idx in indices:
            poly = result.masks.xy[idx]
            if poly is None or len(poly) == 0:
                silhouettes.append(np.empty((0, 2), dtype=np.float32))
            else:
                silhouettes.append(np.asarray(poly, dtype=np.float32))
    else:
        silhouettes = [np.empty((0, 2), dtype=np.float32) for _ in boxes]

    # Now, filter and select relevant people
    boxes, scores, silhouettes = filter_relevant_people(
        boxes,
        scores,
        silhouettes,
        frame,
        keep_top_n=keep_top_n,
        min_area_frac=min_area_frac,
        roi_polygon=roi_polygon,
        select_shirtless=select_shirtless,
        skin_min=skin_min,
    )
    if len(boxes) == 0:
        return _empty_detections()

    # Sort final result by horizontal position (for consistent coloring left->right)
    order = np.argsort((boxes[:, 0] + boxes[:, 2]) * 0.5)
    boxes = boxes[order]
    scores = scores[order]
    silhouettes = [silhouettes[i] for i in order]
    return boxes, scores, silhouettes


# Detect persons, only returning bounding boxes and class id
def detect_person_box(frame, conf: float = 0.5):
    boxes, scores, _ = detect_person_silhouettes(frame, conf=conf)
    classes = np.full((len(boxes),), PERSON_CLASS_ID, dtype=np.int64)
    return boxes, scores, classes

# ---------------------------------- #
# Visualization and I/O (utility API)
# ---------------------------------- #

# Draw silhouettes (with alpha/outline) and, if provided, bounding boxes
def draw_person_silhouettes(
    frame,
    silhouettes,
    scores=None,
    boxes=None,
    fill_alpha: float = 0.28,
    outline_thickness: int = 2,
):
    overlay = frame.copy()
    colors = person_colors(len(silhouettes))

    # Draw filled polygons (for alpha overlay)
    for i, poly in enumerate(silhouettes):
        if poly is None or len(poly) < 3:
            continue
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], colors[i])

    annotated = (
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0)
        if fill_alpha > 0
        else frame.copy()
    )

    # Draw outlines, labels, or fallback to bounding box if no polygon
    for i, poly in enumerate(silhouettes):
        color = colors[i]
        if poly is None or len(poly) < 3:
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, outline_thickness)
            continue

        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            annotated, [pts], isClosed=True, color=color, thickness=outline_thickness
        )
        x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
        label = "person"
        if scores is not None and i < len(scores):
            label = f"person {scores[i]:.2f}"
        cv2.putText(
            annotated,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated

# Process a single image (file path), return detections and annotated frame
def detect_persons_in_image(image_path, conf: float = 0.5, **filter_kwargs):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    boxes, scores, silhouettes = detect_person_silhouettes(
        frame, conf=conf, **filter_kwargs
    )
    annotated = draw_person_silhouettes(frame, silhouettes, scores=scores, boxes=boxes)
    return frame, boxes, scores, silhouettes, annotated

# Process a video file, yields results for each frame
def detect_persons_in_video(video_path, conf: float = 0.5, max_frames=None, **filter_kwargs):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, scores, silhouettes = detect_person_silhouettes(
                frame, conf=conf, **filter_kwargs
            )
            annotated = draw_person_silhouettes(
                frame, silhouettes, scores=scores, boxes=boxes
            )
            yield frame_idx, frame, boxes, scores, silhouettes, annotated
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()

# ----------------------------- #
# Tracking process (person-by-person)
# ----------------------------- #

# Reset tracker state between videos
def reset_tracker(model):
    """Reset tracker state (between videos)."""
    try:
        model.predictor.trackers[0].reset()
    except Exception:
        pass

# Track all persons through a video (ByteTrack), compute running stats per track_id
def track_video(video_path, max_frames=None, conf=0.5, tracker=TRACKER_CFG):
    """Pass 1: ByteTrack tracking and accumulate stats per track."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    reset_tracker(model_person)
    per_frame = []
    stats = {}
    n = 0
    while True:
        if max_frames is not None and n >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = model_person.track(frame, persist=True, tracker=tracker, verbose=False)[0]
        frame_map = {}
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)
            b = res.boxes.xyxy.cpu().numpy()
            c = res.boxes.cls.cpu().numpy().astype(int)
            s = res.boxes.conf.cpu().numpy()
            for k, tid in enumerate(ids):
                if c[k] != PERSON_CLASS_ID or s[k] < conf:
                    continue
                poly = None
                if res.masks is not None:
                    p = res.masks.xy[k]
                    if p is not None and len(p) >= 3:
                        poly = np.asarray(p, dtype=np.float32)
                _, skin = torso_appearance(frame, b[k], poly)
                area = (b[k, 2] - b[k, 0]) * (b[k, 3] - b[k, 1]) / (h * w)
                cx = (b[k, 0] + b[k, 2]) * 0.5 / w
                frame_map[int(tid)] = (b[k].astype(np.float32), poly)
                d = stats.setdefault(int(tid), {"n": 0, "skin": 0.0, "area": 0.0, "cx": 0.0})
                d["n"] += 1
                d["skin"] += skin
                d["area"] += area
                d["cx"] += cx
        per_frame.append(frame_map)
        n += 1
    cap.release()
    return per_frame, stats, n

# After tracking, select which tracks correspond to the actual "fighters"
def select_fighter_tracks(stats, n_frames):
    """Role assignment: return set of track_ids deemed fighters."""
    fighters = set()
    for tid, d in stats.items():
        cnt = d["n"]
        if cnt == 0:
            continue
        pers = cnt / max(1, n_frames)
        skin = d["skin"] / cnt
        area = d["area"] / cnt
        if (
            skin >= FIGHTER_SKIN_MIN
            and pers >= FIGHTER_MIN_PERSISTENCE
            and area >= FIGHTER_MIN_AREA
        ):
            fighters.add(tid)
    return fighters

# Render an output video with annotation of detected fighters (final pass)
def render_fighters(video_path, per_frame, fighter_tids, out_path, max_frames=None):
    """Pass 2: render video showing only 'fighter' tracks (color left-to-right)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    previews = []
    counts = []
    i = 0
    while True:
        if max_frames is not None and i >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        fm = per_frame[i] if i < len(per_frame) else {}
        items = [(tid, box, poly) for tid, (box, poly) in fm.items() if tid in fighter_tids]
        # Sort by horizontal position for coloring
        items.sort(key=lambda it: (it[1][0] + it[1][2]) * 0.5)
        boxes = (
            np.array([it[1] for it in items], dtype=np.float32)
            if items
            else np.empty((0, 4), np.float32)
        )
        sils = [
            it[2] if it[2] is not None else np.empty((0, 2), np.float32) for it in items
        ]
        annotated = draw_person_silhouettes(frame, sils, boxes=boxes)
        writer.write(annotated)
        counts.append(len(items))
        if i % 10 == 0:
            previews.append((i, annotated.copy()))
        i += 1
    cap.release()
    writer.release()
    return out_path, previews, counts

# ------------------------------------- #
# Main process for external invocation
# ------------------------------------- #
def send_prediction(path: Path, frames: int) -> Path:
    """
    Run the full process on the first `frames` frames of a video,
    outputting an annotated video to the predictions/ folder.

    Args:
        path (Path): path to video file (can be relative to project root)
        frames (int): number of frames to process

    Returns:
        Path to the saved output video.
    """
    if frames <= 0:
        raise ValueError(f"frames must be greater than 0, received: {frames}")
    
    print(f"Starting processing of video for {frames} frames...")

    root = project_root()
    video_path = Path(path)
    if not video_path.is_absolute():
        video_path = root / video_path
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Could not find video: {video_path}")

    print(f"Tracking video: {video_path}")

    per_frame, stats, n_frames = track_video(video_path, max_frames=frames)
    print(f"Tracking complete. {min(n_frames, frames)} frames processed.")

    fighter_tids = select_fighter_tracks(stats, n_frames)

    out_dir = root / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{video_path.stem}__first_{frames}_tracked.mp4"

    print("Rendering fighters and writing output video...")

    cap = cv2.VideoCapture(str(video_path))
    cap.release()

    render_fighters(video_path, per_frame, fighter_tids, out_video, max_frames=frames)
    print(f"Reached the desired frame count: {frames} frames processed and saved.")
    print(f"Processing complete. Output saved to: {out_video}")
    return out_video

# -------------
# Try function (uncomment to run example)
# -------------
# pereira_adensaya = send_prediction(
#     Path("data/splits/aggressive_men/adesanya_pereira_1__israel_adesanya_vs_alex_pereira_1__aggressive_men_round1.mp4"), 
#     1000
# )
# print(pereira_adensaya)
