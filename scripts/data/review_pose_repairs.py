"""Generate old/new preview contact sheets for staged repairs (no installation)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ufc_tracker.pose.pipeline import FIGHTER_COLORS, _draw_pose, load_pose_records

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs/dataset-repair-20260903"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, type=int, choices=range(1, 6))
    args = parser.parse_args()
    staged = next((WORK / "staged").glob(f"*round{args.round}"))
    old = ROOT / "data/processed/poses/aggressive_men" / staged.name
    metrics = [json.loads((folder / "pose_metrics.json").read_text()) for folder in (old, staged)]
    if any(metrics[0][key] != metrics[1][key] for key in ("frame_count", "fps", "video")):
        raise RuntimeError("Old/new source contracts do not match")
    frame_count, fps = metrics[0]["frame_count"], metrics[0]["fps"]
    capture = cv2.VideoCapture(str(ROOT / metrics[0]["video"]))
    if not capture.isOpened():
        raise RuntimeError("Source video is not decodable")
    grouped = []
    for folder in (old, staged):
        by_frame = defaultdict(list)
        for record in load_pose_records(folder / "pose.jsonl"):
            by_frame[record.frame_index].append(record)
        grouped.append(by_frame)
    sheet = []
    try:
        for frame in np.linspace(int(fps * 5), frame_count - int(fps * 5), 8).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, source = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode QA frame {frame}")
            pair = []
            for title, records in zip(("BEFORE", "AFTER"), grouped):
                image = source.copy()
                for slot, record in enumerate(records[frame]):
                    _draw_pose(image, record, FIGHTER_COLORS[slot % 2])
                image = cv2.resize(image, (640, 360))
                cv2.rectangle(image, (0, 0), (640, 28), (0, 0, 0), -1)
                cv2.putText(image, f"{title} R{args.round} {frame / fps:.1f}s", (10, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                pair.append(image)
            sheet.append(np.hstack(pair))
    finally:
        capture.release()
    output = WORK / f"qa_round{args.round}.jpg"
    if not cv2.imwrite(str(output), np.vstack(sheet)):
        raise RuntimeError("Failed to save QA contact sheet")
    print(output)


if __name__ == "__main__":
    main()
