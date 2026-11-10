"""Extract every original frame into ordered review sheets without annotations."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit("Output is not empty; choose a fresh directory")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise SystemExit("Invalid FPS")
    frames, sheets, index = [], [], 0
    while True:
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (480, 270))
            cv2.rectangle(frame, (0, 0), (205, 27), (0, 0, 0), -1)
            cv2.putText(frame, f'f{index}  {index/fps:.2f}s', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)
            index += 1
        if len(frames) == 16 or (not ok and frames):
            count = len(frames)
            while len(frames) < 16:
                frames.append(np.zeros((270, 480, 3), dtype=np.uint8))
            sheet = np.vstack([np.hstack(frames[i:i+4]) for i in range(0, 16, 4)])
            first = index - count
            filename = f'frames_{first:06d}_{index-1:06d}.jpg'
            if not cv2.imwrite(str(args.output / filename), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86]):
                raise RuntimeError(f'Could not write {filename}')
            sheets.append({'file': filename, 'start_frame': first, 'end_frame': index-1})
            frames = []
        if not ok:
            break
    cap.release()
    manifest = {'video': str(args.video.resolve()), 'fps': fps, 'declared_frames': declared,
                'decoded_frames': index, 'sheets': sheets, 'reviewed': False}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'decoded={index} declared={declared} sheets={len(sheets)} fps={fps:.5f}')
    if index != declared:
        raise SystemExit('Decoded frame count differs from video metadata')


if __name__ == '__main__':
    main()
