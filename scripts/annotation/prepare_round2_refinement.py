"""Verify the R2 sources and prepare evidence for manual pose identity decisions."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/annotation_audits/volkanovski_holloway_i'
STEM = 'alexander_volkanovski_vs_max_holloway_i_ufc_245__t_5G7bGTPxs'
EDGES = (
    ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'),
    *((side + '_' + a, side + '_' + b) for side in ('left', 'right')
      for a, b in (('shoulder', 'hip'), ('shoulder', 'elbow'), ('elbow', 'wrist'),
                   ('hip', 'knee'), ('knee', 'ankle'))),
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def round_paths(round_number):
    round_name = f'round{round_number}'
    stem = f'{STEM}__{round_name}'
    return (
        ROOT / 'data/processed/poses_direct_actions' / stem,
        BASE / f'{round_name}_all_frames',
        BASE / f'{round_name}_refinement',
        ROOT / 'data/annotations/action_strikes_direct_v1' / f'{stem}.json',
    )


def main(inventory_only=False, round_number=2):
    POSE, RGB, DEST, SOURCE = round_paths(round_number)
    manifest = read(RGB / 'manifest.json')
    data = read(SOURCE)
    video = Path(manifest['video'])
    frames, fps = manifest['decoded_frames'], manifest['fps']
    sheets = manifest['sheets']
    expected = [(start, min(start + 15, frames - 1)) for start in range(0, frames, 16)]
    assert data['source_sha256'] == sha(RGB / 'manifest.json')
    assert (data['frame_count'], data['fps']) == (frames, fps)
    assert [(s['start_frame'], s['end_frame']) for s in sheets] == expected
    reviewed_sheets = [s for s in data['reviewed_ranges'] if s.get('status') == 'reviewed']
    assert sorted((s['start_frame'], s['end_frame']) for s in reviewed_sheets) == expected
    assert all((RGB / s['file']).is_file() for s in sheets)
    actions = data['actions']
    assert len({a['action_id'] for a in actions}) == len(actions)
    assert len({(a['start_frame'], a['end_frame'], a['fighter'], a['label']) for a in actions}) == len(actions)
    margin = round(fps * .8 / 2)
    needed = set()
    events = []
    for a in actions:
        start, end = a['start_frame'], a['end_frame']
        assert 0 <= start <= end < frames
        assert a['evidence'] and all((ROOT / p).is_file() for p in a['evidence'])
        first, last = max(0, start - margin), min(frames - 1, end + margin)
        indices = list(range(first // 16, last // 16 + 1))
        needed.update(indices)
        events.append({
            **{k: a[k] for k in ('action_id', 'label', 'fighter', 'start_frame', 'end_frame')},
            'review_window': [first, last],
            'overlay_sheets': [sheets[i]['file'] for i in indices],
            'identity_review': 'pending',
        })
    DEST.mkdir(exist_ok=True)
    report = {
        'source_revision': data['revision'], 'annotations_sha256': sha(SOURCE),
        'manifest_sha256': data['source_sha256'], 'video_sha256': sha(video),
        'fps': fps, 'frame_count': frames, 'reviewed_sheets': len(sheets),
        'class_counts': dict(Counter(a['label'] for a in actions)),
        'fighter_counts': {fighter: dict(Counter(a['label'] for a in actions if a['fighter'] == fighter))
                           for fighter in sorted({a['fighter'] for a in actions})},
        'simultaneous_positive_pairs': sum(
            a['fighter'] != b['fighter'] and 'ambiguous' not in (a['label'], b['label'])
            and a['start_frame'] <= b['end_frame'] and b['start_frame'] <= a['end_frame']
            for i, a in enumerate(actions) for b in actions[i + 1:]),
        'candidate_overlay_sheets': len(needed), 'context_margin_frames': margin,
        'visual_annotation_complete': True, 'identity_audit_complete': False,
        'training_ready': False, 'actions': events,
    }
    if inventory_only:
        save(DEST / 'source_inventory.json', report)
        print({key: value for key, value in report.items() if key != 'actions'})
        return

    cache = read(POSE / 'preparation_cache.json')
    for name in ('tracking.jsonl', 'pose.jsonl', 'run_metadata.json', 'pose_metrics.json'):
        assert cache['artifacts'][name] == sha(POSE / name), name
    metrics = read(POSE / 'pose_metrics.json')
    assert metrics['frame_count'] == frames and math.isclose(metrics['fps'], fps)
    assert (ROOT / metrics['video']).resolve() == video.resolve()
    tracking = {}
    with (POSE / 'tracking.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                key = row['frame_index'], str(row['fighter_id'])
                assert 0 <= key[0] < frames and key not in tracking
                tracking[key] = row['track_id']
    rows = defaultdict(list)
    seen = set()
    with (POSE / 'pose.jsonl').open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame, slot = row['frame_index'], str(row['fighter_id'])
            assert 0 <= frame < frames and (frame, slot) not in seen
            if row['visible']:
                assert tracking.get((frame, slot)) == row['track_id']
            seen.add((frame, slot))
            if frame // 16 in needed:
                rows[frame].append((line_number, row))
    cap = cv2.VideoCapture(str(video))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    assert width > 0 and height > 0
    overlay = DEST / 'overlays'
    overlay.mkdir(exist_ok=True)
    for i in sorted(needed):
        sheet = sheets[i]
        with Image.open(RGB / sheet['file']) as source:
            image = source.convert('RGB')
        tile_width, tile_height = image.width // 4, image.height // 4
        for frame in range(sheet['start_frame'], sheet['end_frame'] + 1):
            tile = frame - sheet['start_frame']
            x, y = tile % 4 * tile_width, tile // 4 * tile_height
            crop = image.crop((x, y, x + tile_width, y + tile_height))
            draw = ImageDraw.Draw(crop)
            for index, (line_number, row) in enumerate(rows[frame]):
                slot = str(row['fighter_id'])
                color = '#ff5757' if slot == '1' else '#dc69ff'
                def scale(point):
                    return point[0] * tile_width / width, point[1] * tile_height / height
                box = row['bbox_xyxy']
                if box:
                    draw.rectangle([scale(box[:2]), scale(box[2:])], outline=color, width=2)
                points = row['keypoints']
                for a, b in EDGES:
                    if points.get(a) and points.get(b) and min(points[a][2], points[b][2]) >= .5:
                        draw.line([scale(points[a]), scale(points[b])], fill=color, width=2)
                label_y = tile_height - 17 * (index + 1)
                draw.rectangle((0, label_y, 210, label_y + 15), fill='black')
                draw.text((3, label_y), f"ID {slot} T{row['track_id']} | fila {line_number}", fill=color)
            image.paste(crop, (x, y))
        image.save(overlay / sheet['file'], quality=93)
    report['pose_sha256'] = cache['artifacts']['pose.jsonl']
    report['tracking_sha256'] = cache['artifacts']['tracking.jsonl']
    report['overlay_sheets_generated'] = len(needed)
    save(DEST / 'overlay_inventory.json', report)
    print(f'{len(needed)} hojas con overlay, {len(events)} acciones pendientes de auditoría.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=range(1, 6), default=2)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    main(args.inventory_only, args.round)
