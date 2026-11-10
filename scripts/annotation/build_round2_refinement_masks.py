"""Build geometry masks from explicit, visually audited R2 pose references."""
import argparse
import json
import sys
from collections import Counter

import cv2
import numpy as np

from prepare_round2_refinement import EDGES, ROOT, read, round_paths, save, sha

sys.path.insert(0, str(ROOT / 'src'))
from ufc_tracker.pose.estimator import KEYPOINT_NAMES


def filter_pose(row, width, height):
    coordinates = np.zeros((len(KEYPOINT_NAMES), 2), np.float32)
    confidence = np.zeros(len(KEYPOINT_NAMES), np.float32)
    reasons = Counter()
    if not row['visible'] or not row['pose_valid']:
        return coordinates, confidence, 0.0, Counter(pose_invalid=1)
    box = np.asarray(row['bbox_xyxy'], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
        return coordinates, confidence, 0.0, Counter(bbox_invalid=1)
    body_height = box[3] - box[1]
    margin = .15 * body_height
    for index, joint in enumerate(KEYPOINT_NAMES):
        point = row['keypoints'].get(joint)
        if point is None or len(point) != 3 or not np.isfinite(point).all() or not .5 <= point[2] <= 1:
            reasons['joint_missing_or_low_confidence'] += 1
            continue
        x, y, score = point
        if not (0 <= x < width and 0 <= y < height
                and box[0] - margin <= x <= box[2] + margin
                and box[1] - margin <= y <= box[3] + margin):
            reasons['joint_outside_geometry'] += 1
            continue
        coordinates[index], confidence[index] = (x, y), score
    rejected = set()
    for first, second in EDGES:
        a, b = KEYPOINT_NAMES.index(first), KEYPOINT_NAMES.index(second)
        if confidence[a] and confidence[b] and np.linalg.norm(coordinates[a] - coordinates[b]) > .65 * body_height:
            rejected.update((a, b))
    for index in rejected:
        confidence[index] = 0
    reasons['joint_implausible_segment'] = len(rejected)
    coordinates[confidence == 0] = 0
    return coordinates, confidence, body_height, reasons


def build(round_number=2):
    POSE, _, DEST, SOURCE = round_paths(round_number)
    inventory = read(DEST / 'overlay_inventory.json')
    pose_file = POSE / 'pose.jsonl'
    audit_file = DEST / 'identity_decisions.json'
    audit = read(audit_file)
    assert sha(SOURCE) == inventory['annotations_sha256'] == audit['annotations_sha256']
    assert sha(pose_file) == inventory['pose_sha256'] == audit['pose_sha256']
    assert sha(POSE / 'tracking.jsonl') == inventory['tracking_sha256'] == audit['tracking_sha256']
    with pose_file.open(encoding='utf-8') as stream:
        rows = {n: json.loads(line) for n, line in enumerate(stream, 1) if line.strip()}
    annotation = read(SOURCE)
    video = cv2.VideoCapture(str(ROOT / annotation['video']))
    width, height = (int(video.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    assert width > 0 and height > 0
    events = inventory['actions']
    athletes = set(inventory['fighter_counts'])
    assert len(athletes) == 2
    required_frames = {f for a in events for f in range(a['review_window'][0], a['review_window'][1] + 1)}
    decisions = {}
    for item in audit['frames']:
        frame = item['frame_index']
        assert frame in required_frames and frame not in decisions
        assert set(item['fighters']) == athletes
        accepted_lines = set()
        for choice in item['fighters'].values():
            if choice['status'] == 'verified':
                line = choice['line_1based']
                row = rows[line]
                assert line not in accepted_lines and row['frame_index'] == frame
                assert str(row['fighter_id']) == str(choice['fighter_id']) and row['track_id'] == choice['track_id']
                assert isinstance(choice['segment_id'], str) and choice['segment_id']
                accepted_lines.add(line)
            else:
                assert choice['status'] in {'absent', 'wrong_person', 'duplicate', 'unresolved'}
                assert choice.get('line_1based') is None
        decisions[frame] = item['fighters']

    joint_audit_path = DEST / 'joint_exclusions.json'
    joint_exclusions = {}
    if joint_audit_path.is_file():
        joint_audit = read(joint_audit_path)
        assert joint_audit['identity_decisions_sha256'] == sha(audit_file)
        for exclusion in joint_audit['exclusions']:
            start, end = exclusion['frames']
            joint = KEYPOINT_NAMES.index(exclusion['joint'])
            assert exclusion['reason']
            for frame in range(start, end + 1):
                fighter = exclusion['fighter']
                assert decisions[frame][fighter]['status'] == 'verified'
                joint_exclusions.setdefault((frame, fighter), set()).add(joint)

    shape = (len(events), max(a['review_window'][1] - a['review_window'][0] + 1 for a in events),
             2, len(KEYPOINT_NAMES))
    coordinates = np.zeros((*shape, 2), np.float32)
    confidence = np.zeros(shape, np.float32)
    velocity = np.zeros(shape, bool)
    frames = np.full(shape[:2], -1, np.int32)
    references = np.full(shape[:3], -1, np.int32)
    summaries = []
    for index, action in enumerate(events):
        first, last = action['review_window']
        reasons = Counter()
        fighter_order = (action['fighter'], next(iter(athletes - {action['fighter']})))
        previous_segments = [None, None]
        for offset, frame in enumerate(range(first, last + 1)):
            frames[index, offset] = frame
            for slot, fighter in enumerate(fighter_order):
                choice = decisions.get(frame, {}).get(fighter, {'status': 'pending'})
                if choice['status'] != 'verified':
                    reasons[choice['status']] += 1
                    previous_segments[slot] = None
                    continue
                line = choice['line_1based']
                xy, score, body_height, rejected = filter_pose(rows[line], width, height)
                reasons.update(rejected)
                excluded_joints = sorted(joint_exclusions.get((frame, fighter), ()))
                reasons['joint_visual_exclusion'] += int(np.count_nonzero(score[excluded_joints]))
                xy[excluded_joints] = 0
                score[excluded_joints] = 0
                references[index, offset, slot] = line
                segment = (choice['segment_id'], choice['track_id'])
                if offset and previous_segments[slot] == segment and body_height > 0:
                    prev = confidence[index, offset - 1, slot] > 0
                    jump = np.linalg.norm(xy - coordinates[index, offset - 1, slot], axis=-1)
                    bad = prev & (score > 0) & (jump > .30 * body_height)
                    reasons['joint_temporal_jump'] += int(bad.sum())
                    score[bad] = 0
                    xy[bad] = 0
                    velocity[index, offset, slot] = prev & (score > 0)
                previous_segments[slot] = segment
                coordinates[index, offset, slot] = xy
                confidence[index, offset, slot] = score
        active = confidence[index, action['start_frame'] - first:action['end_frame'] - first + 1] > 0
        summaries.append({
            'action_id': action['action_id'], 'label': action['label'], 'fighter': action['fighter'],
            'identity_audit_complete': all(f in decisions for f in range(first, last + 1)),
            'actor_joint_coverage': float(active[:, 0].mean()),
            'opponent_joint_coverage': float(active[:, 1].mean()), 'exclusions': dict(reasons),
        })
    mask = confidence > 0
    assert not coordinates[~mask].any() and np.isfinite(coordinates).all()
    assert not mask[frames < 0].any()
    assert np.all(velocity[:, 1:] <= (mask[:, 1:] & mask[:, :-1]))
    with (DEST / 'geometry_masks.tmp').open('wb') as stream:
        np.savez_compressed(stream, coordinates_px=coordinates, confidence=confidence, frame_indices=frames,
                            point_mask=mask, frame_mask=mask.any(axis=-1), velocity_mask=velocity,
                            pose_line_1based=references, action_ids=[a['action_id'] for a in events],
                            target_mask=[a['label'] != 'ambiguous' for a in events],
                            training_mask=np.zeros(len(events), bool))
    (DEST / 'geometry_masks.tmp').replace(DEST / 'geometry_masks.npz')
    save(DEST / 'geometry_report.json', {
        'source_revision': inventory['source_revision'], 'pose_path': POSE.relative_to(ROOT).as_posix() + '/pose.jsonl',
        'pose_sha256': inventory['pose_sha256'], 'identity_decisions_sha256': sha(audit_file),
        'joint_exclusions_sha256': sha(joint_audit_path) if joint_audit_path.is_file() else None,
        'actions': len(events), 'audited_context_frames': len(decisions), 'required_context_frames': len(required_frames),
        'identity_audit_complete': len(decisions) == len(required_frames), 'training_ready': False,
        'joint_names': KEYPOINT_NAMES, 'axes': ['action', 'context_frame', 'actor_then_opponent', 'joint'],
        'coordinate_units': 'original_video_pixels', 'results': summaries,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=range(1, 6), default=2)
    build(parser.parse_args().round)
