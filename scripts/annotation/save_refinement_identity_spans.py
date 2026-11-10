"""Save identity decisions for inspected frames and regenerate geometry masks."""
import argparse
import json

from build_round2_refinement_masks import build
from prepare_round2_refinement import read, round_paths, save, sha


def commit(round_number, ranges):
    pose, rgb, destination, source = round_paths(round_number)
    inventory = read(destination / 'overlay_inventory.json')
    expected_hashes = {
        'annotations_sha256': sha(source),
        'pose_sha256': sha(pose / 'pose.jsonl'),
        'tracking_sha256': sha(pose / 'tracking.jsonl'),
    }
    assert all(inventory[key] == value for key, value in expected_hashes.items())
    path = destination / 'identity_decisions.json'
    saved = read(path) if path.exists() else {**expected_hashes, 'frames': []}
    assert all(saved[key] == value for key, value in expected_hashes.items())
    manifest = read(rgb / 'manifest.json')
    decisions = {entry['frame_index']: entry for entry in saved['frames']}
    required = {f for a in inventory['actions'] for f in range(a['review_window'][0], a['review_window'][1] + 1)}
    with (pose / 'pose.jsonl').open(encoding='utf-8') as stream:
        rows = {}
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                rows[(row['frame_index'], str(row['fighter_id']))] = (line_number, row)
    for first, last, holloway, volkanovski in ranges:
        assert 0 <= first <= last < inventory['frame_count']
        for frame in range(first, last + 1):
            if frame not in required:
                continue
            sheet = manifest['sheets'][frame // 16]['file']
            assert (destination / 'overlays' / sheet).is_file()
            fighters = {}
            for name, slot in (('Max Holloway', holloway), ('Alexander Volkanovski', volkanovski)):
                if slot in ('1', '2'):
                    line_number, row = rows[(frame, slot)]
                    if not row['visible'] or row['track_id'] < 0:
                        raise ValueError(f'Frame {frame}, slot {slot}, {name}: no visible track to verify.')
                    fighters[name] = {
                        'status': 'verified', 'line_1based': line_number, 'fighter_id': slot,
                        'track_id': row['track_id'], 'segment_id': f'r{round_number}_{first}_{last}',
                    }
                else:
                    assert slot in ('absent', 'wrong_person', 'duplicate', 'unresolved')
                    fighters[name] = {'status': slot}
            assert holloway != volkanovski or holloway not in ('1', '2')
            decision = {'frame_index': frame, 'fighters': fighters, 'overlay_sheet': sheet}
            if frame in decisions and decisions[frame] != decision:
                raise ValueError(f'Existing decision at frame {frame} differs. Review the conflict before replacing it.')
            decisions[frame] = decision
    saved['frames'] = [decisions[frame] for frame in sorted(decisions)]
    save(path, saved)
    build(round_number)
    report = read(destination / 'geometry_report.json')
    completed = sum(action['identity_audit_complete'] for action in report['results'])
    print(f'Audited frames: {len(decisions)}/{len(required)}. Completed windows: {completed}/{report["actions"]}.')
    if completed < report['actions']:
        for action, result in zip(inventory['actions'], report['results'], strict=True):
            if not result['identity_audit_complete']:
                first, last = action['review_window']
                frame = next(f for f in range(first, last + 1) if f not in decisions)
                sheet = manifest['sheets'][frame // 16]['file']
                print(f'Next pending frame: {frame} ({sheet}, window {first}-{last}).')
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=range(1, 6), required=True)
    parser.add_argument('span', nargs='+', help='first:last:Holloway slot or status:Volkanovski slot or status')
    args = parser.parse_args()
    spans = []
    for item in args.span:
        start, end, holloway, volkanovski = item.split(':')
        spans.append((int(start), int(end), holloway, volkanovski))
    commit(args.round, spans)
