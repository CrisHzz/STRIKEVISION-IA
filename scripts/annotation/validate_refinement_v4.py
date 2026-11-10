"""Exercise v4 on reviewed identity windows without publishing training samples."""

import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml

from prepare_round2_refinement import ROOT, read, round_paths, save, sha

sys.path.insert(0, str(ROOT / 'src'))
from ufc_tracker.action_recognition.features import PoseFeatureSpec, extract_pose_sequence
from ufc_tracker.action_recognition.models import build_torch_model
from ufc_tracker.pose.estimator import KEYPOINT_NAMES


def masked_records(frame_indices, references, coordinates, confidence, rows):
    """Map the attacker and opponent to v4 IDs using only retained joints."""
    records = []
    for offset, frame in enumerate(frame_indices):
        for slot, line_number in enumerate(references[offset]):
            if line_number < 0:
                assert not confidence[offset, slot].any()
                continue
            row = rows[line_number]
            assert row['frame_index'] == frame
            joints = {
                name: [*coordinates[offset, slot, joint].tolist(), float(score)]
                for joint, (name, score) in enumerate(zip(KEYPOINT_NAMES, confidence[offset, slot], strict=True))
                if score > 0
            }
            records.append({
                **row, 'fighter_id': str(slot + 1), 'keypoints': joints,
                'pose_valid': bool(joints),
            })
    return records


def validate(round_number):
    pose_dir, _, destination, annotations = round_paths(round_number)
    pose_file = pose_dir / 'pose.jsonl'
    inventory = read(destination / 'overlay_inventory.json')
    report = read(destination / 'geometry_report.json')
    audit_path = destination / 'identity_decisions.json'
    mask_path = destination / 'geometry_masks.npz'
    assert sha(annotations) == inventory['annotations_sha256']
    assert sha(pose_file) == report['pose_sha256'] == inventory['pose_sha256']
    assert sha(audit_path) == report['identity_decisions_sha256']
    assert report['joint_names'] == list(KEYPOINT_NAMES)
    exclusion_path = destination / 'joint_exclusions.json'
    visual_exclusions = []
    if exclusion_path.is_file():
        assert sha(exclusion_path) == report['joint_exclusions_sha256']
        visual_exclusions = read(exclusion_path)['exclusions']
    decisions = {item['frame_index']: item['fighters'] for item in read(audit_path)['frames']}
    with pose_file.open(encoding='utf-8') as stream:
        rows = {number: json.loads(line) for number, line in enumerate(stream, 1) if line.strip()}
    config = yaml.safe_load((ROOT / 'configs/models/action_training_v4.yaml').read_text(encoding='utf-8'))
    spec = PoseFeatureSpec(**config['feature_extraction'])
    classes = yaml.safe_load((ROOT / config['dataset_config']).read_text(encoding='utf-8'))['classes']
    dispositions = Counter()
    results_by_class = defaultdict(Counter)
    details = []
    features, frame_masks = [], []
    with np.load(mask_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
        assert arrays['action_ids'].tolist() == [action['action_id'] for action in inventory['actions']]
        assert np.array_equal(arrays['point_mask'], arrays['confidence'] > 0)
        assert np.array_equal(arrays['frame_mask'], arrays['point_mask'].any(axis=-1))
        for index, (action, result) in enumerate(zip(inventory['actions'], report['results'], strict=True)):
            first, last = action['review_window']
            frames = arrays['frame_indices'][index]
            assert frames[frames >= 0].tolist() == list(range(first, last + 1))
            references = arrays['pose_line_1based'][index]
            athlete = action['fighter']
            opponent = next(name for name in inventory['fighter_counts'] if name != athlete)
            for exclusion in visual_exclusions:
                start, end = exclusion['frames']
                slot = (athlete, opponent).index(exclusion['fighter'])
                joint = KEYPOINT_NAMES.index(exclusion['joint'])
                assert not arrays['point_mask'][index, (frames >= start) & (frames <= end), slot, joint].any()
            for offset, frame in enumerate(frames):
                choices = decisions.get(int(frame), {})
                for slot, name in enumerate((athlete, opponent)):
                    choice = choices.get(name, {})
                    line = references[offset, slot]
                    assert line == choice.get('line_1based', -1)
                    if line >= 0:
                        assert str(rows[line]['fighter_id']) == str(choice['fighter_id'])
                        assert rows[line]['track_id'] == choice['track_id']
                    else:
                        assert not arrays['point_mask'][index, offset, slot].any()
            assert result['identity_audit_complete'] == all(frame in decisions for frame in range(first, last + 1))
            assert bool(arrays['target_mask'][index]) == (action['label'] in classes)
            if not result['identity_audit_complete']:
                dispositions['identity_pending'] += 1
                results_by_class[action['label']]['identity_pending'] += 1
                details.append({'action_id': action['action_id'], 'label': action['label'], 'result': 'identity_pending'})
                continue
            if not arrays['target_mask'][index]:
                dispositions['ambiguous'] += 1
                results_by_class[action['label']]['ambiguous'] += 1
                details.append({'action_id': action['action_id'], 'label': action['label'], 'result': 'ambiguous'})
                continue
            records = masked_records(
                frames, references, arrays['coordinates_px'][index], arrays['confidence'][index], rows,
            )
            edges = dict(zip(range(first, last + 1), arrays['velocity_mask'][index, :last - first + 1], strict=True))
            sample = {**action, 'fps': inventory['fps'], 'attacker': '1'}
            sequence = extract_pose_sequence(sample, records, spec, audited_velocity_edges=edges)
            values = sequence['features']
            points = sequence['point_mask']
            velocities = sequence['velocity_mask']
            assert np.isfinite(values).all()
            assert not points[(sequence['frame_indices'] < first) | (sequence['frame_indices'] > last)].any()
            assert not velocities[0].any()
            for step, (start, end) in enumerate(zip(sequence['frame_indices'], sequence['frame_indices'][1:]), 1):
                if velocities[step].any():
                    assert start < end
                    assert np.all(velocities[step] <= np.logical_and.reduce([edges[f] for f in range(start + 1, end + 1)]))
            attacker_mask = sequence['frame_mask'][:, 0]
            active = (sequence['frame_indices'] >= action['start_frame']) & (sequence['frame_indices'] <= action['end_frame'])
            disposition = 'model_smoke_pass' if attacker_mask[active].any() else 'no_usable_attacker_during_action'
            dispositions[disposition] += 1
            results_by_class[action['label']][disposition] += 1
            details.append({
                'action_id': action['action_id'], 'label': action['label'], 'result': disposition,
                'attacker_active_steps': int(attacker_mask[active].sum()),
                'attacker_point_coverage': float(sequence['pose_coverage']),
            })
            if disposition == 'model_smoke_pass':
                features.append(values)
                frame_masks.append(attacker_mask)
    if features:
        batch = torch.from_numpy(np.stack(features))
        model = build_torch_model(config['models']['temporal'][0], batch.shape[-1], len(classes))
        model.eval()
        with torch.inference_mode():
            logits = model(batch, torch.from_numpy(np.stack(frame_masks)))
        assert logits.shape == (len(features), len(classes))
        assert torch.isfinite(logits).all()
    save(destination / 'v4_validation.json', {
        'scope': 'Technical check with an untrained v4 model. Visual approval and training publication are separate.',
        'source_revision': inventory['source_revision'], 'masks_sha256': sha(mask_path),
        'identity_decisions_sha256': report['identity_decisions_sha256'],
        'steps': spec.steps, 'context_seconds': spec.context_seconds, 'results': dict(dispositions),
        'results_by_class': {label: dict(results) for label, results in sorted(results_by_class.items())},
        'training_ready': False, 'actions': details,
    })
    print(f'Round {round_number}: {dict(dispositions)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rounds', type=int, choices=range(1, 6), nargs='+')
    for number in parser.parse_args().rounds:
        validate(number)
