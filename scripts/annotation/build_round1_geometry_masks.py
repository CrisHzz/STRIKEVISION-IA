"""Build conservative geometry QA masks from audited identity references.

This is an automatic quality audit, not manual joint approval or a v4 corpus.
Historical poses and human action labels are never changed.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from ufc_tracker.pose.estimator import KEYPOINT_NAMES

BASE = ROOT / 'output/annotation_audits/volkanovski_holloway_i'
STEM = 'alexander_volkanovski_vs_max_holloway_i_ufc_245__t_5G7bGTPxs__round1'
H, V = 'Max Holloway', 'Alexander Volkanovski'

def build():
    annotation_path = ROOT / 'data/annotations/action_strikes_direct_v1' / (STEM + '.json')
    pose_path = ROOT / 'data/processed/poses/fights_remaining_mixed' / STEM / 'pose.jsonl'
    original = annotation_path.read_bytes()
    data = json.loads(original)
    pose_bytes = pose_path.read_bytes()
    pose_hash = hashlib.sha256(pose_bytes).hexdigest()
    rows = [json.loads(line) for line in pose_bytes.decode('utf-8').splitlines() if line.strip()]
    indexed = {(r['frame_index'], str(r['fighter_id'])): (i+1, r) for i, r in enumerate(rows)}
    assert len(indexed) == len(rows), 'Duplicate frame/slot'
    links = {}
    for name in ('round1_141_280_pose_links.json', 'round1_289_438_pose_links.json',
                 'round1_vinculos_4566_4596.json', 'round1_vinculos_26_pendientes.json'):
        doc = json.loads((BASE / name).read_text(encoding='utf-8'))
        assert doc['pose_sha256'] == pose_hash, 'Pose source changed'
        for a in doc['actions']:
            links[a['action_id']] = a['frames']
    # First batch only has event-level identity audits. Use stable actor IDs;
    # do not infer the opponent from an actor-only audit.
    quality = json.loads((BASE / 'round1_001_140_tracking_quality.json').read_text(encoding='utf-8'))
    for a in quality['actions']:
        frames = []
        for f in range(a['start_frame'], a['end_frame']+1):
            players = {}
            slot = a.get('observed_slot')
            if slot and a['tracking_status'] in ('identity_stable', 'identity_stable_changed_mapping',
                                                'identity_stable_partial_pose', 'opponent_missing', 'opponent_wrong_person'):
                item = indexed.get((f, str(slot)))
                if item:
                    n, row = item
                    players[a['fighter']] = {'pose_ref': dict(path=str(pose_path), line_1based=n,
                        frame_index=f, fighter_id=str(slot), track_id=row.get('track_id'))}
            frames.append(dict(frame_index=f, fighters=players))
        links[a['action_id']] = frames
    n = len(data['actions'])
    length = max(a['end_frame']-a['start_frame']+1 for a in data['actions'])
    coords = np.zeros((n, length, 2, 17, 2), np.float32)
    mask = np.zeros((n, length, 2, 17), bool)
    confidence = np.zeros(mask.shape, np.float32)
    frame_indices = np.full((n, length), -1, np.int32)
    tracks = np.full((n, length, 2), -1, np.int64)
    reasons = Counter()
    summaries = []
    edges = [(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]
    for ai, action in enumerate(data['actions']):
        assert action['action_id'] in links, 'Missing identity decision'
        by_frame = {r['frame_index']: r['fighters'] for r in links[action['action_id']]}
        actor = action['fighter']
        for ti, f in enumerate(range(action['start_frame'], action['end_frame']+1)):
            frame_indices[ai, ti] = f
            for pi, person in enumerate((actor, H if actor == V else V)):
                ref = by_frame.get(f, {}).get(person, {}).get('pose_ref')
                if not ref:
                    reasons['identity_unavailable'] += 1
                    continue
                row = rows[ref['line_1based']-1]
                assert (row['frame_index'], str(row['fighter_id']), row.get('track_id')) == (f, str(ref['fighter_id']), ref.get('track_id'))
                # Prior batch used a track-ID whitelist across camera cuts.
                # Reject known mixed-person spans; no global track-ID identity inference.
                if 7174 <= f <= 7196 or 7680 <= f <= 7711 or (7803 <= f <= 7808) or (8533 <= f <= 8546 and person == H) or (8874 <= f <= 8890 and person == V) or 8946 <= f <= 8977:
                    reasons['mixed_identity_span'] += 1
                    continue
                if not row.get('pose_valid') or not row.get('visible'):
                    reasons['pose_invalid'] += 1
                    continue
                box = np.asarray(row.get('bbox_xyxy', []), float)
                if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                    reasons['bbox_invalid'] += 1
                    continue
                height = box[3]-box[1]
                tracks[ai,ti,pi] = row.get('track_id', -1)
                for ji, joint in enumerate(KEYPOINT_NAMES):
                    p = row.get('keypoints', {}).get(joint)
                    if p is None or len(p) != 3 or not np.isfinite(p).all() or not .5 <= p[2] <= 1:
                        reasons['missing_or_low_confidence_joint'] += 1
                        continue
                    x,y,c = p
                    if not (0 <= x < 1920 and 0 <= y < 1080 and box[0]-.15*height <= x <= box[2]+.15*height and box[1]-.15*height <= y <= box[3]+.15*height):
                        reasons['joint_outside_geometry'] += 1
                        continue
                    coords[ai,ti,pi,ji] = (x,y)
                    confidence[ai,ti,pi,ji] = c
                    mask[ai,ti,pi,ji] = True
                for j,k in edges:
                    if mask[ai,ti,pi,j] and mask[ai,ti,pi,k] and np.linalg.norm(coords[ai,ti,pi,j]-coords[ai,ti,pi,k]) > .65*height:
                        mask[ai,ti,pi,[j,k]] = False
                        reasons['implausible_segment'] += 1
                if ti and tracks[ai,ti-1,pi] == tracks[ai,ti,pi]:
                    bad = mask[ai,ti,pi] & mask[ai,ti-1,pi] & (np.linalg.norm(coords[ai,ti,pi]-coords[ai,ti-1,pi],axis=-1) > .3*height)
                    mask[ai,ti,pi,bad] = False
                    reasons['temporal_jump'] += int(bad.sum())
        duration = action['end_frame']-action['start_frame']+1
        coverage = float(mask[ai,:duration,0].mean())
        summaries.append(dict(action_id=action['action_id'], label=action['label'], fighter=actor,
            actor_joint_coverage=coverage, candidate_for_manual_validation=coverage >= .65 and action['label'] != 'ambiguous',
            training_approved=False))
    coords[~mask] = 0
    confidence[~mask] = 0
    velocity_mask = np.zeros_like(mask)
    velocity_mask[:,1:] = mask[:,1:] & mask[:,:-1] & (tracks[:,1:, :,None] == tracks[:,:-1,:,None]) & (tracks[:,1:,:,None] >= 0)
    target_mask = np.asarray([a['label'] != 'ambiguous' for a in data['actions']])
    out = BASE / 'round1_geometry_masks'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'geometry_masks.npz', coordinates_px=coords, confidence=confidence,
        point_mask=mask, frame_mask=mask.any(axis=-1), velocity_mask=velocity_mask,
        frame_indices=frame_indices, track_ids=tracks, target_mask=target_mask,
        training_mask=np.zeros(n,bool), action_ids=np.asarray([a['action_id'] for a in data['actions']]))
    assert not mask[frame_indices < 0].any()
    assert not coords[~mask].any()
    assert not target_mask[np.asarray([a['label']=='ambiguous' for a in data['actions']])].any()
    assert annotation_path.read_bytes() == original
    report = dict(schema='direct_actions_geometry_qa_v1', source_revision=data['revision'],
        pose_sha256=pose_hash, annotations_sha256=hashlib.sha256(original).hexdigest(),
        actions=n, candidate_actions=sum(s['candidate_for_manual_validation'] for s in summaries),
        training_ready=False, validation='automatic_geometry_checks_only',
        joint_names=KEYPOINT_NAMES, fighter_axis='actor_then_opponent', coordinate_system='original_video_pixels',
        context='action_frames_only; no interpolation or inferred negative labels',
        thresholds=dict(confidence=.5, bbox_margin_body_height=.15, max_segment_body_height=.65, max_jump_body_height=.3),
        exclusions=dict(reasons), results=summaries)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README.md').write_text('# Round 1: geometria y mascaras\n\nAuditoria automatica de las 155 acciones. Identidad ausente o dudosa, poses invalidas y articulaciones rechazadas tienen mascara falsa y coordenadas cero.\n\nNPZ: coordenadas en pixeles, confianza, point_mask, frame_mask, velocity_mask, frame_indices, track_ids, action_ids, target_mask y training_mask. Eje de peleadores: atacante, rival. Padding: frame -1 y mascara falsa. target_mask solo indica etiqueta no ambigua; training_mask permanece falso.\n\nLos umbrales son conservadores y no calibrados. No equivale a revision visual de articulaciones. No es el cache v4: falta el adaptador temporal con contexto auditado y validacion manual. No alimentar v4 directamente con este NPZ. Los JSON de acciones y poses originales permanecen intactos.\n',encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('actions','candidate_actions','training_ready','exclusions')}))

if __name__ == '__main__':
    build()
