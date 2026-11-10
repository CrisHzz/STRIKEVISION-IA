"""Atomic, revision-controlled annotations over original video frame ranges.

Unreviewed frames carry no implied negative label. Fighter names and tracking
slots are separate because a tracker can swap identities after camera cuts.
"""
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ufc_tracker.annotations.actions import LABELS

_LOCK = threading.RLock()


class DirectActionStore:
    def __init__(self, manifest: Path, destination: Path):
        self.manifest = Path(manifest)
        self.destination = Path(destination)
        raw = self.manifest.read_bytes()
        self.source_hash = hashlib.sha256(raw).hexdigest()
        self.source = json.loads(raw)
        self.frame_count = int(self.source['decoded_frames'])
        self.fps = float(self.source['fps'])
        if self.frame_count <= 0 or self.fps <= 0:
            raise ValueError('Invalid source video metadata')

    def read(self):
        if hashlib.sha256(self.manifest.read_bytes()).hexdigest() != self.source_hash:
            raise ValueError('Review manifest changed')
        if not self.destination.exists():
            return {'schema_version': 'direct_strike_actions_v1', 'source_type': 'frame_review_manifest',
                    'source_sha256': self.source_hash, 'video': self.source['video'],
                    'fps': self.fps, 'frame_count': self.frame_count, 'revision': 0,
                    'actions': [], 'reviewed_ranges': [], 'tracking_notes': []}
        data = json.loads(self.destination.read_text(encoding='utf-8'))
        if data['source_sha256'] != self.source_hash:
            raise ValueError('Review manifest changed')
        return data

    def commit(self, revision, *, kind, start, end, label=None, fighter=None,
               track_slot=None, evidence=None, note=''):
        with _LOCK:
            data = self.read()
            if data['revision'] != revision:
                raise ValueError('Stale annotation revision')
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < self.frame_count:
                raise ValueError('Invalid inclusive frame bounds')
            if not evidence:
                raise ValueError('Evidence is required')
            common = {'start_frame': start, 'end_frame': end, 'evidence': evidence,
                      'note': note, 'annotated_at': datetime.now(timezone.utc).isoformat()}
            if kind == 'action':
                if label not in LABELS.values() or not fighter:
                    raise ValueError('Invalid action or fighter')
                if track_slot not in (None, '1', '2', 'unknown'):
                    raise ValueError('Invalid tracking slot')
                if any((a['start_frame'], a['end_frame'], a['label'], a['fighter']) ==
                       (start, end, label, fighter) for a in data['actions']):
                    raise ValueError('Duplicate action')
                if any(r['status'] == 'no_strike' and start <= r['end_frame'] and end >= r['start_frame']
                       for r in data['reviewed_ranges']):
                    raise ValueError('Action overlaps confirmed no_strike')
                data['actions'].append({**common, 'action_id': uuid.uuid4().hex,
                                        'label': label, 'fighter': fighter, 'track_slot': track_slot})
            elif kind == 'review':
                if label not in ('no_strike', 'reviewed', 'ambiguous'):
                    raise ValueError('Invalid review status')
                if label == 'no_strike' and any(start <= a['end_frame'] and end >= a['start_frame']
                                                for a in data['actions']):
                    raise ValueError('no_strike overlaps action')
                data['reviewed_ranges'].append({**common, 'status': label})
            elif kind == 'tracking':
                if not note:
                    raise ValueError('Tracking note is required')
                data['tracking_notes'].append({**common, 'fighter': fighter, 'track_slot': track_slot})
            else:
                raise ValueError('Unknown decision kind')
            data['revision'] += 1
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            temp = self.destination.with_suffix(f'.{uuid.uuid4().hex}.tmp')
            try:
                temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                temp.replace(self.destination)
            finally:
                temp.unlink(missing_ok=True)
            return data
