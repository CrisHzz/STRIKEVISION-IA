import json

import pytest

from ufc_tracker.annotations.direct_actions import DirectActionStore


def test_simultaneous_actions_revisions_and_negative_conflicts(tmp_path):
    source = tmp_path / 'manifest.json'
    source.write_text(json.dumps({'video': 'v.mp4', 'fps': 30, 'decoded_frames': 100}))
    store = DirectActionStore(source, tmp_path / 'actions.json')
    first = store.commit(0, kind='action', start=10, end=20, label='punch_head',
                         fighter='Holloway', track_slot='1', evidence=['sheet1'])
    second = store.commit(first['revision'], kind='action', start=15, end=25,
                          label='kick_leg', fighter='Volkanovski', track_slot='2', evidence=['sheet1'])
    assert len(second['actions']) == 2
    with pytest.raises(ValueError, match='Stale'):
        store.commit(0, kind='review', start=0, end=5, label='no_strike', evidence=['sheet1'])
    with pytest.raises(ValueError, match='overlaps'):
        store.commit(2, kind='review', start=12, end=13, label='no_strike', evidence=['sheet1'])
    third = store.commit(2, kind='review', start=0, end=9, label='no_strike', evidence=['sheet1'])
    assert third['revision'] == 3
    assert DirectActionStore(source, store.destination).read() == third
    source.write_text(json.dumps({'video': 'other', 'fps': 30, 'decoded_frames': 100}))
    with pytest.raises(ValueError, match='manifest changed'):
        store.read()
