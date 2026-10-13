import json

import pytest

from ufc_tracker.annotations.actions import ActionStore, strike_intervals


def rows():
    return [dict(video_id='round1', video_path='round1.mp4', pose_path='pose.jsonl', fps=30,
                 start_frame=start, end_frame=start+23, window_id=f'w{start}', label=label)
            for start,label in [(0,'no_strike'),(12,'strike'),(24,'strike'),(60,'no_strike'),(84,'strike')]]


def store(tmp_path):
    path=tmp_path/'source.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows()), encoding='utf-8')
    return ActionStore(path,tmp_path/'actions.json')


def test_only_positive_windows_are_unioned_without_bridging_negative_gap():
    events=strike_intervals(rows())
    assert [(r['start_frame'],r['end_frame']) for r in events] == [(12,47),(84,107)]
    assert events[0]['window_ids'] == ['w12','w24']


def test_duplicate_click_rejected_and_undo_restores_saved_state(tmp_path):
    s=store(tmp_path)
    original=s.source.read_bytes()
    event=s.events[0]['event_id']
    data,_=s.commit(0,event,label='punch_head',attacker='1',start=12,end=24,complete=True)
    assert len(data['actions']) == 1
    with pytest.raises(ValueError,match='repetida'):
        s.commit(0,event,label='kick_body',start=25,end=47,complete=True)
    reopened=ActionStore(s.source,s.path)
    assert reopened.read()['completed'] == [event]
    undone,_=reopened.commit(1,event,undo=True)
    assert undone['actions'] == [] and undone['completed'] == []
    assert s.source.read_bytes() == original


def test_combinations_and_simultaneous_attackers_are_preserved(tmp_path):
    s=store(tmp_path)
    event=s.events[0]['event_id']
    first,_=s.commit(0,event,label='punch_head',attacker='1',start=12,end=24)
    assert first['completed'] == []
    second,_=s.commit(1,event,label='punch_body',attacker='2',start=12,end=24)
    assert second['completed'] == []
    s.commit(2,event,label='kick_leg',attacker='1',start=25,end=47)
    data,_=s.commit(3,event,complete=True)
    assert len(data['actions']) == 3
    assert data['completed'] == [event]


def test_playhead_range_is_24_frames_clamped_and_respects_manual_bounds():
    import shutil
    import subprocess
    from ufc_tracker.ui.action_annotation_app import ACTION_RANGE_JS

    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the action range regression test')
    harness = r'''
const assert = require('node:assert/strict');
let currentTime = 1.5;
let pauses = 0;
global.document = {querySelector: () => ({currentTime, pause: () => pauses++})};
const choose = CHOOSE;
const state = {index: 0};
assert.deepEqual(choose(state,'0:f100-200','1',100,200), [state,'0:f100-200','1',133,156]);
currentTime = 0;
assert.deepEqual(choose(state,'0:f100-200','1',100,200).slice(3), [100,123]);
currentTime = 99;
assert.deepEqual(choose(state,'0:f100-200','2',100,200).slice(3), [177,200]);
assert.deepEqual(choose(state,'0:f100-200','2',120,130).slice(3), [120,130]);
assert.equal(pauses, 4);
'''
    subprocess.run([node, '-e', harness.replace('CHOOSE', ACTION_RANGE_JS)],
                   check=True, capture_output=True, text=True)


def test_action_store_accepts_its_declared_video_fps(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({**rows()[1], 'fps': 24}) + '\n', encoding='utf-8')
    action_store = ActionStore(source, tmp_path / 'actions.json')
    assert action_store.fps == 24


def test_controller_keeps_event_open_until_finish_and_retains_attacker(tmp_path):
    from ufc_tracker.ui.action_annotation_app import ActionController

    class Controller(ActionController):
        def __init__(self, action_store):
            self.action_store = action_store

        def store(self, source):
            return self.action_store

        def view(self, action_store, index, message='', attacker_value='unknown'):
            data = action_store.read()
            return ({'index': index}, '', 'preview', None, None, attacker_value,
                    message, list(data['actions']))

    action_store = store(tmp_path)
    event = action_store.events[0]['event_id']
    state = {'source': str(action_store.source), 'index': 0, 'revision': 0, 'event_id': event}
    controller = Controller(action_store)

    result = controller.decide(state, f'0:{event}', '2', 12, 24, label='punch_head')
    assert result[0]['index'] == 0
    assert result[2] != 'preview'
    assert result[5] == '2'
    assert len(result[7]) == 1
    assert action_store.read()['completed'] == []

    state['revision'] = 1
    result = controller.decide(state, f'1:{event}', '2', 12, 24, command='next')
    assert result[0]['index'] == 1
    assert result[2] == 'preview'
    assert result[5] == 'unknown'
    assert action_store.read()['completed'] == [event]


def test_action_ui_names_the_fixed_tracking_colors():
    from ufc_tracker.ui.action_annotation_app import GUIDE

    assert 'ID 1 siempre corresponde al tracking rojo' in GUIDE
    assert 'ID 2 al morado' in GUIDE


def test_invalid_bounds_and_changed_source_are_rejected(tmp_path):
    s=store(tmp_path)
    event=s.events[0]['event_id']
    with pytest.raises(ValueError,match='dentro'):
        s.commit(0,event,label='kick_head',start=0,end=23)
    s.commit(0,event,label='ambiguous',start=12,end=47)
    s.source.write_text(s.source.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='cambiaron'):
        ActionStore(s.source,s.path).read()


def test_no_strike_correction_is_traceable_and_undoable(tmp_path):
    s = store(tmp_path)
    event = s.events[0]['event_id']
    data, _ = s.commit(0, event, complete=True, source_correction=True)
    assert data['actions'] == []
    assert data['completed'] == [event]
    correction = data['source_corrections'][0]
    assert correction['label'] == 'no_strike'
    assert correction['reason'] == 'prior_strike_annotation_error'
    assert correction['source_window_ids'] == s.events[0]['window_ids']
    undone, _ = s.commit(1, event, undo=True)
    assert undone['source_corrections'] == []
    assert undone['completed'] == []


def test_no_strike_correction_rejects_event_with_action(tmp_path):
    s = store(tmp_path)
    event = s.events[0]['event_id']
    s.commit(0, event, label='kick_leg', start=12, end=47)
    with pytest.raises(ValueError, match='Deshaz'):
        s.commit(1, event, complete=True, source_correction=True)


def test_keyboard_ignores_repeat_busy_and_text_but_accepts_radio_focus():
    import shutil
    import subprocess
    from ufc_tracker.ui.action_annotation_app import SHORTCUTS_JS

    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the keyboard event regression test')
    harness = r'''
const assert = require('node:assert/strict');
const handlers = {}, clicks = [];
global.window = {addEventListener: (type, fn) => handlers[type] = fn};
global.document = {
  addEventListener: (type, fn) => handlers[type] = fn,
  querySelector: selector => { assert.ok(selector.includes('textarea')); return {value:'0:f12-47'}; },
  getElementById: id => ({matches: () => true, disabled:false, click: () => { clicks.push(id); window.actionBusy=true; }})
};
INSTALL();
function event(code, extra={}) {
  return {code, key:code.slice(6), target:{closest: () => null}, preventDefault(){}, ...extra};
}
for (const key of ['7','9','4','5','6','1','3','0']) {
  const e=event('Numpad'+key);
  window.actionBusy=false;
  handlers.keydown(e);
  handlers.keydown({...e, repeat:true});
  window.actionBusy=false;
  handlers.keydown(e); // held key without keyup is also rejected
  handlers.keyup(e);
}
assert.deepEqual(clicks, ['7','9','4','5','6','1','3'].map(k=>'action-key-'+k).concat('action-next'));
window.actionBusy=true;
handlers.keydown(event('Numpad7')); handlers.keyup(event('Numpad7'));
window.actionBusy=false;
handlers.keydown(event('Numpad7',{target:{closest: () => ({})}})); handlers.keyup(event('Numpad7'));
assert.equal(clicks.length,8);
handlers.keydown(event('Numpad7',{target:{closest: selector => selector.includes(':not([type="radio"])') ? null : ({})}}));
assert.equal(clicks.length,9);
let blocked=0;
handlers.click({target:{closest:()=>true},preventDefault(){blocked++},stopImmediatePropagation(){blocked++}});
assert.equal(blocked,2);
'''
    subprocess.run([node,'-e',harness.replace('INSTALL()',f'({SHORTCUTS_JS})()')],check=True,capture_output=True,text=True)


def test_clip_renderer_preserves_absolute_frames_and_limits_output(tmp_path):
    import cv2
    import numpy as np
    from ufc_tracker.pose.pipeline import render_pose_preview
    source=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,64))
    for index in range(30):
        writer.write(np.full((64,64,3),index*5,dtype=np.uint8))
    writer.release()
    seen=[]
    destination=tmp_path/'clip.mp4'
    render_pose_preview(source,[],destination,start_frame=12,frame_count=24,
                        frame_overlay=lambda frame,index: seen.append((index,float(frame.mean()))))
    assert [index for index,_ in seen] == list(range(12,24))
    assert abs(seen[0][1] - 60) < 8
    capture=cv2.VideoCapture(str(destination))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 12
    assert capture.get(cv2.CAP_PROP_FPS) == 30
    capture.release()
