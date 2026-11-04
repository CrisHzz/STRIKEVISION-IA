import numpy as np

from ufc_tracker.action_recognition.inference import ActionInferenceEngine


def _window(fighter: str, start: int, probabilities: list[float], coverage: float = 1.0) -> dict:
    return {
        "fighter_id": fighter,
        "start_frame": start,
        "end_frame": start + 23,
        "stride_frames": 12,
        "pose_coverage": coverage,
        "probabilities": probabilities,
    }


def test_display_events_choose_one_simultaneous_technique_per_fighter():
    windows = [
        _window("1", 0, [0.81, 0.79, 0.1, 0.1, 0.1, 0.1]),
        _window("2", 0, [0.1, 0.1, 0.84, 0.1, 0.1, 0.1]),
    ]
    events, candidates = ActionInferenceEngine._display_events(windows, np.full(6, 0.5), 30)
    assert [(event["fighter_id"], event["label"]) for event in events] == [("1", "punch_head"), ("2", "kick_leg")]
    assert sum(candidate["eligible"] for candidate in candidates if candidate["fighter_id"] == "1") == 2


def test_display_events_reject_low_confidence_and_low_pose_coverage():
    windows = [
        _window("1", 0, [0.99, 0.1, 0.1, 0.1, 0.1, 0.1], coverage=0.5),
        _window("2", 0, [0.49, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ]
    events, _ = ActionInferenceEngine._display_events(windows, np.full(6, 0.05), 30)
    assert events == []


def test_display_events_apply_action_gate_before_technique():
    window = _window("1", 0, [0.99, 0.1, 0.1, 0.1, 0.1, 0.1])
    window["action_probability"] = 0.4
    events, candidates = ActionInferenceEngine._display_events(
        [window], np.full(6, 0.5), 30, action_threshold=0.6,
    )
    assert events == []
    assert candidates[0]["eligible"] is False


def test_display_events_anchor_short_event_at_window_center():
    events, _ = ActionInferenceEngine._display_events(
        [_window("1", 60, [0.8, 0.1, 0.1, 0.1, 0.1, 0.1])], np.full(6, 0.5), 30,
    )
    assert events[0]["start_frame"] == 66
    assert events[0]["end_frame"] == 78


def test_display_events_do_not_overlap_distinct_labels_for_one_fighter():
    windows = [
        _window("1", 0, [0.8, 0.1, 0.1, 0.1, 0.1, 0.1]),
        _window("1", 12, [0.1, 0.8, 0.1, 0.1, 0.1, 0.1]),
    ]
    events, _ = ActionInferenceEngine._display_events(windows, np.full(6, 0.5), 30)
    assert events[0]["end_frame"] < events[1]["start_frame"]
