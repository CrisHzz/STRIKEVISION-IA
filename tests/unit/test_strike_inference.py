from ufc_tracker.strike.inference import (
    WINDOW_FRAMES,
    frame_predictions,
    sampled_offsets,
    window_starts,
)


def test_window_starts_match_annotated_24_frame_windows_with_12_frame_stride():
    assert window_starts(WINDOW_FRAMES - 1) == []
    assert window_starts(WINDOW_FRAMES) == [0]
    assert window_starts(48) == [0, 12, 24]


def test_sampled_offsets_are_thirteen_distinct_inclusive_positions():
    offsets = sampled_offsets()
    assert offsets.shape == (13,)
    assert offsets[0] == 0
    assert offsets[-1] == WINDOW_FRAMES - 1
    assert len(set(offsets.tolist())) == 13


def test_preview_prefers_stronger_overlapping_window_without_changing_window_results():
    windows = [
        {"start_frame": 0, "end_frame": 23, "p_strike": 0.2, "predicted_label": "no_strike"},
        {"start_frame": 12, "end_frame": 35, "p_strike": 0.8, "predicted_label": "strike"},
    ]
    rendered = frame_predictions(windows, 36)
    assert rendered[0] == windows[0]
    assert rendered[12] == windows[1]
    assert rendered[35] == windows[1]
