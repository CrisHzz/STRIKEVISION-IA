from ufc_tracker.action_recognition.features import PoseFeatureSpec
from ufc_tracker.action_recognition.rgb import _sampled_indices


def test_rgb_context_is_resampled_inside_available_video_frames():
    sample = {"fps": 29.97002997002997, "start_frame": 9047, "end_frame": 9047}
    indices = _sampled_indices(sample, PoseFeatureSpec(), max_frame_index=9056)
    assert indices[0] == 9035
    assert indices[-1] == 9056
    assert all(left <= right for left, right in zip(indices, indices[1:]))
