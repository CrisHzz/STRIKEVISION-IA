from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from ufc_tracker.strike.dataset import (
    clip_tensor, iter_rgb_windows, letterbox, sample_indices, split_summary, validate_window,
)


def row(start=0, end=23, fps=30.0, **kwargs):
    return {"window_id": f"w{start}", "video_id": "v", "video_path": "source.mkv",
            "fight_id": "fight", "category": "test", "label": "strike", "fps": fps,
            "start_frame": start, "end_frame": end, "frame_count": end - start + 1,
            "start_seconds": start / fps, "end_seconds": end / fps,
            "duration_seconds": (end - start + 1) / fps,
            "sample_span_seconds": (end - start) / fps, **kwargs}


@pytest.mark.parametrize("fps", [24, 30, 60, 30000 / 1001, 60000 / 1001])
def test_inclusive_sampling(fps):
    r = row(12, 35, fps)
    indices = sample_indices(r)
    assert len(indices) == len(set(indices)) == 13
    assert indices[0] == 12 and indices[-1] == 35
    assert np.all(np.diff(indices) > 0)
    assert r["duration_seconds"] == 24 / fps


@pytest.mark.parametrize("changes", [{"label": "unknown_occluded"}, {"fps": float("nan")},
                                       {"frame_count": 23}, {"end_seconds": 0.8},
                                       {"start_frame": 0.5}])
def test_bad_contract_rejected(changes):
    with pytest.raises(ValueError):
        validate_window(row(**changes))


def test_rgb_order_padding_and_shape():
    rgb = np.zeros((20, 40, 3), np.uint8)
    rgb[:, :, 0] = 255
    padded = letterbox(rgb)
    assert np.all(padded[0] == 0)
    assert tuple(padded[91, 91]) == (255, 0, 0)
    tensor = clip_tensor([padded] * 13)
    assert tensor.shape == (3, 13, 182, 182)
    assert tensor[0, 0, 91, 91] == pytest.approx((1 - 0.45) / 0.225)
    assert tensor[2, 0, 91, 91] == pytest.approx(-2.0)


@pytest.mark.parametrize("rate", [Fraction(24), Fraction(30), Fraction(60),
                                  Fraction(30000, 1001), Fraction(60000, 1001)])
def test_decode_exact_indices_overlap_and_fractional_pts(tmp_path, rate):
    with av.open(str(tmp_path / "source.mkv"), "w") as output:
        stream = output.add_stream("ffv1", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "bgr0"
        for index in range(36):
            rgb = np.zeros((32, 32, 3), np.uint8)
            rgb[:, :, 0] = index * 5
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    rows = [row(0, 23, float(rate), pose_usable=False), row(12, 35, float(rate))]
    clips = list(iter_rgb_windows(tmp_path, rows))
    assert len(clips) == 2
    for clip in clips:
        reds = (clip["tensor"][0, :, 91, 91] * 0.225 + 0.45) * 255
        assert torch.allclose(reds, torch.tensor(clip["sampled_indices"]) * 5., atol=1e-4)
        assert clip["buffer_frames"] <= 24
        assert np.allclose(clip["sampled_pts_seconds"],
                           np.array(clip["sampled_indices"]) / float(rate), atol=0.001)
    with pytest.raises(ValueError, match="EOF"):
        list(iter_rgb_windows(tmp_path, [row(24, 47, float(rate))]))
    with pytest.raises(ValueError, match="FPS"):
        list(iter_rgb_windows(tmp_path, [row(fps=25)]))


def test_fight_leakage_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        split_summary([row()], {"train": ["fight"], "validation": ["fight"], "test": []})


def test_variable_timestamps_are_explicitly_mapped(tmp_path):
    with av.open(str(tmp_path / "source.mkv"), "w") as output:
        stream = output.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "bgr0"
        for index in range(24):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            frame.pts = index if index < 12 else index + 3
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    clips = list(iter_rgb_windows(tmp_path, [row()]))
    assert clips[0]["timestamp_mapping"] == "decoded_frame_index_to_pts"
    assert clips[0]["timestamp_discrepancy_count"] > 0
    assert clips[0]["max_abs_timestamp_drift_seconds"] > .001
    assert clips[0]["sampled_indices"][-1] == 23
    assert clips[0]["sampled_pts_seconds"][-1] > 23 / 30
