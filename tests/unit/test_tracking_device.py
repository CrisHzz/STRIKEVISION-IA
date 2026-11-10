import pytest
import torch

from ufc_tracker.tracking.device import resolve_tracking_device


def test_auto_uses_available_cuda_and_explicit_cpu_is_preserved(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_tracking_device() == "0"
    assert resolve_tracking_device("cpu") == "cpu"
    assert resolve_tracking_device("cuda:0") == "0"
    with pytest.raises(ValueError):
        resolve_tracking_device("cuda:1")


def test_cpu_build_falls_back_visibly_but_rejects_explicit_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="CUDA is unavailable"):
        assert resolve_tracking_device() == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_tracking_device("0")


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="tracking_device"):
        resolve_tracking_device("cuda:banana")
