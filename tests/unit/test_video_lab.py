from pathlib import Path

import cv2
import numpy as np
import pytest

import ufc_tracker.services.video_lab as video_lab
from ufc_tracker.services.video_lab import STAGE_INSPECT, STAGE_VISUAL_V0, run_video_lab


def test_video_lab_rejects_missing_file() -> None:
    with pytest.raises(ValueError, match="archivo de video válido"):
        run_video_lab(Path("missing.mp4"), STAGE_INSPECT)


def test_video_lab_rejects_unknown_stage(tmp_path: Path) -> None:
    video = tmp_path / "round.mp4"
    video.touch()

    with pytest.raises(ValueError, match="Etapa no soportada"):
        run_video_lab(video, "Unknown")


def test_visual_v0_creates_a_playable_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "round.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(video_lab, "JOBS_ROOT", tmp_path / "jobs")
    result = run_video_lab(input_path, STAGE_VISUAL_V0)

    assert result.output_path is not None
    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0
    assert result.metadata.frame_count == 5
