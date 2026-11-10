import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_preparation_reuses_verified_artifacts_and_rejects_changed_inputs(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/ml/run_pose_pipeline.py"
    spec = importlib.util.spec_from_file_location("pose_preparation_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "project_root", lambda *_: tmp_path)
    monkeypatch.setattr(cli, "preparation_key", lambda root, video, options: json.dumps(options, sort_keys=True))
    video = tmp_path / "round.mp4"
    video.write_bytes(b"fixture video")
    config = tmp_path / "config.yaml"
    config.write_text("output_root: poses\ntracking_confidence: 0.5\nmin_track_frames: 15\ngenerate_preview: false\n")
    calls = []

    def prepare(video, output, **options):
        calls.append(options)
        output.mkdir(parents=True)
        paths = {name: output / filename for name, filename in (
            ("tracking_path", "tracking.jsonl"), ("pose_path", "pose.jsonl"),
            ("metrics_path", "pose_metrics.json"), ("metadata_path", "run_metadata.json"),
        )}
        for path in paths.values():
            path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(output_dir=output, preview_path=None, **paths)

    monkeypatch.setattr(cli, "run_pose_pipeline", prepare)
    argv = [str(script), "--config", str(config), "--video", str(video), "--reuse", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    cli.main()
    assert len(calls) == 1
    assert calls[0]["generate_preview"] is False
    monkeypatch.setattr(sys, "argv", argv + ["--max-frames", "10"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", argv)
    (tmp_path / "poses/round/pose.jsonl").write_text("changed")
    with pytest.raises(SystemExit):
        cli.main()
    assert len(calls) == 1
