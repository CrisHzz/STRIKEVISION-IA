from pathlib import Path

import numpy as np
import pytest
import torch

from ufc_tracker.strike.cache import (
    _compatible_identity, extract_video, groups_for, read_json, video_state,
)
from ufc_tracker.strike.final_evaluation import test_groups as reserved_test_groups
from ufc_tracker.strike.dataset import sample_indices
from ufc_tracker.strike.metrics import choose_threshold, evaluation, metrics
from ufc_tracker.strike.training import fit_head, probabilities
from ufc_tracker.annotations.dataset import sha256


def make_rows(n):
    return [{"window_id": f"w{i}", "video_id": "v", "video_path": "source.bin",
             "fight_id": "fight", "category": "test", "label": "strike" if i % 2 else "no_strike",
             "fps": 30., "start_frame": i * 12, "end_frame": i * 12 + 23, "frame_count": 24,
             "start_seconds": i * 12 / 30, "end_seconds": (i * 12 + 23) / 30,
             "duration_seconds": .8, "sample_span_seconds": 23 / 30} for i in range(n)]


class FakeExtractor:
    def __init__(self):
        self.processed = 0

    def embeddings(self, tensor):
        self.processed += len(tensor)
        return tensor.reshape(len(tensor), -1).mean(1, keepdim=True).repeat(1, 2048)


def clips(root, rows):
    for row in rows:
        indices = sample_indices(row).tolist()
        yield {"row": row, "tensor": torch.tensor([float(row["start_frame"])]),
               "sampled_indices": indices, "sampled_pts_seconds": [i / 30 for i in indices]}


def test_extraction_resume_after_interruption_does_not_repeat_features(tmp_path, monkeypatch):
    from ufc_tracker.strike import cache
    (tmp_path / "source.bin").write_bytes(b"immutable video stand-in")
    rows = make_rows(5)
    model = FakeExtractor()
    directory = tmp_path / "cache"

    def interrupted(root, rows):
        for i, clip in enumerate(clips(root, rows)):
            if i == 3:
                raise RuntimeError("simulated crash")
            yield clip

    monkeypatch.setattr(cache, "iter_rgb_windows", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        extract_video(tmp_path, directory, rows, model, "key", 1, 2, progress=lambda s: None)
    assert read_json(directory / "index.json")["shards"][-1]["stop"] == 2
    resumed = FakeExtractor()
    monkeypatch.setattr(cache, "iter_rgb_windows", clips)
    extract_video(tmp_path, directory, rows, resumed, "key", 1, 2, progress=lambda s: None)
    assert resumed.processed == 3
    extract_video(tmp_path, directory, rows, resumed, "key", 1, 2, progress=lambda s: None)
    assert resumed.processed == 3
    state, count = video_state(directory, rows, sha256(tmp_path / "source.bin"), "key")
    assert count == 5
    assert len(state["shards"]) == 3
    (tmp_path / "source.bin").write_bytes(b"changed video")
    with pytest.raises(ValueError, match="changed"):
        extract_video(tmp_path, directory, rows, resumed, "key", 1, 2)


def test_corrupt_cache_refused(tmp_path, monkeypatch):
    from ufc_tracker.strike import cache
    (tmp_path / "source.bin").write_bytes(b"video")
    monkeypatch.setattr(cache, "iter_rgb_windows", clips)
    rows = make_rows(2)
    extract_video(tmp_path, tmp_path / "cache", rows, FakeExtractor(), "key", 1, 2)
    with (tmp_path / "cache/000000.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Corrupt"):
        video_state(tmp_path / "cache", rows, sha256(tmp_path / "source.bin"), "key")


def test_test_partition_closed():
    with pytest.raises(ValueError, match="closed"):
        groups_for({}, "test")


def test_cache_compatibility_allows_only_local_source_hash_changes():
    old = {"schema": "features", "manifest": "same", "input": {"frames": 13},
           "source_code": {"dataset.py": "old"}}
    new = {**old, "source_code": {"dataset.py": "new"}}
    assert _compatible_identity(old, new)
    assert not _compatible_identity(old, {**new, "input": {"frames": 12}})


def test_final_test_accessor_uses_only_approved_test_fights():
    rows = make_rows(4)
    rows[0]["fight_id"] = "train-fight"
    rows[1]["fight_id"] = "validation-fight"
    rows[2]["fight_id"] = "test-fight"
    rows[3]["fight_id"] = "test-fight"
    result = reserved_test_groups({"rows": rows, "split": {"assignments": {"test": ["test-fight"]}}})
    assert list(result) == ["v"]
    assert [row["window_id"] for row in result["v"]] == ["w2", "w3"]


def settings():
    return {"seed": 123, "threads": 1, "training": {
        "selection_metric": "validation_strike_f1", "losses": ["standard", "class_weighted"],
        "learning_rate": .01, "weight_decay": .0001, "batch_size": 8, "max_epochs": 4,
        "patience": 4, "threshold_min": .1, "threshold_max": .9, "threshold_step": .1}}


def data():
    rng = np.random.default_rng(3)
    x = rng.normal(0, .05, (32, 2048)).astype(np.float32)
    y = np.arange(32) % 2
    x[:, 0] = (2 * y - 1) * 4
    return x[:24], y[:24], x[24:], y[24:]


@pytest.mark.parametrize("loss", ["standard", "class_weighted"])
def test_training_resume_matches_uninterrupted(tmp_path, loss):
    x, y, vx, vy = data()
    kwargs = {"settings": settings(), "identity": {"data": "synthetic"}, "loss_name": loss,
              "progress": lambda s: None}
    expected, best, _ = fit_head(x, y, vx, vy, directory=tmp_path / "full", **kwargs)

    def interrupt(epoch):
        raise RuntimeError("stop after committed epoch")

    with pytest.raises(RuntimeError, match="committed"):
        fit_head(x, y, vx, vy, directory=tmp_path / "resumed", interrupt_after_epoch=interrupt, **kwargs)
    actual, resumed_best, history = fit_head(x, y, vx, vy, directory=tmp_path / "resumed", **kwargs)
    assert len(history) == 4
    assert resumed_best["threshold"] == best["threshold"]
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])
    assert np.array_equal(probabilities(actual, torch.from_numpy(vx)), probabilities(expected, torch.from_numpy(vx)))
    changed = settings()
    changed["training"]["learning_rate"] = .1
    with pytest.raises(ValueError, match="changed"):
        fit_head(x, y, vx, vy, changed, tmp_path / "resumed", {"data": "synthetic"}, loss)


def test_metrics_threshold_baselines_and_groups():
    y, p = [0, 0, 1, 1], [.1, .2, .35, .4]
    threshold = choose_threshold(y, p, [.2, .3, .5])
    assert threshold == .3
    report = metrics(y, p, threshold)
    assert report["f1_strike"] == 1
    assert report["pr_auc_average_precision"] == 1
    rows = make_rows(4)
    result = evaluation(rows, [.1, .9, .2, .8], .5)
    assert result["always_no_strike"]["f1_strike"] == 0
    assert result["always_no_strike"]["accuracy"] == .5
    assert result["model"]["confusion_matrix_true_rows_pred_columns"] == [[2, 0], [0, 2]]
    assert result["groups"]["fight"]["fight"]["model"]["f1_strike"] == 1


def test_manual_training_export_and_fresh_process_reload(tmp_path, monkeypatch):
    import subprocess
    import sys
    from ufc_tracker.strike import training
    x, y, vx, vy = data()
    ctx = {"key": "synthetic-key", "split": {"assignments": {"train": ["train-fight"],
           "validation": ["validation-fight"], "test": ["closed-fight"]}}, "identity": {"test": "synthetic"}}

    def features(root, context, split):
        assert split != "test"
        selected = x if split == "train" else vx
        rows = make_rows(len(selected))
        for row in rows:
            row["fight_id"] = f"{split}-fight"
        return selected, rows, {"synthetic": split}

    monkeypatch.setattr(training, "load_features", features)
    output = training.train(tmp_path, ctx, settings(), "synthetic-integration")
    assert read_json(output / "validation.json")["test_evaluated"] is False
    script = Path(__file__).resolve().parents[2] / "scripts/ml/verify_strike_candidate.py"
    result = subprocess.run([sys.executable, "-s", str(script), "--run-dir", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(output / "reload_check.json")["head_reload_ok"]
    assert len((output / "validation_predictions.jsonl").read_text().splitlines()) == len(vx)
