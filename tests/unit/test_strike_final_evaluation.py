import numpy as np
import pytest
import torch

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.strike import final_evaluation as final
from ufc_tracker.strike.cache import atomic_json, read_json


@pytest.fixture
def candidate(tmp_path):
    assignments = {"train": ["a"], "validation": ["b"], "test": ["c", "d"]}
    ctx = {"identity": {"schema": "synthetic"}, "split": {"assignments": assignments}}
    head = torch.nn.Linear(2048, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    torch.save({"status": "validation_selected_test_not_evaluated",
                "extractor": ctx["identity"], "identity": {"split": ctx["split"]},
                "head": head.state_dict(), "class_map": {"no_strike": 0, "strike": 1},
                "feature_dim": 2048, "threshold": .33, "selected_loss": "class_weighted",
                "selected_epoch": 8}, tmp_path / "candidate.pt")
    atomic_json(tmp_path / "validation.json", {"test_evaluated": False})
    candidate_hash = sha256(tmp_path / "candidate.pt")
    atomic_json(tmp_path / "READY.json", {"candidate_sha256": candidate_hash,
                "validation_sha256": sha256(tmp_path / "validation.json")})
    atomic_json(tmp_path / "reload_check.json", {"head_reload_ok": True,
                "candidate_sha256": candidate_hash})
    return tmp_path, ctx


def test_final_metrics_keep_threshold_and_refuse_repeat(candidate, monkeypatch):
    run_dir, ctx = candidate
    rows = [{"window_id": f"w{i}", "video_id": fight, "fight_id": fight,
             "category": "synthetic", "start_frame": 0, "end_frame": 23,
             "fps": 30., "start_seconds": 0., "end_seconds": 23 / 30,
             "duration_seconds": .8, "label": label}
            for i, (fight, label) in enumerate([("c", "strike"), ("d", "no_strike")])]
    monkeypatch.setattr(final, "load_test_features", lambda root, context: (
        np.zeros((2, 2048), dtype=np.float32), rows, {"scope": "synthetic"}))
    before = sha256(run_dir / "candidate.pt")
    final.evaluate_test(run_dir, ctx, run_dir)
    report = read_json(run_dir / "test.json")
    assert report["threshold_fixed_from_validation"] == .33
    assert report["selected_epoch_fixed_from_validation"] == 8
    assert report["test"]["model"]["confusion_matrix_true_rows_pred_columns"] == [[0, 1], [0, 1]]
    assert sha256(run_dir / "candidate.pt") == before
    with pytest.raises(ValueError, match="already evaluated"):
        final.evaluate_test(run_dir, ctx, run_dir)


def test_unverified_candidate_refused(candidate):
    run_dir, ctx = candidate
    atomic_json(run_dir / "reload_check.json", {"head_reload_ok": False})
    with pytest.raises(ValueError, match="clean process"):
        final._load_candidate(run_dir, ctx)
