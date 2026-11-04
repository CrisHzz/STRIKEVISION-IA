from __future__ import annotations

import numpy as np
import torch

from ufc_tracker.action_recognition.metrics import multilabel_metrics, optimize_thresholds
from ufc_tracker.action_recognition.models import AttentivePoseTCN, CompactSTGCN, PoseTCN, masked_bce_with_logits
from ufc_tracker.action_recognition.training import _binary_threshold, _evaluate_hierarchical, _hierarchical_loss


def test_masked_metrics_exclude_unknown_and_tune_only_known_labels():
    targets = np.zeros((4, 6)); probabilities = np.zeros((4, 6)); masks = np.zeros((4, 6), dtype=bool)
    targets[:2, 0] = [1, 0]; probabilities[:2, 0] = [.9, .1]; masks[:2, 0] = True
    thresholds = optimize_thresholds(targets, probabilities, masks)
    report = multilabel_metrics(targets, probabilities, masks, thresholds)
    assert report["classes"]["punch_head"]["f1"] == 1
    assert report["classes"]["kick_leg"]["support"] == 0


def test_pose_models_and_masked_loss_accept_sequence_features():
    features = torch.zeros((3, 8, 206))
    features[:, :, -2] = 1  # attacker frame availability in the shared feature layout
    mask = torch.ones((3, 8), dtype=torch.bool)
    for model in (PoseTCN(206, 6, channels=16, blocks=1), CompactSTGCN(206, 6, channels=16, blocks=1)):
        output = model(features, mask)
        assert output.shape == (3, 6)
    loss = masked_bce_with_logits(torch.zeros((2, 6)), torch.ones((2, 6)), torch.zeros((2, 6), dtype=torch.bool))
    assert loss.item() == 0


def test_attentive_tcn_handles_partial_and_empty_pose_masks():
    features = torch.randn((2, 8, 206))
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    output = AttentivePoseTCN(206, 7, channels=16, blocks=1)(features, mask)
    assert output.shape == (2, 7)
    assert torch.isfinite(output).all()


def test_hierarchical_loss_and_action_threshold_separate_action_from_technique():
    logits = torch.zeros((2, 7), requires_grad=True)
    targets = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    masks = torch.ones_like(targets, dtype=torch.bool)
    loss = _hierarchical_loss(logits, targets, masks, torch.ones(6), torch.tensor(1.0))
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None
    threshold, metrics = _binary_threshold(np.array([1, 0]), np.array([0.9, 0.1]))
    assert 0.1 < threshold <= 0.9
    assert metrics["f1"] == 1


def test_hierarchical_evaluation_scores_techniques_after_action_gate():
    targets = np.zeros((4, 6), dtype=np.float32)
    targets[:2, 0] = 1
    masks = np.ones_like(targets, dtype=bool)
    probabilities = np.zeros_like(targets)
    probabilities[:, 0] = [0.9, 0.8, 0.9, 0.8]
    action_probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    _, threshold, _, gated, ungated = _evaluate_hierarchical(
        targets, probabilities, masks, action_probabilities,
    )
    assert 0.2 < threshold <= 0.8
    assert gated["classes"]["punch_head"]["f1"] == 1
    assert gated["primary_macro_f1"] >= ungated["primary_macro_f1"]
