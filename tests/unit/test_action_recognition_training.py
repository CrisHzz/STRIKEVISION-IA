from __future__ import annotations

import numpy as np

from ufc_tracker.action_recognition.training import select_pose_samples


def _sample(fight, kind, index, fold=0):
    return {
        "sample_id": f"{fight}-{kind}-{index}", "fight_id": fight, "sample_type": kind,
        "pose_usable": True, "split": "development", "development_fold": fold,
    }


def test_pose_selection_keeps_all_positive_and_caps_negatives_per_fight():
    samples = [_sample("a", "positive_action", index) for index in range(3)]
    samples += [_sample("a", "confirmed_no_strike", index) for index in range(20)]
    samples += [_sample("b", "positive_action", index, fold=1) for index in range(2)]
    samples += [_sample("b", "confirmed_no_strike", index, fold=1) for index in range(20)]
    train = select_pose_samples(samples, split="train", validation_fold=1, negative_ratio=2, seed=1)
    assert sum(row["sample_type"] == "positive_action" for row in train) == 3
    assert sum(row["sample_type"] == "confirmed_no_strike" for row in train) == 6


def test_pose_selection_excludes_untrusted_pose_and_requires_fold_for_development():
    samples = [_sample("a", "positive_action", 0), _sample("a", "confirmed_no_strike", 0)]
    samples[0]["pose_usable"] = False
    try:
        select_pose_samples(samples, split="train", validation_fold=None, negative_ratio=1, seed=1)
    except ValueError as error:
        assert "validation_fold" in str(error)
    else:
        raise AssertionError("Expected a fold requirement")
    try:
        select_pose_samples(samples, split="train", validation_fold=1, negative_ratio=1, seed=1)
    except ValueError as error:
        assert "No positive" in str(error)
    else:
        raise AssertionError("Expected missing positives to fail")


def test_pose_selection_includes_train_only_fights_but_never_validates_on_them():
    samples = [_sample("validation", "positive_action", 0, fold=0)]
    samples += [_sample("training", "positive_action", 0, fold=1)]
    samples += [_sample("added", "positive_action", 0, fold=None)]
    train = select_pose_samples(samples, split="train", validation_fold=0, negative_ratio=0, seed=1)
    validation = select_pose_samples(samples, split="validation", validation_fold=0, negative_ratio=0, seed=1)
    assert {row["fight_id"] for row in train} == {"training", "added"}
    assert {row["fight_id"] for row in validation} == {"validation"}
