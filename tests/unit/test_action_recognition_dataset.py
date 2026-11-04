from __future__ import annotations

import csv
import hashlib
import json

import pytest
import yaml

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES, canonical_fighter_id, intervals_overlap
from ufc_tracker.action_recognition.dataset import build_action_corpus


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _source(round_id, fight_id, label="strike"):
    return [{
        "window_id": f"{round_id}-window", "video_id": round_id, "video_path": f"videos/{round_id}.mp4",
        "pose_path": f"poses/{round_id}/pose.jsonl", "fps": 30.0, "start_frame": 0,
        "end_frame": 23, "frame_count": 24, "label": label,
    }]


def _setup_round(root, round_id, fight_id, *, actions, correction=False):
    source_path = root / "binary" / f"{round_id}.jsonl"
    source_rows = _source(round_id, fight_id, "strike")
    _write_jsonl(source_path, source_rows)
    (root / "videos").mkdir(exist_ok=True)
    (root / "videos" / f"{round_id}.mp4").touch()
    pose_path = root / "poses" / round_id / "pose.jsonl"
    _write_jsonl(pose_path, [])
    payload = {
        "schema_version": "strike_actions_v1", "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "video_id": round_id, "video_path": f"videos/{round_id}.mp4", "fps": 30.0,
        "actions": actions, "completed": ["f0-23"], "source_corrections": [],
    }
    if correction:
        payload["source_corrections"] = [{"correction_id": f"c-{round_id}", "event_id": "f0-23", "start_frame": 0, "end_frame": 23, "label": "no_strike"}]
    (root / "actions").mkdir(exist_ok=True)
    (root / "actions" / f"{round_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_contract_normalizes_ids_and_uses_inclusive_overlap():
    assert canonical_fighter_id("fighter_left") == "1"
    assert canonical_fighter_id("fighter_right") == "2"
    assert canonical_fighter_id("referee") is None
    assert intervals_overlap(0, 5, 5, 7)
    assert not intervals_overlap(0, 4, 5, 7)


def test_corpus_masks_ambiguous_preserves_overlapping_positive_labels_and_uses_corrections(tmp_path):
    fights = []
    labels = list(ACTION_CLASSES[:5])
    for index, label in enumerate(labels + ["punch_head", "kick_leg", "punch_body"]):
        round_id, fight_id = f"round{index}", f"fight{index}"
        fights.append({"relative_path": f"videos/{round_id}.mp4", "fight_id": fight_id, "category": "men"})
        actions = [{"action_id": f"a-{index}", "event_id": "f0-23", "start_frame": 0, "end_frame": 23, "attacker": "1", "label": label}]
        if index == 0:
            actions.append({"action_id": "combo", "event_id": "f0-23", "start_frame": 12, "end_frame": 23, "attacker": "1", "label": "punch_body"})
            actions.append({"action_id": "amb", "event_id": "f0-23", "start_frame": 0, "end_frame": 23, "attacker": "unknown", "label": "ambiguous"})
        _setup_round(
            tmp_path, round_id, fight_id,
            actions=[] if index == 7 else actions,
            correction=(index == 7),
        )
    with (tmp_path / "splits.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "fight_id", "category"])
        writer.writeheader(); writer.writerows(fights)
    config = {
        "schema_version": "strikevision_action_recognition_v1", "action_annotations_dir": "actions",
        "binary_annotations_dir": "binary", "splits_manifest": "splits.csv", "classes": list(ACTION_CLASSES),
        "positive_interval_policy": "exhaustive_unless_uncertain_overlap",
    }
    config_path = tmp_path / "config.yml"; config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    samples, audit, report = build_action_corpus(tmp_path, config_path)
    first = next(row for row in samples if row["sample_id"] == "action:a-0")
    assert first["target_labels"] == ["punch_head", "punch_body"]
    assert first["target_mask"] == first["target"]
    complete = next(row for row in samples if row["sample_id"] == "action:a-1")
    assert complete["target_mask"] == [1] * len(ACTION_CLASSES)
    assert complete["action_supervision"] == "complete"
    assert report["fully_supervised_action_samples"] > 0
    assert report["positive_only_action_samples"] > 0
    assert report["excluded_ambiguous"] == 1
    assert report["source_corrections"] == 1
    assert not any("ambiguous" in row["target_labels"] for row in samples)
    assert any(row["sample_type"] == "confirmed_no_strike" for row in samples)
    assert len(report["test_fights"]) == 2
    assert {row["split"] for row in samples} == {"development", "test"}
    assert any(row["record_type"] == "source_correction" for row in audit)


def test_corpus_rejects_stale_action_source(tmp_path):
    _setup_round(tmp_path, "round", "fight", actions=[])
    (tmp_path / "splits.csv").write_text("relative_path,fight_id,category\nvideos/round.mp4,fight,men\n", encoding="utf-8")
    config = {"schema_version": "strikevision_action_recognition_v1", "action_annotations_dir": "actions", "binary_annotations_dir": "binary", "splits_manifest": "splits.csv", "classes": list(ACTION_CLASSES)}
    config_path = tmp_path / "config.yml"; config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "binary" / "round.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        build_action_corpus(tmp_path, config_path)


def test_corpus_combines_annotation_directories_and_preserves_fixed_splits(tmp_path):
    fights = []
    for index in range(9):
        round_id, fight_id = f"round{index}", f"fight{index}"
        fights.append({"relative_path": f"videos/{round_id}.mp4", "fight_id": fight_id, "category": "mixed"})
        label = ACTION_CLASSES[index % 5]
        _setup_round(tmp_path, round_id, fight_id, actions=[{
            "action_id": f"a-{index}", "event_id": "f0-23", "start_frame": 0, "end_frame": 23,
            "attacker": "1", "label": label,
        }])
    (tmp_path / "actions_new").mkdir()
    (tmp_path / "actions" / "round8.json").replace(tmp_path / "actions_new" / "round8.json")
    with (tmp_path / "splits.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "fight_id", "category"])
        writer.writeheader(); writer.writerows(fights)
    config = {
        "schema_version": "strikevision_action_recognition_v1",
        "action_annotations_dirs": ["actions", "actions_new"],
        "binary_annotations_dir": "binary", "splits_manifest": "splits.csv",
        "classes": list(ACTION_CLASSES),
        "positive_interval_policy": "exhaustive_unless_uncertain_overlap",
        "fixed_splits": {
            "test_fights": ["fight0", "fight1"],
            "development_folds": {
                "fight2": 0, "fight3": 0, "fight4": 1,
                "fight5": 1, "fight6": 2, "fight7": 2,
            },
            "train_only_fights": ["fight8"],
        },
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    samples, _, report = build_action_corpus(tmp_path, config_path)
    added = [row for row in samples if row["fight_id"] == "fight8"]
    assert added
    assert {row["split"] for row in added} == {"development"}
    assert {row["development_fold"] for row in added} == {None}
    assert report["train_only_fights"] == ["fight8"]
    assert report["test_fights"] == ["fight0", "fight1"]
