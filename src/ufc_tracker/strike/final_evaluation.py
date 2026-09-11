"""One-shot manual final evaluation on the previously closed test fights."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.strike.cache import (
    atomic_json, cache_path, check_shard, extract_video, read_json, video_state,
)
from ufc_tracker.strike.dataset import CLASS_MAP
from ufc_tracker.strike.metrics import evaluation
from ufc_tracker.strike.model import build_model
from ufc_tracker.strike.training import probabilities


def test_groups(ctx: dict) -> dict[str, list[dict]]:
    """The only test accessor. It is called after candidate selection, never before."""
    groups: dict[str, list[dict]] = defaultdict(list)
    test_fights = set(ctx["split"]["assignments"]["test"])
    for row in ctx["rows"]:
        if row["fight_id"] in test_fights:
            groups[row["video_id"]].append(row)
    if not groups:
        raise ValueError("Approved test partition is empty")
    return {
        video: sorted(rows, key=lambda item: (item["end_frame"], item["start_frame"]))
        for video, rows in sorted(groups.items())
    }


def _load_candidate(run_dir: Path, ctx: dict) -> tuple[dict, dict]:
    ready = read_json(run_dir / "READY.json")
    candidate_path = run_dir / "candidate.pt"
    if sha256(candidate_path) != ready["candidate_sha256"]:
        raise ValueError("Candidate changed after validation")
    if sha256(run_dir / "validation.json") != ready["validation_sha256"]:
        raise ValueError("Validation report changed")
    reload_check = read_json(run_dir / "reload_check.json")
    if (not reload_check.get("head_reload_ok")
            or reload_check.get("candidate_sha256") != ready["candidate_sha256"]):
        raise ValueError("Verify this candidate in a clean process before opening test")
    artifact = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if artifact.get("status") != "validation_selected_test_not_evaluated":
        raise ValueError("Candidate is not eligible for its first test evaluation")
    if artifact.get("extractor") != ctx["identity"]:
        raise ValueError("Candidate/extractor contract differs from the current audited cache")
    if artifact.get("class_map") != CLASS_MAP or artifact.get("feature_dim") != 2048:
        raise ValueError("Unsupported candidate class or feature contract")
    if artifact["identity"]["split"]["assignments"] != ctx["split"]["assignments"]:
        raise ValueError("Candidate split changed")
    return artifact, ready


def run_final_test(root: Path, ctx: dict, settings: dict, run_name: str) -> Path:
    """Keep the candidate fixed and exclude concurrent writers for the whole run."""
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Invalid run name")
    run_dir = root / "outputs/strike-detection" / run_name
    if not run_dir.is_dir():
        raise ValueError("Training run does not exist")
    with FileLock(str(run_dir / "training.lock"), timeout=0):
        extract_test(root, ctx, settings, run_dir)
        return evaluate_test(root, ctx, run_dir)


def extract_test(root: Path, ctx: dict, settings: dict, run_dir: Path) -> Path:
    """Resume exact frozen-feature extraction for test only after candidate validation."""
    _load_candidate(run_dir, ctx)
    target = cache_path(root, ctx)
    identity_path = target / "identity.json"
    if not identity_path.is_file() or read_json(identity_path) != ctx["identity"]:
        raise ValueError("Train/validation cache contract is unavailable or changed")
    if (run_dir / "test.json").exists():
        raise ValueError("Test was already evaluated; it must not be rerun")
    with FileLock(str(target / "writer.lock"), timeout=0):
        config = ctx["model_config"]["extractor"]
        torch.set_num_threads(settings["threads"])
        torch.use_deterministic_algorithms(True)
        model = build_model(root / config["weights_path"], config["weights_sha256"]).eval()
        groups = test_groups(ctx)
        for number, (video, rows) in enumerate(groups.items(), 1):
            print(f"test: video {number}/{len(groups)}", flush=True)
            extract_video(
                root,
                target / "test" / video,
                rows,
                model,
                ctx["key"],
                settings["extraction_batch_size"],
                settings["cache_chunk_windows"],
                progress=lambda message: print(message, flush=True),
            )
    return target


def load_test_features(root: Path, ctx: dict) -> tuple[np.ndarray, list[dict], dict]:
    """Load only complete, hash-verified test shards in source-video order."""
    target = cache_path(root, ctx)
    features, rows_out, provenance = [], [], {}
    with FileLock(str(target / "writer.lock"), timeout=0):
        for video, rows in test_groups(ctx).items():
            directory = target / "test" / video
            source_hash = sha256(root / rows[0]["video_path"])
            state, completed = video_state(directory, rows, source_hash, ctx["key"])
            if completed != len(rows):
                raise ValueError(
                    f"Incomplete test cache: {video} {completed}/{len(rows)}; resume test extraction"
                )
            for shard in state["shards"]:
                features.append(check_shard(directory / shard["file"], shard, rows))
            rows_out.extend(rows)
            provenance[video] = {
                "video_sha256": source_hash,
                "index_sha256": sha256(directory / "index.json"),
            }
    return np.concatenate(features), rows_out, provenance


def evaluate_test(root: Path, ctx: dict, run_dir: Path) -> Path:
    """Write final metrics once. No tuning, threshold choice, or checkpoint change occurs here."""
    if (run_dir / "test.json").exists():
        raise ValueError("Test was already evaluated; do not tune or rerun it")
    artifact, ready = _load_candidate(run_dir, ctx)
    x, rows, sources = load_test_features(root, ctx)
    head = torch.nn.Linear(2048, 2)
    head.load_state_dict(artifact["head"], strict=True)
    head.eval()
    probability = probabilities(head, torch.from_numpy(x))
    report = {
        "schema": "strike_test_evaluation_v1",
        "candidate_sha256": ready["candidate_sha256"],
        "threshold_fixed_from_validation": artifact["threshold"],
        "selected_loss_fixed_from_validation": artifact["selected_loss"],
        "selected_epoch_fixed_from_validation": artifact["selected_epoch"],
        "test": evaluation(rows, probability, artifact["threshold"]),
        "source_videos": sources,
        "test_fights": ctx["split"]["assignments"]["test"],
        "limits": [
            "Two fights; window metrics are not independent observations.",
            "No threshold, checkpoint, transform or training choice was changed using test.",
            "Strike/no_strike windows do not count individual strikes or identify an attacker.",
        ],
    }
    temp = run_dir / "test_predictions.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        for row, p_strike in zip(rows, probability):
            record = {
                key: row[key]
                for key in (
                    "window_id", "video_id", "fight_id", "category", "start_frame", "end_frame",
                    "fps", "start_seconds", "end_seconds", "duration_seconds", "label",
                )
            }
            record.update(
                p_strike=float(p_strike),
                predicted_label="strike" if p_strike >= artifact["threshold"] else "no_strike",
                threshold=artifact["threshold"],
                model_version=ready["candidate_sha256"],
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, run_dir / "test_predictions.jsonl")
    atomic_json(run_dir / "test.json", report)
    return run_dir / "test.json"
