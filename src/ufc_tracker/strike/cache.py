"""Resumable, provenance-checked frozen features. Test is closed in this stage."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pytorchvideo
import torch
import yaml
from filelock import FileLock

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.strike.dataset import (
    CLASS_MAP, iter_rgb_windows, load_manifest, sample_indices, split_summary,
)
from ufc_tracker.strike.model import build_model


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def context(root: Path) -> dict:
    rows, manifest_hash = load_manifest(root)
    split = read_json(root / "configs/data/strike_split_v1.json")
    if split["status"] != "approved_by_user" or split["manifest_sha256"] != manifest_hash:
        raise ValueError("An approved split for this exact manifest is required")
    if digest(split["assignments"]) != split["assignment_sha256"]:
        raise ValueError("Approved fight assignment hash mismatch")
    summary = split_summary(rows, split["assignments"])
    config = yaml.safe_load((root / "configs/models/strike_x3d_s.yaml").read_text())
    expected_input = {"frames": 13, "size": 182, "sampling": "uniform_round_inclusive",
                      "spatial": "aspect_ratio_letterbox_black", "mean": [0.45] * 3,
                      "std": [0.225] * 3, "timestamp_tolerance_seconds": 0.001}
    if config["input"] != expected_input or config["class_map"] != CLASS_MAP:
        raise ValueError("Input configuration differs from implemented preprocessing")
    weights = root / config["extractor"]["weights_path"]
    if sha256(weights) != config["extractor"]["weights_sha256"]:
        raise ValueError("Pretrained weights changed")
    package = Path(pytorchvideo.__file__).parent
    identity = {"schema": "strike_frozen_features_v1", "manifest_sha256": manifest_hash,
                "split_sha256": split["assignment_sha256"], "input": expected_input,
                "weights_sha256": config["extractor"]["weights_sha256"],
                "feature_dim": 2048, "class_map": CLASS_MAP,
                "versions": {p: importlib.metadata.version(p) for p in
                             ("torch", "torchvision", "pytorchvideo", "av", "numpy",
                              "opencv-python-headless")},
                "pytorchvideo_code": digest({p.relative_to(package).as_posix(): sha256(p)
                                             for p in sorted(package.rglob("*.py"))}),
                "source_code": {p: sha256(Path(__file__).parent / p)
                                for p in ("dataset.py", "model.py", "cache.py")}}
    return {"rows": rows, "split": split, "summary": summary, "model_config": config,
            "identity": identity, "key": digest(identity)}


def groups_for(ctx: dict, split: str) -> dict:
    if split not in ("train", "validation"):
        raise ValueError("Test remains closed; this command supports train/validation only")
    groups = defaultdict(list)
    for row in ctx["rows"]:
        if row["fight_id"] in ctx["split"]["assignments"][split]:
            groups[row["video_id"]].append(row)
    return {video: sorted(rows, key=lambda r: (r["end_frame"], r["start_frame"]))
            for video, rows in sorted(groups.items())}


def cache_path(root: Path, ctx: dict) -> Path:
    return root / "data/processed/features/strike_x3d_s" / ctx["key"]


def _compatible_identity(old: dict, new: dict) -> bool:
    """Features remain compatible when only audited implementation hashes changed."""
    old_core = {k: v for k, v in old.items() if k != "source_code"}
    new_core = {k: v for k, v in new.items() if k != "source_code"}
    return old_core == new_core


def migrate_compatible_cache(root: Path, ctx: dict, progress=print) -> list[dict]:
    """Copy verified immutable shards to the new code-keyed cache.

    Every semantic contract, package version, video hash, window ID, label,
    sampled index, PTS and shard hash must validate. Only audited local source
    hashes may differ.
    """
    base = root / "data/processed/features/strike_x3d_s"
    target = cache_path(root, ctx)
    migrated = []
    if not base.exists():
        return migrated
    candidates = []
    for directory in sorted(base.iterdir()):
        identity_path = directory / "identity.json"
        if directory != target and identity_path.is_file():
            old_identity = read_json(identity_path)
            if (digest(old_identity) == directory.name
                    and _compatible_identity(old_identity, ctx["identity"])):
                candidates.append((directory, old_identity))
    if not candidates:
        return migrated
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "identity.json", ctx["identity"])
    for source_root, old_identity in candidates:
        for split in ("train", "validation"):
            for video, rows in groups_for(ctx, split).items():
                old_directory = source_root / split / video
                if not (old_directory / "index.json").is_file():
                    continue
                source_hash = sha256(root / rows[0]["video_path"])
                old_state, completed = video_state(
                    old_directory, rows, source_hash, source_root.name
                )
                destination = target / split / video
                if (destination / "index.json").exists():
                    _, existing = video_state(destination, rows, source_hash, ctx["key"])
                    if existing >= completed:
                        continue
                    raise ValueError(
                        "New cache has less progress than compatible cache; inspect manually"
                    )
                destination.mkdir(parents=True, exist_ok=True)
                new_shards = []
                for shard in old_state["shards"]:
                    source_file = old_directory / shard["file"]
                    temporary = destination / (shard["file"] + ".tmp")
                    shutil.copy2(source_file, temporary)
                    os.replace(temporary, destination / shard["file"])
                    check_shard(destination / shard["file"], shard, rows)
                    new_shards.append(shard)
                new_state = {
                    "contract": {**old_state["contract"], "cache_key": ctx["key"]},
                    "shards": new_shards,
                    "migration": {
                        "from_cache_key": source_root.name,
                        "old_source_code": old_identity["source_code"],
                        "reason": "PTS mapping support; frozen feature math unchanged",
                    },
                }
                atomic_json(destination / "index.json", new_state)
                migrated.append({"split": split, "video_id": video, "windows": completed,
                                 "from_cache_key": source_root.name})
                progress(f"Migrated verified cache {video}: {completed}/{len(rows)} windows")
    if migrated:
        atomic_json(target / "migration.json", {
            "schema": "strike_cache_migration_v1",
            "target_cache_key": ctx["key"],
            "entries": migrated,
            "validation": "All copied shards and source videos verified before reuse",
        })
    return migrated


def check_shard(path: Path, entry: dict, rows: list[dict]) -> np.ndarray:
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Corrupt cache shard: {path}; preserve it and use a new cache")
    expected = rows[entry["start"]:entry["stop"]]
    with np.load(path, allow_pickle=False) as data:
        x = data["embeddings"]
        if x.dtype != np.float32 or x.shape != (len(expected), 2048) or not np.isfinite(x).all():
            raise ValueError("Invalid cached embedding tensor")
        if data["window_ids"].tolist() != [r["window_id"] for r in expected]:
            raise ValueError("Cached feature/label alignment mismatch")
        if data["labels"].tolist() != [CLASS_MAP[r["label"]] for r in expected]:
            raise ValueError("Cached labels differ from human labels")
        if not np.array_equal(data["indices"], np.stack([sample_indices(r) for r in expected])):
            raise ValueError("Cached sampling differs from annotated intervals")
        if data["pts"].shape != (len(expected), 13) or not np.isfinite(data["pts"]).all():
            raise ValueError("Invalid cached timestamps")
        if np.any(np.diff(data["pts"], axis=1) <= 0):
            raise ValueError("Cached PTS mapping must be strictly increasing")
        return x.copy()


def video_state(directory: Path, rows: list[dict], source_hash: str, key: str):
    contract = {"cache_key": key, "video_sha256": source_hash,
                "video_path": rows[0]["video_path"], "windows": len(rows),
                "window_ids_sha256": digest([r["window_id"] for r in rows])}
    path = directory / "index.json"
    state = read_json(path) if path.exists() else {"contract": contract, "shards": []}
    if state["contract"] != contract:
        raise ValueError(f"Source/contract changed: {directory}; cache reuse refused")
    count = 0
    for shard in state["shards"]:
        if shard["start"] != count or not count < shard["stop"] <= len(rows):
            raise ValueError("Noncontiguous cache checkpoint")
        if shard["file"] != f"{count:06d}.npz":
            raise ValueError("Invalid shard name")
        check_shard(directory / shard["file"], shard, rows)
        count = shard["stop"]
    return state, count


def extract_video(root: Path, directory: Path, rows: list[dict], model,
                  key: str, batch_size: int, chunk_size: int, progress=print):
    """Commit shards after each chunk; a restart never recomputes committed features.

    Decoder resumes from frame zero for exact indices, discarding the saved prefix.
    This trades a short decode prefix for avoiding approximate seeks.
    """
    directory.mkdir(parents=True, exist_ok=True)
    source = root / rows[0]["video_path"]
    source_hash = sha256(source)
    state, completed = video_state(directory, rows, source_hash, key)
    if completed == len(rows):
        progress(f"Reused {rows[0]['video_id']}: {completed}/{len(rows)}")
        return
    batch, features, clips = [], [], []
    chunk_started = perf_counter()

    def flush_batch():
        if batch:
            features.append(model.embeddings(torch.stack([c["tensor"] for c in batch])).numpy())
            clips.extend({k: v for k, v in c.items() if k != "tensor"} for c in batch)
            batch.clear()

    def commit():
        nonlocal completed, chunk_started
        flush_batch()
        if not clips:
            return
        if sha256(source) != source_hash:
            raise ValueError("Video changed during extraction; checkpoint not committed")
        name = f"{completed:06d}.npz"
        path = directory / name
        with (directory / (name + ".tmp")).open("wb") as stream:
            np.savez(stream, embeddings=np.concatenate(features).astype(np.float32),
                     window_ids=np.array([c["row"]["window_id"] for c in clips]),
                     labels=np.array([CLASS_MAP[c["row"]["label"]] for c in clips]),
                     indices=np.array([c["sampled_indices"] for c in clips]),
                     pts=np.array([c["sampled_pts_seconds"] for c in clips]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(directory / (name + ".tmp"), path)
        entry = {"file": name, "start": completed, "stop": completed + len(clips),
                 "sha256": sha256(path), "seconds": perf_counter() - chunk_started}
        check_shard(path, entry, rows)
        state["shards"].append(entry)
        atomic_json(directory / "index.json", state)
        completed = entry["stop"]
        progress(f"Saved {rows[0]['video_id']}: {completed}/{len(rows)} windows; "
                 f"chunk {entry['seconds']:.1f}s")
        features.clear()
        clips.clear()
        chunk_started = perf_counter()

    reader = iter_rgb_windows(root, rows[completed:])
    try:
        for clip in reader:
            batch.append(clip)
            if len(batch) == batch_size or len(clips) + len(batch) == chunk_size:
                flush_batch()
            if len(clips) == chunk_size:
                commit()
        commit()
    finally:
        reader.close()


def extract(root: Path, ctx: dict, settings: dict, splits: list[str]):
    if not splits or any(s not in ("train", "validation") for s in splits):
        raise ValueError("Only train/validation extraction is allowed")
    batch, chunk = settings["extraction_batch_size"], settings["cache_chunk_windows"]
    if not 1 <= batch <= chunk or chunk <= 0:
        raise ValueError("Invalid extraction batch/chunk sizes")
    target = cache_path(root, ctx)
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target / "writer.lock"), timeout=0):
        migrate_compatible_cache(root, ctx, progress=lambda s: print(s, flush=True))
        atomic_json(target / "identity.json", ctx["identity"])
        config = ctx["model_config"]["extractor"]
        torch.set_num_threads(settings["threads"])
        torch.use_deterministic_algorithms(True)
        model = build_model(root / config["weights_path"], config["weights_sha256"]).eval()
        for split in splits:
            groups = groups_for(ctx, split)
            for i, (video, rows) in enumerate(groups.items(), 1):
                print(f"{split}: video {i}/{len(groups)}", flush=True)
                extract_video(root, target / split / video, rows, model, ctx["key"], batch,
                              chunk, progress=lambda s: print(s, flush=True))
        if load_manifest(root)[1] != ctx["identity"]["manifest_sha256"]:
            raise ValueError("Source annotations changed during extraction")
    return target


def load_features(root: Path, ctx: dict, split: str):
    target = cache_path(root, ctx)
    if not (target / "identity.json").exists() or read_json(target / "identity.json") != ctx["identity"]:
        raise ValueError("Cache not ready for this extractor; run extract first")
    features, ordered_rows, provenance = [], [], {}
    with FileLock(str(target / "writer.lock"), timeout=0):
        for video, rows in groups_for(ctx, split).items():
            directory = target / split / video
            source_hash = sha256(root / rows[0]["video_path"])
            state, count = video_state(directory, rows, source_hash, ctx["key"])
            if count != len(rows):
                raise ValueError(f"Incomplete {split} cache: {video} {count}/{len(rows)}; resume extract")
            for shard in state["shards"]:
                features.append(check_shard(directory / shard["file"], shard, rows))
            ordered_rows.extend(rows)
            provenance[video] = {"video_sha256": source_hash, "index_sha256": sha256(directory / "index.json")}
    return np.concatenate(features), ordered_rows, provenance
