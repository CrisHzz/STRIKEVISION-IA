"""Manual CPU linear-head training with exact epoch resume and validation selection."""
from __future__ import annotations

import json
import importlib.metadata
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from filelock import FileLock

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.strike.cache import atomic_json, digest, load_features
from ufc_tracker.strike.dataset import CLASS_MAP
from ufc_tracker.strike.metrics import choose_threshold, evaluation, metrics


def atomic_torch(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def settings_check(settings):
    t = settings["training"]
    if t["selection_metric"] != "validation_strike_f1":
        raise ValueError("Only validation strike F1 selection is implemented")
    if not t["losses"] or len(set(t["losses"])) != len(t["losses"]) or not set(t["losses"]).issubset({"standard", "class_weighted"}):
        raise ValueError("Unsupported/duplicated loss candidates")
    for name in ("batch_size", "max_epochs", "patience"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError(f"Invalid {name}")
    if not math.isfinite(t["learning_rate"]) or t["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate")
    if not math.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if not 0 < t["threshold_min"] <= t["threshold_max"] < 1 or not 0 < t["threshold_step"] <= 1:
        raise ValueError("Invalid threshold grid")
    if type(settings["threads"]) is not int or settings["threads"] < 1:
        raise ValueError("Invalid CPU thread count")


def probabilities(head, x):
    with torch.no_grad():
        chunks = [head(batch).softmax(1)[:, 1] for batch in x.split(1024)]
    return torch.cat(chunks).numpy()


def fit_head(x_train, y_train, x_validation, y_validation, settings, directory: Path,
             identity: dict, loss_name: str, progress=print, interrupt_after_epoch=None):
    """Input arrays must be train/validation only. The optional hook is for interruption tests."""
    settings_check(settings)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = settings["training"]
    x, y = torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long)
    vx, vy = torch.as_tensor(x_validation, dtype=torch.float32), torch.as_tensor(y_validation, dtype=torch.long)
    if x.ndim != 2 or vx.ndim != 2 or x.shape[1] != 2048 or vx.shape[1] != 2048:
        raise ValueError("Expected 2048-dimensional frozen features")
    if not torch.isfinite(x).all() or not torch.isfinite(vx).all():
        raise ValueError("Nonfinite training/validation features")
    if len(x) != len(y) or len(vx) != len(vy) or set(y.tolist()) != {0, 1} or set(vy.tolist()) != {0, 1}:
        raise ValueError("Aligned features and both classes required in train/validation")
    if loss_name not in cfg["losses"]:
        raise ValueError("Loss is not an enabled candidate")
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings["seed"])
    head = torch.nn.Linear(2048, 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    counts = torch.bincount(y, minlength=2)
    weights = len(y) / (2 * counts.float()) if loss_name == "class_weighted" else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    thresholds = np.arange(cfg["threshold_min"], cfg["threshold_max"] + 1e-9, cfg["threshold_step"]).round(8)
    checkpoint = directory / "last.pt"
    signature = digest({"identity": identity, "settings": settings, "loss": loss_name})
    history, best, stale, first_epoch = [], None, 0, 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Run configuration/data changed; use a new --run name")
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        history, best, stale = state["history"], state["best"], state["stale"]
        first_epoch = state["epoch"] + 1
    for epoch in range(first_epoch, cfg["max_epochs"]):
        if stale >= cfg["patience"]:
            break
        started = perf_counter()
        head.train()
        # No dropout; epoch-specific permutation makes resumed runs identical.
        permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(settings["seed"] + epoch))
        loss_sum = denominator = 0.0
        for indices in permutation.split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()):
                raise ValueError("Nonfinite gradients")
            optimizer.step()
            normalization = len(indices) if weights is None else float(weights[y[indices]].sum())
            loss_sum += float(loss.detach()) * normalization
            denominator += normalization
        head.eval()
        p = probabilities(head, vx)
        threshold = choose_threshold(vy.numpy(), p, thresholds)
        validation = metrics(vy.numpy(), p, threshold)
        with torch.no_grad():
            val_loss = float(torch.nn.functional.cross_entropy(head(vx), vy))
        score = (validation["f1_strike"], -val_loss)
        if best is None or score > tuple(best["score"]):
            best = {"score": list(score), "epoch": epoch, "threshold": threshold,
                    "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                    "validation": validation, "validation_loss": val_loss}
            stale = 0
        else:
            stale += 1
        record = {"epoch": epoch + 1, "train_loss": loss_sum / denominator,
                  "validation_loss": val_loss, "validation_strike_f1": validation["f1_strike"],
                  "threshold": threshold, "seconds": perf_counter() - started}
        history.append(record)
        atomic_torch(checkpoint, {"signature": signature, "epoch": epoch, "head": head.state_dict(),
                     "optimizer": optimizer.state_dict(), "history": history, "best": best, "stale": stale})
        atomic_json(directory / "history.json", history)
        progress(f"{loss_name} epoch {epoch + 1}/{cfg['max_epochs']}: "
                 f"val F1={validation['f1_strike']:.4f}, threshold={threshold:.2f}, "
                 f"{record['seconds']:.2f}s")
        if interrupt_after_epoch is not None:
            interrupt_after_epoch(epoch)
    if best is None:
        raise ValueError("No completed training epoch")
    head.load_state_dict(best["head"])
    head.eval()
    atomic_json(directory / "history.json", history)
    atomic_torch(directory / "best.pt", {"head": best["head"], "threshold": best["threshold"],
                 "epoch": best["epoch"], "signature": signature})
    return head, best, history


def train(root: Path, ctx: dict, settings: dict, run_name: str):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Run name must contain only letters, digits, underscore or hyphen")
    settings_check(settings)
    output = root / "outputs/strike-detection" / run_name
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "training.lock"), timeout=0):
        started = perf_counter()
        tx, tr, tp = load_features(root, ctx, "train")
        vx, vr, vp = load_features(root, ctx, "validation")
        load_seconds = perf_counter() - started
        y, vy = np.array([CLASS_MAP[r["label"]] for r in tr]), np.array([CLASS_MAP[r["label"]] for r in vr])
        identity = {"cache_key": ctx["key"], "split": ctx["split"],
                    "scikit_learn_version": importlib.metadata.version("scikit-learn"),
                    "sources": {"train": tp, "validation": vp},
                    "code": {p: sha256(Path(__file__).parent / p) for p in ("training.py", "metrics.py")}}
        run_info = {"identity": identity, "settings": settings}
        path = output / "run.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != run_info:
            raise ValueError("Existing run differs; choose a new --run name")
        atomic_json(path, run_info)
        candidates = []
        for name in settings["training"]["losses"]:
            head, best, history = fit_head(tx, y, vx, vy, settings, output / name, identity, name,
                                         progress=lambda s: print(s, flush=True))
            candidates.append((name, head, best, history))
        name, head, best, history = max(candidates, key=lambda c: tuple(c[2]["score"]))
        threshold = best["threshold"]
        p = probabilities(head, torch.from_numpy(vx))
        artifact = {"schema": "strike_head_candidate_v1", "head": head.state_dict(),
                    "threshold": threshold, "class_map": CLASS_MAP, "feature_dim": 2048,
                    "selected_loss": name, "selected_epoch": best["epoch"] + 1,
                    "training_settings": settings, "validation": best["validation"],
                    "identity": identity, "extractor": ctx["identity"],
                    "status": "validation_selected_test_not_evaluated"}
        atomic_torch(output / "candidate.pt", artifact)
        report = {"selected_loss": name, "selected_epoch": best["epoch"] + 1,
                  "train": evaluation(tr, probabilities(head, torch.from_numpy(tx)), threshold),
                  "threshold": threshold, "validation": evaluation(vr, p, threshold),
                  "cache_load_seconds": load_seconds,
                  "candidate_epoch_seconds": {n: sum(h["seconds"] for h in hist) for n, _, _, hist in candidates},
                  "candidates": {n: b["validation"] for n, _, b, _ in candidates},
                  "test_evaluated": False}
        atomic_json(output / "validation.json", report)
        prediction_path = output / "validation_predictions.jsonl"
        candidate_hash = sha256(output / "candidate.pt")
        with prediction_path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as stream:
            for row, probability in zip(vr, p):
                record = {k: row[k] for k in ("window_id", "video_id", "fight_id", "category",
                          "start_frame", "end_frame", "fps", "start_seconds", "end_seconds", "label")}
                record.update(p_strike=float(probability), predicted_label="strike" if probability >= threshold else "no_strike",
                              threshold=threshold, model_version=candidate_hash)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(prediction_path.with_suffix(".jsonl.tmp"), prediction_path)
        np.savez(output / "reload_reference.npz", embeddings=vx[:8], probabilities=p[:8])
        atomic_json(output / "READY.json", {"candidate_sha256": sha256(output / "candidate.pt"),
                    "validation_sha256": sha256(output / "validation.json"),
                    "reload_reference_sha256": sha256(output / "reload_reference.npz"),
                    "status": "Ready for clean-process reload check and human validation review; test closed"})
    return output
