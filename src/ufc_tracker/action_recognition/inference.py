"""Per-fighter local inference for a trained action-recognition candidate."""

from __future__ import annotations

import json
import pickle
import uuid
import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES
from ufc_tracker.action_recognition.features import PoseFeatureSpec, extract_pose_sequence, pooled_pose_features
from ufc_tracker.action_recognition.models import build_torch_model
from ufc_tracker.action_recognition.rgb import iter_attacker_clips
from ufc_tracker.pose.pipeline import FIGHTER_COLORS_BY_ID, load_pose_records, render_pose_preview, run_pose_pipeline
from ufc_tracker.strike.model import build_model


class ActionInferenceEngine:
    """Run a candidate over merged fighter poses without changing the video FPS."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _load_candidate(self, run_dir: Path) -> tuple[str, dict[str, Any], PoseFeatureSpec, np.ndarray]:
        run_dir = Path(run_dir).resolve()
        classic = run_dir / "candidate.pkl"
        temporal = run_dir / "candidate.pt"
        if classic.is_file():
            with classic.open("rb") as stream:
                payload = pickle.load(stream)
            if payload.get("schema") == "action_classical_candidate_v1":
                kind = "classical"
            elif payload.get("schema") == "action_x3d_linear_probe_v1":
                kind = "x3d"
            else:
                raise ValueError("El candidato clásico no usa el contrato esperado")
        elif temporal.is_file():
            payload = torch.load(temporal, map_location="cpu", weights_only=False)
            if payload.get("schema") == "action_temporal_candidate_v1":
                kind = "temporal"
            elif payload.get("schema") == "action_temporal_candidate_v2":
                kind = "hierarchical_temporal"
                action_threshold = payload.get("action_threshold")
                if action_threshold is None or not 0 < float(action_threshold) < 1:
                    raise ValueError("El candidato jerárquico no tiene un umbral de acción válido")
            else:
                raise ValueError("El candidato temporal no usa el contrato esperado")
        else:
            raise FileNotFoundError(f"No existe candidate.pkl ni candidate.pt en {run_dir}")
        if tuple(payload.get("classes", ())) != ACTION_CLASSES:
            raise ValueError("El candidato usa otra taxonomía de acciones")
        spec = PoseFeatureSpec(**payload["feature_spec"])
        thresholds = np.asarray(payload["thresholds"], dtype=np.float32)
        if thresholds.shape != (len(ACTION_CLASSES),) or np.any((thresholds <= 0) | (thresholds >= 1)):
            raise ValueError("Los umbrales del candidato son inválidos")
        return kind, payload, spec, thresholds

    @staticmethod
    def _display_events(
        windows: list[dict[str, Any]], thresholds: np.ndarray, fps: float,
        *, minimum_confidence: float = 0.5, minimum_pose_coverage: float = 0.65,
        action_threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Turn overlapping multilabel windows into short, readable per-ID events.

        Training remains multilabel. The preview deliberately selects one strongest
        technique per fighter and decision instant: a fighter cannot visibly throw
        two distinct techniques in the same instant, while consecutive techniques
        and attacks from both fighters remain independent events.
        """
        raw_candidates: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for window in windows:
            coverage = float(window["pose_coverage"])
            action_probability = window.get("action_probability")
            action_eligible = (
                action_threshold is None
                or (action_probability is not None and float(action_probability) >= action_threshold)
            )
            decision_frame = int(round((int(window["start_frame"]) + int(window["end_frame"])) / 2))
            candidates = []
            for index, label in enumerate(ACTION_CLASSES):
                score = float(window["probabilities"][index])
                required = max(float(thresholds[index]), minimum_confidence)
                candidate = {
                    "fighter_id": window["fighter_id"], "label": label, "probability": score,
                    "threshold": required, "pose_coverage": coverage,
                    "action_probability": action_probability,
                    "action_threshold": action_threshold,
                    "window_start_frame": int(window["start_frame"]),
                    "window_end_frame": int(window["end_frame"]), "decision_frame": decision_frame,
                    "eligible": action_eligible and coverage >= minimum_pose_coverage and score >= required,
                }
                raw_candidates.append(candidate)
                if candidate["eligible"]:
                    candidates.append(candidate)
            if candidates:
                decisions.append(max(candidates, key=lambda item: (float(item["probability"]), item["label"])))

        events: list[dict[str, Any]] = []
        half_width = max(1, int(round(int(windows[0]["stride_frames"]) / 2))) if windows else 1
        max_duration = max(half_width * 2, int(round(0.9 * fps)))
        for decision in sorted(decisions, key=lambda item: (str(item["fighter_id"]), int(item["decision_frame"]))):
            start = max(0, int(decision["decision_frame"]) - half_width)
            end = int(decision["decision_frame"]) + half_width
            previous = next((event for event in reversed(events) if event["fighter_id"] == decision["fighter_id"]), None)
            if (
                previous is not None
                and previous["label"] == decision["label"]
                and start <= previous["end_frame"] + half_width
                and end - previous["start_frame"] + 1 <= max_duration
            ):
                previous["end_frame"] = max(previous["end_frame"], end)
                previous["max_probability"] = max(previous["max_probability"], float(decision["probability"]))
                previous["windows"] += 1
                previous["source_decision_frames"].append(int(decision["decision_frame"]))
                continue
            if previous is not None and start <= previous["end_frame"]:
                start = int(previous["end_frame"]) + 1
                end = max(end, start)
            events.append({
                "fighter_id": decision["fighter_id"], "label": decision["label"],
                "start_frame": start, "end_frame": end,
                "max_probability": float(decision["probability"]), "windows": 1,
                "source_decision_frames": [int(decision["decision_frame"])],
            })
        return events, raw_candidates

    @staticmethod
    def _write_timeline(destination: Path, windows: list[dict[str, Any]], candidates: list[dict[str, Any]], events: list[dict[str, Any]]) -> Path:
        path = destination / "action_timeline.csv"
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            rows.append({"record_type": "candidate", **candidate})
        for event in events:
            rows.append({"record_type": "display_event", **event})
        fields = sorted({key for row in rows for key in row} | {"record_type"})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def analyze(
        self, video_path: str | Path, requested_frames: int, candidate_dir: str | Path,
        *, progress=None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"No existe el video: {video_path}")
        if requested_frames < 24:
            raise ValueError("Se necesitan al menos 24 frames")
        kind, candidate, spec, thresholds = self._load_candidate(Path(candidate_dir))
        destination = self.root / "outputs/action-recognition/gradio" / uuid.uuid4().hex
        app_config = yaml.safe_load((self.root / "configs/app/pose_pipeline.yaml").read_text(encoding="utf-8"))
        def pose_progress(stage: str, current: int, total: int) -> None:
            if progress is not None:
                progress(0.75 * current / max(total, 1), f"{stage}: {current}/{total} frames")
        pose = run_pose_pipeline(
            video_path, destination, tracking_confidence=float(app_config.get("tracking_confidence", 0.5)),
            merge_track_fragments=True, max_frames=int(requested_frames), generate_preview=False,
            progress_callback=pose_progress,
        )
        metrics = json.loads(pose.metrics_path.read_text(encoding="utf-8"))
        fps, frame_count = float(metrics["fps"]), int(metrics["frame_count"])
        records = [record.__dict__ for record in load_pose_records(pose.pose_path)]
        window_frames = max(24, int(round(0.8 * fps)))
        stride = max(1, window_frames // 2)
        windows = []
        samples = []
        for start in range(0, max(frame_count - window_frames + 1, 0), stride):
            for fighter in ("1", "2"):
                samples.append({"sample_id": f"{start}:{fighter}", "attacker": fighter, "fps": fps,
                                "start_frame": start, "end_frame": start + window_frames - 1, "pose_usable": True})
        if not samples:
            raise ValueError("El video no contiene una ventana completa para analizar")
        sequences = [extract_pose_sequence(sample, records, spec) for sample in samples]
        features = np.stack([sequence["features"] for sequence in sequences])
        masks = np.stack([sequence["frame_mask"][:, 0] for sequence in sequences])
        if kind == "classical":
            pooled = np.stack([pooled_pose_features(sequence) for sequence in sequences])
            probabilities = np.column_stack([model.predict_proba(pooled)[:, 1] for model in candidate["models"]])
        elif kind == "x3d":
            rgb_samples = [{**sample, "video_path": str(video_path.resolve()), "pose_path": str(pose.pose_path.resolve())} for sample in samples]
            rgb_by_id = dict(iter_attacker_clips(self.root, rgb_samples, spec))
            compatible = [sample for sample in samples if sample["sample_id"] in rgb_by_id]
            if not compatible:
                raise ValueError("El tracking no produjo recortes completos para el candidato RGB")
            rgb_config = yaml.safe_load((self.root / "configs/models/strike_x3d_s.yaml").read_text(encoding="utf-8"))
            extractor = rgb_config["extractor"]
            backbone = build_model(self.root / extractor["weights_path"], extractor["weights_sha256"]).eval()
            tensors = torch.stack([rgb_by_id[sample["sample_id"]] for sample in compatible])
            with torch.no_grad():
                embeddings = backbone.embeddings(tensors).numpy()
            probability_by_id = {
                sample["sample_id"]: values for sample, values in zip(
                    compatible, np.column_stack([model.predict_proba(embeddings)[:, 1] for model in candidate["models"]]), strict=True,
                )
            }
            sequence_by_id = {sample["sample_id"]: sequence for sample, sequence in zip(samples, sequences, strict=True)}
            samples = [sample for sample in samples if sample["sample_id"] in probability_by_id]
            sequences = [sequence_by_id[sample["sample_id"]] for sample in samples]
            probabilities = np.stack([probability_by_id[sample["sample_id"]] for sample in samples])
        else:
            hierarchical = kind == "hierarchical_temporal"
            model = build_torch_model(candidate["name"], int(candidate["feature_dim"]), len(ACTION_CLASSES) + int(hierarchical), **candidate["parameters"])
            model.load_state_dict(candidate["state_dict"], strict=True); model.eval()
            with torch.no_grad():
                raw_probabilities = torch.sigmoid(model(torch.from_numpy(features), torch.from_numpy(masks))).numpy()
            if hierarchical:
                action_probabilities = raw_probabilities[:, 0]
                probabilities = raw_probabilities[:, 1:]
            else:
                action_probabilities = np.full(len(raw_probabilities), np.nan, dtype=np.float32)
                probabilities = raw_probabilities
        if kind in {"classical", "x3d"}:
            action_probabilities = np.full(len(probabilities), np.nan, dtype=np.float32)
        for sample, probability, action_probability, sequence in zip(samples, probabilities, action_probabilities, sequences, strict=True):
            windows.append({"fighter_id": sample["attacker"], "start_frame": sample["start_frame"], "end_frame": sample["end_frame"],
                            "start_seconds": sample["start_frame"] / fps, "end_seconds": sample["end_frame"] / fps,
                            "stride_frames": stride, "pose_coverage": float(sequence["pose_coverage"]),
                            "action_probability": None if np.isnan(action_probability) else float(action_probability),
                            "probabilities": [float(value) for value in probability]})
        action_threshold = float(candidate["action_threshold"]) if kind == "hierarchical_temporal" else None
        events, candidates = self._display_events(windows, thresholds, fps, action_threshold=action_threshold)
        timeline_path = self._write_timeline(destination, windows, candidates, events)
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            for frame in range(event["start_frame"], event["end_frame"] + 1):
                by_frame.setdefault(frame, []).append(event)
        pose_by_frame: dict[int, dict[str, Any]] = defaultdict(dict)
        for record in load_pose_records(pose.pose_path):
            pose_by_frame[record.frame_index][record.fighter_id] = record
        preview = destination / "action_preview.mp4"
        def overlay(frame: np.ndarray, index: int) -> None:
            current: dict[str, dict[str, Any]] = {}
            for event in by_frame.get(index, []):
                fighter_id = str(event["fighter_id"])
                if fighter_id not in current or float(event["max_probability"]) > float(current[fighter_id]["max_probability"]):
                    current[fighter_id] = event
            cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 390), 30), (12, 12, 12), -1)
            if not current:
                cv2.putText(frame, "SIN ACCION CLARA", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 185, 185), 1, cv2.LINE_AA)
                return
            for fighter_id, event in sorted(current.items()):
                record = pose_by_frame.get(index, {}).get(fighter_id)
                color = FIGHTER_COLORS_BY_ID.get(fighter_id, (255, 255, 255))
                label = f"ID {fighter_id}  {str(event['label']).upper().replace('_', ' ')}"
                confidence = f" {float(event['max_probability']):.0%}"
                text = label + confidence
                (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
                if record is not None and record.bbox_xyxy is not None:
                    x1, y1, x2, y2 = (int(value) for value in record.bbox_xyxy)
                    x = max(4, min(x1, frame.shape[1] - width - 14))
                    y = y1 - 10
                    if y - height - baseline - 10 < 30:
                        y = min(frame.shape[0] - 8, y2 + height + baseline + 10)
                else:
                    x = 8 if fighter_id == "1" else max(8, frame.shape[1] - width - 14)
                    y = 58 if fighter_id == "1" else 88
                cv2.rectangle(frame, (x - 6, y - height - 7), (x + width + 7, y + baseline + 6), color, -1)
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (16, 16, 16), 2, cv2.LINE_AA)
        render_pose_preview(video_path, load_pose_records(pose.pose_path), preview, frame_count=frame_count, frame_overlay=overlay,
                            progress_callback=(lambda current, total: progress(0.75 + 0.25 * current / max(total, 1), "Renderizando…")) if progress else None)
        result = {"schema_version": "action_inference_v1", "candidate_dir": str(Path(candidate_dir).resolve()),
                  "candidate_kind": kind, "classes": list(ACTION_CLASSES), "thresholds": thresholds.tolist(),
                  "action_threshold": action_threshold,
                  "requested_frames": int(requested_frames), "analyzed_frames": frame_count, "fps": fps,
                  "feature_spec": asdict(spec), "windows": windows, "candidates": candidates, "events": events,
                  "postprocessing": {"decision_anchor": "window_center", "minimum_confidence": 0.5,
                                     "minimum_pose_coverage": 0.65, "maximum_event_seconds": 0.9},
                  "pose_metrics": metrics.get("metrics", {}),
                  "timeline_csv_path": str(timeline_path),
                  "note": "Los eventos son técnicas predichas. No son golpes conectados ni un conteo clínicamente exacto."}
        result_path = destination / "results.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**result, "preview_path": str(preview), "json_path": str(result_path)}
