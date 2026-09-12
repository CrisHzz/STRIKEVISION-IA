"""Combined pose/tracking and scene-level strike visualization."""

import json
import uuid

import cv2
import gradio as gr
import yaml

from ufc_tracker.pose.pipeline import load_pose_records, render_pose_preview, run_pose_pipeline
from ufc_tracker.strike.inference import frame_predictions
from ufc_tracker.ui.strike_inference_app import ENGINE, ROOT, build_app as build_strike_app


def analyze_video(video_path, frames, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Carga un video antes de analizarlo.")
    try:
        strike = ENGINE.analyze(
            video_path, frames, generate_preview=False,
            progress=lambda value, description: progress(value * 0.35, desc=description),
        )
        destination = ROOT / "outputs/strike-detection/gradio-pose-strike" / uuid.uuid4().hex
        config = yaml.safe_load((ROOT / "configs/app/pose_pipeline.yaml").read_text(encoding="utf-8"))

        def pose_progress(stage, current, total):
            base = 0.35 if stage == "tracking" else 0.60
            progress(base + 0.25 * current / max(total, 1), desc=f"{stage}: {current}/{total} frames")

        pose = run_pose_pipeline(
            video_path, destination,
            tracking_confidence=float(config.get("tracking_confidence", 0.5)),
            merge_track_fragments=True, max_frames=strike["analyzed_frames"],
            generate_preview=False, progress_callback=pose_progress,
        )
        metrics = json.loads(pose.metrics_path.read_text(encoding="utf-8"))
        if metrics["frame_count"] != strike["analyzed_frames"] or abs(metrics["fps"] - strike["fps"]) > 0.01:
            raise ValueError("Los resultados de pose y strike no coinciden en frames o FPS.")
        predictions = frame_predictions(strike["windows"], strike["analyzed_frames"])

        def overlay(frame, index):
            prediction = predictions[index]
            text = "Escena: sin ventana completa"
            color = (190, 190, 190)
            if prediction is not None:
                text = f"Escena: {prediction['predicted_label']} | p(strike)={prediction['p_strike']:.3f}"
                color = (40, 190, 40) if prediction["predicted_label"] == "strike" else (40, 140, 240)
            cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 680), 54), (0, 0, 0), -1)
            cv2.putText(frame, text, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)

        preview = destination / "pose_strike_preview.mp4"
        render_pose_preview(
            video_path, load_pose_records(pose.pose_path), preview,
            frame_count=strike["analyzed_frames"], frame_overlay=overlay,
            progress_callback=lambda current, total: progress(0.85 + 0.15 * current / max(total, 1), desc="Generando video combinado…"),
        )
        metadata = {key: strike[key] for key in (
            "model_run", "threshold_fixed_from_validation", "requested_frames", "analyzed_frames", "fps", "strike_windows",
        )}
        metadata["pose_metrics"] = metrics["metrics"]
        metadata["note"] = "Strike/no strike describe la escena, no atribuye golpes a un peleador. Las ventanas solapadas no son conteos de golpes."
        result_path = destination / "results.json"
        result_path.write_text(json.dumps({
            **metadata, "windows": strike["windows"],
            "pose_file": str(pose.pose_path), "tracking_file": str(pose.tracking_path),
            "pose_run": json.loads(pose.metadata_path.read_text(encoding="utf-8")),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = [[w["start_frame"], w["end_frame"], round(w["start_seconds"], 3), round(w["end_seconds"], 3), round(w["p_strike"], 4), w["predicted_label"]] for w in strike["windows"]]
        return str(preview), rows, metadata, f"Se analizaron {strike['analyzed_frames']} frames con pose y strike/no strike.", str(result_path)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error


def build_app():
    return build_strike_app(handler=analyze_video, combined=True)
