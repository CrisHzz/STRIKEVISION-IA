"""Gradio UI for bounded manual testing of the strike/no_strike model."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ufc_tracker.detection.weights import project_root
from ufc_tracker.strike.inference import (
    MAX_ANALYSIS_FRAMES,
    WINDOW_FRAMES,
    StrikeInferenceEngine,
)


ROOT = project_root(Path(__file__).resolve())
ENGINE = StrikeInferenceEngine(ROOT)


def analyze_video(
    video_path: str | None,
    frames: int | float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, list[list[object]], dict[str, object], str, str]:
    if not video_path:
        raise gr.Error("Carga un video antes de analizarlo.")
    try:
        result = ENGINE.analyze(
            video_path,
            int(frames),
            progress=lambda value, description: progress(value, desc=description),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    rows = [
        [
            item["start_frame"],
            item["end_frame"],
            round(float(item["start_seconds"]), 3),
            round(float(item["end_seconds"]), 3),
            round(float(item["p_strike"]), 4),
            item["predicted_label"],
        ]
        for item in result["windows"]
    ]
    metadata = {
        key: result[key]
        for key in (
            "model_run",
            "threshold_fixed_from_validation",
            "requested_frames",
            "analyzed_frames",
            "fps",
            "strike_windows",
            "note",
        )
    }
    status = (
        f"Se analizaron {result['analyzed_frames']} frames en {len(rows)} ventanas. "
        f"{result['strike_windows']} ventanas superaron el umbral fijo."
    )
    return str(result["preview_path"]), rows, metadata, status, str(result["json_path"])


def build_app() -> gr.Blocks:
    with gr.Blocks(title="StrikeVision · Probador strike/no_strike") as app:
        gr.Markdown(
            f"""
            # StrikeVision · Probador `strike/no_strike`

            Carga un video e indica cuántos frames iniciales analizar. Se procesan
            únicamente esos frames; cada predicción cubre 24 frames y las ventanas
            avanzan 12 frames. El umbral del modelo `head-v1` está fijado en 0.33.
            Los resultados son una revisión del baseline, no un conteo final de golpes.
            """
        )
        with gr.Row():
            input_video = gr.Video(label="Video de la pelea", sources=["upload"])
            with gr.Column():
                frame_count = gr.Number(
                    value=240,
                    minimum=WINDOW_FRAMES,
                    maximum=MAX_ANALYSIS_FRAMES,
                    precision=0,
                    label="Frames iniciales a analizar",
                )
                gr.Markdown("Atajos para video a 30 FPS")
                presets = [
                    ("30 s", 900),
                    ("1:00", 1_800),
                    ("1:30", 2_700),
                    ("2:00", 3_600),
                    ("2:30", 4_500),
                    ("3:00", 5_400),
                ]
                with gr.Row():
                    preset_buttons = [gr.Button(label) for label, _ in presets]
                for button, (_, preset_frames) in zip(preset_buttons, presets):
                    button.click(
                        lambda value=preset_frames: value,
                        outputs=frame_count,
                        queue=False,
                    )
                analyze = gr.Button("Analizar strike / no_strike", variant="primary")
                status = gr.Markdown("Listo para recibir un video.")
        preview = gr.Video(label="Vista previa con predicción por ventana")
        results = gr.Dataframe(
            headers=["Inicio", "Fin", "Inicio (s)", "Fin (s)", "p(strike)", "Predicción"],
            label="Ventanas analizadas",
            interactive=False,
        )
        with gr.Row():
            metadata = gr.JSON(label="Resumen")
            download = gr.File(label="Descargar resultados JSON")
        analyze.click(
            analyze_video,
            inputs=[input_video, frame_count],
            outputs=[preview, results, metadata, status, download],
            show_progress="full",
        )
    return app
