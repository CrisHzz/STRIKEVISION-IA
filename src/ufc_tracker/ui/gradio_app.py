"""Gradio interface for the local StrikeVision video lab."""

from __future__ import annotations

import gradio as gr

from ufc_tracker.services.video_lab import AVAILABLE_STAGES, STAGE_VISUAL_V0, run_video_lab


def process_video(
    video_path: str | None,
    stage: str,
    draw_frame_number: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, dict[str, object], str, str | None]:
    if not video_path:
        raise gr.Error("Carga un video antes de iniciar el análisis.")

    try:
        result = run_video_lab(
            video_path,
            stage,
            draw_frame_number=draw_frame_number,
            progress=lambda value, description: progress(value, desc=description),
        )
    except (RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    playable_video = result.output_path or result.input_path
    download = str(result.output_path) if result.output_path else None
    status = f"{result.status} Job: `{result.job_id}`"
    return str(playable_video), result.metadata.to_dict(), status, download


def build_app() -> gr.Blocks:
    with gr.Blocks(title="StrikeVision Video Lab") as app:
        gr.Markdown(
            """
            # StrikeVision Video Lab
            Sube un round, elige una etapa y prueba el pipeline local de computer vision.
            La versión inicial implementa inspección y reconstrucción visual; los módulos ML se
            conectarán progresivamente al mismo servicio.
            """
        )
        with gr.Row():
            with gr.Column(scale=3):
                input_video = gr.Video(label="Video de entrada", sources=["upload"])
            with gr.Column(scale=2):
                stage = gr.Dropdown(
                    choices=list(AVAILABLE_STAGES),
                    value=STAGE_VISUAL_V0,
                    label="Etapa del pipeline",
                )
                draw_frame = gr.Checkbox(value=True, label="Mostrar frame y timestamp")
                run_button = gr.Button("Procesar video", variant="primary")
                status = gr.Markdown("Listo para recibir un video.")

        with gr.Tabs():
            with gr.Tab("Resultado"):
                output_video = gr.Video(label="Video procesado")
                download = gr.File(label="Descargar resultado")
            with gr.Tab("Metadata"):
                metadata = gr.JSON(label="Información del video")
            with gr.Tab("Próximas etapas"):
                gr.Markdown(
                    "Detection → Tracking → Roles → Pose → Strike candidates → "
                    "Clasificación → Video anotado + JSON"
                )

        run_button.click(
            fn=process_video,
            inputs=[input_video, stage, draw_frame],
            outputs=[output_video, metadata, status, download],
            show_progress="full",
        )
    return app


demo = build_app()


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
