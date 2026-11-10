"""Gradio tester for a trained per-fighter action-recognition candidate."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES
from ufc_tracker.action_recognition.inference import ActionInferenceEngine
from ufc_tracker.detection.weights import project_root
from ufc_tracker.strike.inference import MAX_ANALYSIS_FRAMES

ROOT = project_root(Path(__file__).resolve())
ENGINE = ActionInferenceEngine(ROOT)
CURRENT_CANDIDATE_DIR = ROOT / "outputs/action-recognition-v4/tcn-v4-combinado-verificacion-20261110-140614"


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}".replace(".", ",")


def _validation_metrics(candidate_dir: Path) -> str:
    report_path = candidate_dir / "report.json"
    if not report_path.is_file():
        return "## Métricas de clasificación\n\nEste candidato no tiene un reporte de validación disponible."
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("candidate_metrics") or report.get("metrics") or {}
    if not metrics.get("classes"):
        return "## Métricas de clasificación\n\nEl reporte no contiene métricas por clase."

    summary = "Validación de desarrollo"
    if report.get("validation_samples") is not None:
        samples = f"{report['validation_samples']:,}".replace(",", ".")
        summary += f" · {samples} muestras"
    lines = [
        "## Métricas de clasificación",
        "",
        summary,
        "",
        f"**Macro-F1 principal** {_format_metric(metrics.get('primary_macro_f1'))} · "
        f"**Macro-AP principal** {_format_metric(metrics.get('primary_macro_average_precision'))}",
        "",
        "| Clase | Soporte | Precisión | Recall | F1 | AP | Umbral |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ACTION_CLASSES:
        scores = metrics["classes"][label]
        values = [_format_metric(scores.get(key)) for key in ("precision", "recall", "f1", "average_precision")]
        threshold = _format_metric(metrics.get("thresholds", {}).get(label))
        lines.append(f"| `{label}` | {scores['support']} | {' | '.join(values)} | {threshold} |")
    lines.extend(("", "Nunes–Shevchenko II queda reservada para evaluación externa."))
    return "\n".join(lines)


def analyze_video(video_path, frames, candidate_dir, tracking_device=None, pose_python=None, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Carga un video antes de analizarlo.")
    if not candidate_dir:
        raise gr.Error("Indica el directorio de un candidato entrenado.")
    try:
        result = ENGINE.analyze(
            video_path, int(frames), candidate_dir, tracking_device=tracking_device, pose_python=pose_python,
            progress=lambda value, description: progress(value, desc=description),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    rows = [[event["fighter_id"], event["label"], event["start_frame"], event["end_frame"],
             round(event["max_probability"], 4), event["windows"]] for event in result["events"]]
    metadata = {key: result[key] for key in ("candidate_kind", "classes", "thresholds", "action_threshold", "requested_frames", "analyzed_frames", "fps", "feature_spec", "postprocessing", "pose_metrics", "tracking_device", "note")}
    return (result["preview_path"], rows, metadata,
            f"Se analizaron {result['analyzed_frames']} frames y se consolidaron {len(rows)} eventos legibles por ID.",
            result["json_path"], result["timeline_csv_path"])


def build_app(
    candidate_dir: Path | None = None, *, tracking_device: str | None = None, pose_python: Path | None = None,
) -> gr.Blocks:
    selected = (candidate_dir or CURRENT_CANDIDATE_DIR).resolve()
    ENGINE._load_candidate(selected)
    title = "v4 combinado" if selected == CURRENT_CANDIDATE_DIR else selected.name
    metrics = _validation_metrics(selected)
    with gr.Blocks(title="StrikeVision · Reconocimiento de acciones") as app:
        candidate = gr.State(str(selected))
        device = gr.State(tracking_device)
        worker = gr.State(str(pose_python.resolve()) if pose_python is not None else None)
        gr.Markdown(f"""# StrikeVision · Reconocimiento de acciones

**Modelo en uso · {title}**

Carga un video para ver el tracking, la pose y la técnica predicha de cada peleador.
El análisis conserva el FPS del archivo. ID 1 es rojo e ID 2 es morado.
""")
        with gr.Row():
            video = gr.Video(label="Video de la pelea", sources=["upload"])
            with gr.Column():
                frames = gr.Number(value=900, minimum=24, maximum=MAX_ANALYSIS_FRAMES, precision=0, label="Frames iniciales a analizar")
                with gr.Row():
                    buttons = [("30 s", 900), ("1:00", 1800), ("1:30", 2700), ("2:00", 3600), ("2:30", 4500), ("3:00", 5400)]
                    for name, value in buttons:
                        gr.Button(name).click(lambda preset=value: preset, outputs=frames, queue=False)
                analyze = gr.Button("Analizar acciones", variant="primary")
                status = gr.Markdown("Selecciona un video y cuántos frames quieres analizar.")
        preview = gr.Video(label="Pose, IDs y técnica predicha")
        table = gr.Dataframe(headers=["ID", "Técnica", "Inicio", "Fin", "Probabilidad máxima", "Ventanas"], interactive=False, label="Eventos predichos")
        with gr.Row():
            metadata = gr.JSON(label="Resumen")
            download = gr.File(label="Descargar JSON")
            timeline = gr.File(label="Diagnóstico temporal CSV")
        analyze.click(analyze_video, inputs=[video, frames, candidate, device, worker], outputs=[preview, table, metadata, status, download, timeline], show_progress="full")
        gr.Markdown(metrics)
    return app
