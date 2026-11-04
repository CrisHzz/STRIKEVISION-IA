"""Gradio tester for a trained per-fighter action-recognition candidate."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from ufc_tracker.action_recognition.inference import ActionInferenceEngine
from ufc_tracker.detection.weights import project_root
from ufc_tracker.strike.inference import MAX_ANALYSIS_FRAMES

ROOT = project_root(Path(__file__).resolve())
ENGINE = ActionInferenceEngine(ROOT)
CURRENT_CANDIDATE_NAME = "tcn-women-v4-fold0-cachefix"


def _action_output_roots(root: Path) -> tuple[Path, ...]:
    return (
        Path(root) / "outputs/action-recognition-v4",
        Path(root) / "outputs/action-recognition-v3",
        Path(root) / "outputs/action-recognition-v2",
        Path(root) / "outputs/action-recognition",
    )


def _has_verified_pose_cache_order(report_path: Path) -> bool:
    metadata_path = report_path.parent / "pose_features.json"
    if not metadata_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_ids = report.get("sample_ids", {})
    expected = list(sample_ids.get("train", ())) + list(sample_ids.get("validation", ()))
    return bool(expected) and metadata.get("sample_ids") == expected


def available_development_candidates(root: Path = ROOT) -> list[tuple[str, str]]:
    """Expose completed TCN trials without making the user navigate output paths."""
    best_by_fold: dict[int, tuple[float, str, str]] = {}
    report_paths = []
    for output_root in _action_output_roots(root):
        report_paths.extend(output_root.glob("*-optuna-fold*/trials/trial-*/report.json"))
    for report_path in report_paths:
        candidate_path = report_path.parent / "candidate.pt"
        if not candidate_path.is_file() or not _has_verified_pose_cache_order(report_path):
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        score = report.get("selection_score", report.get("metrics", {}).get("primary_macro_f1"))
        if score is None:
            continue
        fold = int(report.get("validation_fold", -1))
        family = str(report.get("family", "tcn"))
        version = "V2 jerárquico" if family == "hierarchical_tcn" else "TCN"
        metric_name = "score integrado" if family == "hierarchical_tcn" else "F1 desarrollo"
        label = f"{version} sintonizado · fold {fold} · {metric_name} {float(score):.3f}"
        entry = (float(score), label, str(candidate_path.parent))
        if fold not in best_by_fold or entry[0] > best_by_fold[fold][0]:
            best_by_fold[fold] = entry
    candidates = list(best_by_fold.values())
    seen = {entry[2] for entry in candidates}
    direct_reports = []
    for output_root in _action_output_roots(root):
        direct_reports.extend(output_root.glob("*/report.json"))
    for report_path in direct_reports:
        candidate_path = report_path.parent / "candidate.pt"
        candidate_dir = str(candidate_path.parent)
        if (
            not candidate_path.is_file()
            or candidate_dir in seen
            or not _has_verified_pose_cache_order(report_path)
        ):
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        score = report.get("selection_score", report.get("metrics", {}).get("primary_macro_f1", 0.0))
        fold = report.get("validation_fold", "?")
        run_name = report_path.parent.name
        family = str(report.get("family", "modelo"))
        if "women-v4" in run_name:
            version = "TCN + dos peleas femeninas V4"
        elif "women-v3" in run_name:
            version = "TCN + Zhang–Joanna V3"
        elif "labels-v2" in run_name:
            version = "TCN etiquetas V2"
        elif family == "stgcn_compact":
            version = "ST-GCN compacto"
        elif family == "hierarchical_tcn":
            version = "TCN V2 jerárquico"
        elif family == "tcn":
            version = "TCN base"
        else:
            version = family
        active = run_name == CURRENT_CANDIDATE_NAME
        prefix = "MODELO ACTIVO · " if active else ""
        candidates.append((float(score or 0.0), f"{prefix}{version} · fold {fold} · F1 desarrollo {float(score or 0):.3f}", candidate_dir))
        seen.add(candidate_dir)
    # The evaluated V4 candidate stays selected even if an older experiment
    # happens to have a higher development score.
    ranked = sorted(
        candidates,
        key=lambda entry: (Path(entry[2]).name == CURRENT_CANDIDATE_NAME, entry[0], entry[1]),
        reverse=True,
    )
    return [(label, path) for _, label, path in ranked]


def analyze_video(video_path, frames, candidate_dir, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Carga un video antes de analizarlo.")
    if not candidate_dir:
        raise gr.Error("Indica el directorio de un candidato entrenado.")
    try:
        result = ENGINE.analyze(video_path, int(frames), candidate_dir, progress=lambda value, description: progress(value, desc=description))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    rows = [[event["fighter_id"], event["label"], event["start_frame"], event["end_frame"],
             round(event["max_probability"], 4), event["windows"]] for event in result["events"]]
    metadata = {key: result[key] for key in ("candidate_kind", "classes", "thresholds", "action_threshold", "requested_frames", "analyzed_frames", "fps", "feature_spec", "postprocessing", "pose_metrics", "note")}
    return (result["preview_path"], rows, metadata,
            f"Se analizaron {result['analyzed_frames']} frames y se consolidaron {len(rows)} eventos legibles por ID.",
            result["json_path"], result["timeline_csv_path"])


def build_app() -> gr.Blocks:
    candidates = available_development_candidates()
    if not candidates:
        raise FileNotFoundError("No hay candidatos TCN entrenados disponibles para el probador.")
    with gr.Blocks(title="StrikeVision · Probador de acciones") as app:
        gr.Markdown("""# StrikeVision · Probador de acciones

**Modelo activo: TCN V4 — Zhang–Joanna + Joanna–Andrade (macro-F1 de desarrollo: 0,353).**

Carga un video y prueba el candidato V4 actual. El análisis conserva el FPS del archivo,
aplica tracking y pose por peleador, y muestra las técnicas predichas por ID.
ID 1 es rojo e ID 2 es morado. `ambiguous` no existe como predicción: si la confianza
no supera el umbral, el video muestra «Sin técnica confiable». Este probador es para revisión visual;
las otras versiones del desplegable son únicamente referencias de comparación. El test reservado todavía
no participa en la selección del modelo.""")
        with gr.Row():
            video = gr.Video(label="Video de la pelea", sources=["upload"])
            with gr.Column():
                candidate = gr.Dropdown(choices=candidates, value=candidates[0][1], label="Modelo activo / referencia")
                frames = gr.Number(value=900, minimum=24, maximum=MAX_ANALYSIS_FRAMES, precision=0, label="Frames iniciales a analizar")
                with gr.Row():
                    buttons = [("30 s", 900), ("1:00", 1800), ("1:30", 2700), ("2:00", 3600), ("2:30", 4500), ("3:00", 5400)]
                    for name, value in buttons:
                        gr.Button(name).click(lambda preset=value: preset, outputs=frames, queue=False)
                analyze = gr.Button("Analizar acciones", variant="primary")
                status = gr.Markdown("Selecciona un video y la cantidad de frames; no necesitas escribir una ruta de modelo.")
        preview = gr.Video(label="Pose, IDs y técnica predicha")
        table = gr.Dataframe(headers=["ID", "Técnica", "Inicio", "Fin", "Probabilidad máxima", "Ventanas"], interactive=False, label="Eventos predichos")
        with gr.Row():
            metadata = gr.JSON(label="Resumen")
            download = gr.File(label="Descargar JSON")
            timeline = gr.File(label="Diagnóstico temporal CSV")
        analyze.click(analyze_video, inputs=[video, frames, candidate], outputs=[preview, table, metadata, status, download, timeline], show_progress="full")
    return app
