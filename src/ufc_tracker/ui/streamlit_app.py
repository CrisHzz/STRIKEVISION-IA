"""Streamlit interface for the local StrikeVision video lab."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ufc_tracker.services.video_lab import AVAILABLE_STAGES, run_video_lab

st.set_page_config(page_title="StrikeVision Video Lab", page_icon="🥊", layout="wide")
st.title("🥊 StrikeVision Video Lab")
st.caption("Laboratorio local para probar progresivamente el pipeline de video y ML.")

input_column, settings_column = st.columns([3, 2])
with input_column:
    upload = st.file_uploader(
        "Sube un round",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=False,
    )
    if upload is not None:
        st.video(upload)

with settings_column:
    stage = st.selectbox("Etapa del pipeline", AVAILABLE_STAGES, index=1)
    draw_frame = st.checkbox("Mostrar frame y timestamp", value=True)
    run_clicked = st.button("Procesar video", type="primary", use_container_width=True)

if run_clicked:
    if upload is None:
        st.error("Carga un video antes de iniciar el análisis.")
    else:
        suffix = Path(upload.name).suffix.lower()
        progress_bar = st.progress(0, text="Preparando video")

        def update_progress(value: float, description: str) -> None:
            progress_bar.progress(min(int(value * 100), 100), text=description)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(upload.getbuffer())
                temporary_path = Path(temporary_file.name)

            result = run_video_lab(
                temporary_path,
                stage,
                draw_frame_number=draw_frame,
                progress=update_progress,
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            st.success(f"{result.status} Job: {result.job_id}")
            result_tab, metadata_tab, roadmap_tab = st.tabs(
                ["Resultado", "Metadata", "Próximas etapas"]
            )
            with result_tab:
                playable_video = result.output_path or result.input_path
                st.video(str(playable_video))
                if result.output_path is not None:
                    st.download_button(
                        "Descargar video procesado",
                        data=result.output_path.read_bytes(),
                        file_name=result.output_path.name,
                        mime="video/mp4",
                    )
            with metadata_tab:
                st.json(result.metadata.to_dict())
            with roadmap_tab:
                st.write(
                    "Detection → Tracking → Roles → Pose → Strike candidates → "
                    "Clasificación → Video anotado + JSON"
                )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
