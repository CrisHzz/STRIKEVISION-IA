"""Shared video-lab service used by the Gradio and Streamlit interfaces."""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOBS_ROOT = PROJECT_ROOT / "outputs" / "jobs"
ALLOWED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MAX_DURATION_SECONDS = 5 * 60

STAGE_INSPECT = "Inspección del video"
STAGE_VISUAL_V0 = "Pipeline visual (V0)"
AVAILABLE_STAGES = (STAGE_INSPECT, STAGE_VISUAL_V0)

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class VideoMetadata:
    filename: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str
    size_mb: float

    def to_dict(self) -> dict[str, str | int | float]:
        data = asdict(self)
        data["duration_seconds"] = round(self.duration_seconds, 2)
        data["fps"] = round(self.fps, 2)
        data["size_mb"] = round(self.size_mb, 2)
        return data


@dataclass(frozen=True)
class LabResult:
    input_path: Path
    output_path: Path | None
    metadata: VideoMetadata
    job_id: str
    stage: str

    @property
    def status(self) -> str:
        if self.output_path is None:
            return "Video validado. No se generó un archivo nuevo en esta etapa."
        return "Pipeline visual V0 completado correctamente."


def run_video_lab(
    uploaded_path: str | Path,
    stage: str,
    *,
    draw_frame_number: bool = True,
    progress: ProgressCallback | None = None,
) -> LabResult:
    """Persist an uploaded video and run one supported experimentation stage."""
    if stage not in AVAILABLE_STAGES:
        raise ValueError(f"Etapa no soportada: {stage}")

    source = Path(uploaded_path)
    _validate_source(source)
    job_id = uuid.uuid4().hex
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / f"input{source.suffix.lower()}"
    shutil.copy2(source, input_path)

    _report(progress, 0.05, "Validando video")
    metadata = inspect_video(input_path)
    if metadata.duration_seconds > MAX_DURATION_SECONDS:
        shutil.rmtree(job_dir)
        raise ValueError("El video supera el límite de 5 minutos definido para un round.")

    output_path: Path | None = None
    if stage == STAGE_VISUAL_V0:
        output_path = job_dir / "visual_preview.mp4"
        _render_visual_v0(
            input_path,
            output_path,
            metadata,
            draw_frame_number=draw_frame_number,
            progress=progress,
        )

    _report(progress, 1.0, "Proceso terminado")
    return LabResult(input_path, output_path, metadata, job_id, stage)


def inspect_video(path: str | Path) -> VideoMetadata:
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("OpenCV no pudo abrir el video cargado.")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_number = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("El video no contiene metadata válida para ser procesado.")

    codec = "".join(chr((codec_number >> (8 * index)) & 0xFF) for index in range(4))
    return VideoMetadata(
        filename=video_path.name,
        duration_seconds=frame_count / fps,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        codec=codec.strip() or "unknown",
        size_mb=video_path.stat().st_size / (1024 * 1024),
    )


def _render_visual_v0(
    input_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    *,
    draw_frame_number: bool,
    progress: ProgressCallback | None,
) -> None:
    raw_output_path = output_path.with_name("visual_preview_raw.mp4")
    capture = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("No fue posible inicializar la lectura o escritura del video.")

    try:
        frame_index = 0
        progress_interval = max(1, metadata.frame_count // 100)
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            cv2.rectangle(frame, (18, 18), (390, 78), (12, 12, 12), -1)
            cv2.putText(
                frame,
                "STRIKEVISION | PIPELINE VISUAL V0",
                (32, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (60, 215, 190),
                2,
                cv2.LINE_AA,
            )
            if draw_frame_number:
                timestamp = frame_index / metadata.fps
                cv2.putText(
                    frame,
                    f"Frame {frame_index} | {timestamp:07.2f}s",
                    (32, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            frame_index += 1

            if frame_index % progress_interval == 0:
                fraction = min(frame_index / metadata.frame_count, 1.0)
                _report(progress, 0.1 + (fraction * 0.85), "Renderizando frames")
    finally:
        capture.release()
        writer.release()

    if not raw_output_path.exists() or raw_output_path.stat().st_size == 0:
        raise RuntimeError("El renderer no produjo un video de salida válido.")
    _encode_for_browser(raw_output_path, output_path)


def _encode_for_browser(raw_path: Path, output_path: Path) -> None:
    """Encode OpenCV's intermediate MP4 as browser-compatible H.264 when FFmpeg exists."""
    executable = _find_ffmpeg()
    if executable is None:
        raw_path.replace(output_path)
        return

    command = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raw_path.replace(output_path)
        return
    raw_path.unlink(missing_ok=True)


def _find_ffmpeg() -> Path | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable)

    candidates = (
        Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",
        Path(sys.prefix) / "bin" / "ffmpeg",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _validate_source(source: Path) -> None:
    if not source.exists() or not source.is_file():
        raise ValueError("Debes cargar un archivo de video válido.")
    if source.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_SUFFIXES))
        raise ValueError(f"Formato no soportado. Usa uno de estos: {allowed}.")


def _report(callback: ProgressCallback | None, value: float, description: str) -> None:
    if callback is not None:
        callback(value, description)
