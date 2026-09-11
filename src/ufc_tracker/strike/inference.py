"""Bounded local inference for the X3D-S strike/no_strike candidate."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import yaml

from ufc_tracker.strike.dataset import CLASS_MAP, clip_tensor, letterbox
from ufc_tracker.strike.model import build_model


WINDOW_FRAMES = 24
WINDOW_STRIDE = 12
SAMPLED_FRAMES = 13
# Five minutes of a standard 30 FPS UFC broadcast round.
MAX_ANALYSIS_FRAMES = 9_000


def window_starts(frame_count: int) -> list[int]:
    """Return the same 24-frame, 12-frame-stride windows used by the dataset."""
    if frame_count < WINDOW_FRAMES:
        return []
    return list(range(0, frame_count - WINDOW_FRAMES + 1, WINDOW_STRIDE))


def sampled_offsets() -> np.ndarray:
    """Uniform inclusive sample positions matching the training preprocessing."""
    return np.rint(np.linspace(0, WINDOW_FRAMES - 1, SAMPLED_FRAMES)).astype(int)


def frame_predictions(windows: list[dict[str, object]], frame_count: int) -> list[dict[str, object] | None]:
    """Choose the strongest overlapping window only for preview rendering."""
    output: list[dict[str, object] | None] = []
    for frame_index in range(frame_count):
        candidates = [
            window
            for window in windows
            if int(window["start_frame"]) <= frame_index <= int(window["end_frame"])
        ]
        output.append(max(candidates, key=lambda item: float(item["p_strike"]), default=None))
    return output


@dataclass(frozen=True)
class LoadedCandidate:
    backbone: torch.nn.Module
    head: torch.nn.Linear
    threshold: float
    run_name: str


class StrikeInferenceEngine:
    """Load one validated candidate and apply it only to a bounded video prefix."""

    def __init__(self, root: Path, run_name: str = "head-v1") -> None:
        self.root = root.resolve()
        self.run_name = run_name
        self._candidate: LoadedCandidate | None = None

    def _load(self) -> LoadedCandidate:
        if self._candidate is not None:
            return self._candidate
        run_dir = self.root / "outputs" / "strike-detection" / self.run_name
        artifact_path = run_dir / "candidate.pt"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"No existe el candidato: {artifact_path}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if artifact.get("schema") != "strike_head_candidate_v1":
            raise ValueError("El artefacto no es un candidato de strike compatible.")
        if artifact.get("class_map") != CLASS_MAP or artifact.get("feature_dim") != 2048:
            raise ValueError("El candidato no usa el contrato binario strike/no_strike esperado.")
        threshold = float(artifact["threshold"])
        if not 0 < threshold < 1:
            raise ValueError("El umbral del candidato es inválido.")

        config_path = self.root / "configs" / "models" / "strike_x3d_s.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        extractor = config["extractor"]
        torch.set_num_threads(int(config["threads"]))
        backbone = build_model(
            self.root / extractor["weights_path"], extractor["weights_sha256"]
        ).eval()
        head = torch.nn.Linear(2048, 2)
        head.load_state_dict(artifact["head"], strict=True)
        head.eval()
        self._candidate = LoadedCandidate(backbone, head, threshold, self.run_name)
        return self._candidate

    @staticmethod
    def _decode_prefix(video_path: Path, requested_frames: int) -> tuple[list[np.ndarray], float]:
        frames: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate or 0)
            if fps <= 0:
                raise ValueError("El video no declara una tasa de frames válida.")
            for frame in container.decode(stream):
                frames.append(letterbox(frame.to_ndarray(format="rgb24")))
                if len(frames) >= requested_frames:
                    break
        if not frames:
            raise ValueError("No se pudieron leer frames del video.")
        return frames, fps

    @staticmethod
    def _write_preview(
        source: Path,
        destination: Path,
        frame_count: int,
        fps: float,
        predictions: list[dict[str, object] | None],
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer: cv2.VideoWriter | None = None
        try:
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                for index, frame in enumerate(container.decode(stream)):
                    if index >= frame_count:
                        break
                    image = frame.to_ndarray(format="bgr24")
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(destination),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("No se pudo crear la vista previa MP4.")
                    prediction = predictions[index]
                    if prediction is None:
                        text, color = "Sin ventana completa", (190, 190, 190)
                    else:
                        label = str(prediction["predicted_label"])
                        score = float(prediction["p_strike"])
                        text = f"{label} | p(strike)={score:.3f}"
                        color = (40, 190, 40) if label == "strike" else (40, 140, 240)
                    cv2.rectangle(image, (0, 0), (min(image.shape[1], 530), 54), (0, 0, 0), -1)
                    cv2.putText(image, text, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)
                    cv2.putText(image, f"frame {index}", (14, image.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    writer.write(image)
        finally:
            if writer is not None:
                writer.release()

    def analyze(
        self,
        video_path: str | Path,
        requested_frames: int,
        *,
        progress=lambda _value, _description: None,
    ) -> dict[str, object]:
        if isinstance(requested_frames, bool) or int(requested_frames) != requested_frames:
            raise ValueError("La cantidad de frames debe ser un número entero.")
        requested_frames = int(requested_frames)
        if not WINDOW_FRAMES <= requested_frames <= MAX_ANALYSIS_FRAMES:
            raise ValueError(
                f"Ingresa entre {WINDOW_FRAMES} y {MAX_ANALYSIS_FRAMES} frames."
            )
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError("El video cargado ya no está disponible.")

        progress(0.02, "Cargando el clasificador strike/no_strike…")
        candidate = self._load()
        progress(0.10, f"Leyendo como máximo {requested_frames} frames…")
        frames, fps = self._decode_prefix(source, requested_frames)
        starts = window_starts(len(frames))
        if not starts:
            raise ValueError(
                f"El video sólo contiene {len(frames)} frames; se necesitan {WINDOW_FRAMES}."
            )

        offsets = sampled_offsets()
        windows: list[dict[str, object]] = []
        batch_size = 2
        with torch.no_grad():
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start:batch_start + batch_size]
                clips = torch.stack(
                    [clip_tensor([frames[start + offset] for offset in offsets]) for start in batch_starts]
                )
                embeddings = candidate.backbone.embeddings(clips)
                probabilities = candidate.head(embeddings).softmax(1)[:, 1].tolist()
                for start, probability in zip(batch_starts, probabilities):
                    end = start + WINDOW_FRAMES - 1
                    windows.append(
                        {
                            "start_frame": start,
                            "end_frame": end,
                            "start_seconds": start / fps,
                            "end_seconds": end / fps,
                            "p_strike": float(probability),
                            "predicted_label": "strike"
                            if probability >= candidate.threshold
                            else "no_strike",
                        }
                    )
                progress(
                    0.15 + 0.65 * (batch_start + len(batch_starts)) / len(starts),
                    f"Clasificando ventana {batch_start + len(batch_starts)}/{len(starts)}…",
                )

        frame_labels = frame_predictions(windows, len(frames))
        job_id = uuid.uuid4().hex[:12]
        output_dir = self.root / "outputs" / "strike-detection" / "gradio-inference"
        preview_path = output_dir / f"{job_id}_first_{len(frames)}_frames.mp4"
        progress(0.84, "Generando vista previa anotada…")
        self._write_preview(source, preview_path, len(frames), fps, frame_labels)
        result = {
            "schema": "strike_gradio_inference_v1",
            "model_run": candidate.run_name,
            "threshold_fixed_from_validation": candidate.threshold,
            "requested_frames": requested_frames,
            "analyzed_frames": len(frames),
            "fps": fps,
            "windows": windows,
            "strike_windows": sum(item["predicted_label"] == "strike" for item in windows),
            "note": "Ventanas solapadas: no representan un conteo final de golpes.",
        }
        json_path = output_dir / f"{job_id}_results.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        progress(1.0, "Análisis terminado.")
        return {**result, "preview_path": str(preview_path), "json_path": str(json_path)}
