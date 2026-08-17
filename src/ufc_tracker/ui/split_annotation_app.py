"""Gradio workflow for preparing and annotating one video from any split folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from ufc_tracker.annotations.contracts import AnnotationConfig
from ufc_tracker.annotations.session import AnnotationMediaResolver, AnnotationStore
from ufc_tracker.annotations.windows import generate_annotation_windows, read_jsonl, write_jsonl
from ufc_tracker.pose.pipeline import (
    ensure_browser_compatible_preview,
    refresh_pose_preview_if_legacy_labels,
    run_pose_pipeline,
)
from ufc_tracker.ui.strike_annotation_app import (
    KEYBOARD_SHORTCUTS_JS,
    StrikeAnnotationApp,
    _play_after,
)


def _videos_in(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.mp4") if path.is_file())


def discover_split_categories(splits_root: Path) -> dict[str, Path]:
    """Map category name -> folder for every split directory that contains MP4s.

    ``splits_root`` can be ``data/splits`` (all categories) or one category
    folder such as ``data/splits/aggressive_men``.
    """
    root = splits_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Splits directory does not exist: {root}")

    categories: dict[str, Path] = {}
    if _videos_in(root):
        categories[root.name] = root
    for child in sorted(root.iterdir()):
        if child.is_dir() and _videos_in(child):
            categories[child.name] = child
    if not categories:
        raise ValueError(f"No MP4 videos found under: {root}")
    return categories


class SplitAnnotationApp:
    """Prepare one selected round and bind it to the annotation controls."""

    def __init__(
        self,
        *,
        root: Path,
        split_dir: Path,
        config: AnnotationConfig,
        pose_config: dict[str, Any],
        pose_root: Path,
        annotation_root: Path,
        media: AnnotationMediaResolver,
    ) -> None:
        self.root = root.resolve()
        self.splits_root = split_dir.resolve()
        self.config = config
        self.pose_config = pose_config
        self.pose_root = pose_root.resolve()
        self.annotation_root = annotation_root.resolve()
        self.media = media
        self.categories = discover_split_categories(self.splits_root)
        self.active: StrikeAnnotationApp | None = None

    @property
    def category_names(self) -> list[str]:
        return list(self.categories)

    def videos_in(self, category: str) -> list[str]:
        folder = self.categories.get(category)
        if folder is None:
            return []
        return [video.name for video in _videos_in(folder)]

    def _video_path(self, category: str, video_name: str) -> Path:
        folder = self.categories.get(category)
        if folder is None:
            raise gr.Error("Selecciona una carpeta válida de splits.")
        video = folder / video_name
        if not video.is_file():
            raise gr.Error("Selecciona un video válido de esa carpeta.")
        return video

    def _pose_dir(self, video: Path) -> Path:
        return self.pose_root / video.parent.name / video.stem

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    @staticmethod
    def _pose_ready(pose_dir: Path) -> bool:
        return all(
            (pose_dir / filename).is_file()
            for filename in (
                "pose.jsonl",
                "pose_metrics.json",
                "run_metadata.json",
                "pose_preview.mp4",
            )
        )

    def _annotation_path(self, video: Path) -> Path:
        return self.annotation_root / f"{video.stem}.jsonl"

    def _generate_annotations(self, video: Path, pose_dir: Path, output: Path) -> None:
        pose_path = pose_dir / "pose.jsonl"
        metrics = json.loads((pose_dir / "pose_metrics.json").read_text(encoding="utf-8"))
        windows = generate_annotation_windows(
            read_jsonl(pose_path),
            config=self.config,
            video_id=video.stem,
            video_path=self._relative(video),
            pose_path=self._relative(pose_path),
            fps=float(metrics.get("fps", 0.0)),
            frame_count=int(metrics.get("frame_count", 0)),
        )
        write_jsonl(output, windows)

    @staticmethod
    def _report_pipeline_progress(
        progress: gr.Progress,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        ranges = {
            "tracking": (0.00, 0.55, "Detectando y haciendo tracking"),
            "pose": (0.55, 0.90, "Calculando pose"),
            "render": (0.90, 1.00, "Generando preview"),
        }
        start, end, description = ranges[stage]
        ratio = current / total if total > 0 else 0.0
        progress(
            min(end, start + ((end - start) * ratio)),
            desc=f"{description}: {current}/{total} frames",
        )

    def select_category(self, category: str) -> tuple[Any, ...]:
        videos = self.videos_in(category)
        if not videos:
            raise gr.Error("Esa carpeta no tiene videos MP4.")
        return (
            gr.update(choices=videos, value=videos[0]),
            (
                f"Carpeta **{category}**. Selecciona un video y pulsa "
                "**Preparar video**."
            ),
            gr.update(visible=False),
        )

    def prepare_video(
        self,
        category: str,
        video_name: str,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[Any, ...]:
        video = self._video_path(category, video_name)
        pose_dir = self._pose_dir(video)
        annotation_path = self._annotation_path(video)
        if not self._pose_ready(pose_dir):
            progress(0.0, desc="Iniciando preparación del video")
            run_pose_pipeline(
                video,
                pose_dir,
                tracking_confidence=float(self.pose_config.get("tracking_confidence", 0.5)),
                min_track_frames=int(self.pose_config.get("min_track_frames", 15)),
                merge_track_fragments=True,
                progress_callback=lambda stage, current, total: self._report_pipeline_progress(
                    progress, stage, current, total
                ),
            )
        else:
            progress(0.90, desc="Reutilizando pose y tracking existentes")
            if refresh_pose_preview_if_legacy_labels(
                video,
                pose_dir,
                progress_callback=lambda current, total: self._report_pipeline_progress(
                    progress, "render", current, total
                ),
            ):
                progress(0.95, desc="Actualizando etiquetas del preview a 1 y 2")

        preview_path = pose_dir / "pose_preview.mp4"
        if ensure_browser_compatible_preview(preview_path):
            progress(0.96, desc="Optimizando preview para el navegador")

        if not annotation_path.is_file():
            progress(0.97, desc="Creando ventanas de anotación")
            self._generate_annotations(video, pose_dir, annotation_path)

        store = AnnotationStore(annotation_path, self.config)
        self.active = StrikeAnnotationApp(store, self.media)
        initial = self.active.window_view(store.next_unlabeled())
        progress(1.0, desc="Video listo para anotar")
        summary = store.summary()
        status = (
            f"**{category}/{video.name}** listo · {summary.labeled}/{summary.total} "
            "ventanas etiquetadas."
        )
        return (
            gr.update(value=initial[0], minimum=0, maximum=max(0, len(store) - 1)),
            initial[1],
            initial[2],
            initial[3],
            initial[4],
            initial[5],
            initial[6],
            initial[7],
            initial[8],
            initial[9],
            status,
            gr.update(visible=True),
        )

    def _require_active(self) -> StrikeAnnotationApp:
        if self.active is None:
            raise gr.Error("Primero selecciona y prepara un video.")
        return self.active

    def window_view(self, index: float | int) -> tuple[Any, ...]:
        return self._require_active().window_state(index)

    def move(self, index: float | int, delta: int) -> tuple[Any, ...]:
        return self._require_active().move(index, delta)

    def next_unlabeled(self, index: float | int) -> tuple[Any, ...]:
        return self._require_active().next_unlabeled(index)

    def save_and_advance(
        self,
        label: str,
        index: float | int,
        annotator: str | None,
        notes: str | None,
    ) -> tuple[Any, ...]:
        return self._require_active().save_and_advance(label, index, annotator, notes)


def build_split_app(controller: SplitAnnotationApp) -> gr.Blocks:
    """Build a folder-then-video annotation interface over ``data/splits``."""
    categories = controller.category_names
    initial_category = categories[0]
    video_names = controller.videos_in(initial_category)
    with gr.Blocks(title="StrikeVision Split Annotator", js=KEYBOARD_SHORTCUTS_JS) as app:
        gr.Markdown(
            """
            # StrikeVision · Anotador por split

            Elige una carpeta de `data/splits` y el video que hay dentro.
            Pulsa **Preparar video**. La barra muestra el progreso de tracking,
            pose y render. Cuando termine, anota con **O = Strike** y
            **P = No strike**. Cada video conserva su propio progreso.
            """
        )
        with gr.Row():
            category_selector = gr.Dropdown(
                choices=categories,
                value=initial_category,
                label="Carpeta de splits",
            )
            video_selector = gr.Dropdown(
                choices=video_names,
                value=video_names[0] if video_names else None,
                label="Video",
            )
            prepare = gr.Button("Preparar video", variant="primary")
        status = gr.Markdown(
            f"Carpeta **{initial_category}**. Selecciona un video y pulsa "
            "**Preparar video**."
        )

        with gr.Group(visible=False) as annotation_group:
            summary = gr.JSON(label="Progreso del video")
            index = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Ventana")
            title = gr.Markdown()
            with gr.Row():
                original = gr.Video(
                    label="Video original",
                    autoplay=False,
                    elem_id="annotation-original",
                )
                pose_preview = gr.Video(
                    label="Tracking y pose",
                    autoplay=False,
                    elem_id="annotation-pose",
                )
            quality = gr.JSON(label="Calidad automática")
            window_start = gr.Number(visible="hidden")
            window_end = gr.Number(visible="hidden")
            with gr.Row():
                annotator = gr.Textbox(label="Anotador")
                notes = gr.Textbox(label="Notas", lines=2)
            with gr.Row():
                previous = gr.Button("← Anterior")
                next_unlabeled = gr.Button("Siguiente sin etiquetar")
                following = gr.Button("Siguiente →")
            with gr.Row():
                strike = gr.Button(
                    "Strike (O)",
                    variant="primary",
                    elem_id="annotation-strike",
                )
                no_strike = gr.Button(
                    "No strike (P)",
                    elem_id="annotation-no-strike",
                )
                unknown = gr.Button("No se puede decidir")

        navigation_outputs = [
            index,
            quality,
            title,
            annotator,
            notes,
            summary,
            window_start,
            window_end,
        ]
        prepared = prepare.click(
            controller.prepare_video,
            inputs=[category_selector, video_selector],
            outputs=[
                index,
                original,
                pose_preview,
                quality,
                title,
                annotator,
                notes,
                summary,
                window_start,
                window_end,
                status,
                annotation_group,
            ],
        )
        _play_after(prepared, window_start, window_end)
        category_selector.change(
            controller.select_category,
            inputs=category_selector,
            outputs=[video_selector, status, annotation_group],
        )
        video_selector.change(
            lambda folder, name: (
                f"Seleccionado **{folder}/{name}**. Pulsa **Preparar video** para cargarlo.",
                gr.update(visible=False),
            ),
            inputs=[category_selector, video_selector],
            outputs=[status, annotation_group],
        )
        changed = index.change(
            controller.window_view,
            inputs=index,
            outputs=navigation_outputs,
            show_progress="hidden",
        )
        _play_after(changed, window_start, window_end)
        moved_previous = previous.click(
            lambda current: controller.move(current, -1),
            inputs=index,
            outputs=navigation_outputs,
            show_progress="hidden",
        )
        _play_after(moved_previous, window_start, window_end)
        moved_following = following.click(
            lambda current: controller.move(current, 1),
            inputs=index,
            outputs=navigation_outputs,
            show_progress="hidden",
        )
        _play_after(moved_following, window_start, window_end)
        moved_unlabeled = next_unlabeled.click(
            controller.next_unlabeled,
            inputs=index,
            outputs=navigation_outputs,
            show_progress="hidden",
        )
        _play_after(moved_unlabeled, window_start, window_end)
        for button, label in (
            (strike, "strike"),
            (no_strike, "no_strike"),
            (unknown, "unknown_occluded"),
        ):
            saved = button.click(
                lambda current, user, comment, selected=label: controller.save_and_advance(
                    selected, current, user, comment
                ),
                inputs=[index, annotator, notes],
                outputs=[*navigation_outputs, status],
                show_progress="hidden",
            )
            _play_after(saved, window_start, window_end)
    return app
