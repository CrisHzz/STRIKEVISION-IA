"""Gradio application for human strike/no_strike annotation."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ufc_tracker.annotations.session import AnnotationMediaCache, AnnotationStore


class StrikeAnnotationApp:
    """Bind one annotation JSONL file to a local Gradio review interface."""

    def __init__(
        self,
        store: AnnotationStore,
        media_cache: AnnotationMediaCache,
    ) -> None:
        self.store = store
        self.media_cache = media_cache

    def window_view(
        self, index: float | int
    ) -> tuple[int, str, str, dict[str, Any], str, str, str, dict[str, int]]:
        safe_index = max(0, min(int(index), len(self.store) - 1))
        row = self.store.row(safe_index)
        original = str(self.media_cache.clip_for(row, "original"))
        pose_preview = str(self.media_cache.clip_for(row, "pose_preview"))
        label = row.get("label") or "Sin etiquetar"
        suggested = row.get("suggested_label") or "ninguna"
        title = (
            f"### Ventana {safe_index + 1}/{len(self.store)} · "
            f"{row['start_seconds']:.3f}s–{row['end_seconds']:.3f}s\n"
            f"`{row['window_id']}` · etiqueta actual: **{label}** · sugerencia: **{suggested}**"
        )
        return (
            safe_index,
            original,
            pose_preview,
            row["quality"],
            title,
            row.get("annotator") or "",
            row.get("notes") or "",
            self.store.summary().to_dict(),
        )

    def save_and_advance(
        self,
        label: str,
        index: float | int,
        annotator: str | None,
        notes: str | None,
    ) -> tuple[int, str, str, dict[str, Any], str, str, str, dict[str, int], str]:
        safe_index = max(0, min(int(index), len(self.store) - 1))
        saved = self.store.save_label(
            safe_index,
            label,
            annotator=annotator,
            notes=notes,
        )
        next_index = self.store.next_unlabeled(safe_index + 1)
        view = self.window_view(next_index)
        status = f"Guardado: {saved['window_id']} → {label}."
        return (*view, status)

    def move(self, index: float | int, delta: int) -> tuple[Any, ...]:
        return self.window_view(int(index) + delta)

    def next_unlabeled(self, index: float | int) -> tuple[Any, ...]:
        return self.window_view(self.store.next_unlabeled(int(index) + 1))


def build_app(store: AnnotationStore, media_cache: AnnotationMediaCache) -> gr.Blocks:
    """Build an isolated local annotation interface for one JSONL file."""
    controller = StrikeAnnotationApp(store, media_cache)
    initial_index = store.next_unlabeled()
    initial = controller.window_view(initial_index)
    maximum_index = max(0, len(store) - 1)

    with gr.Blocks(title="StrikeVision Annotator") as app:
        gr.Markdown(
            """
            # StrikeVision · Anotador `strike/no_strike`

            Revisa el video original y el preview de pose. Marca `strike` si hay una
            ejecución ofensiva clara; `no_strike` si no existe; usa `unknown_occluded`
            cuando el video o la pose no permitan decidir con confianza.
            """
        )
        summary = gr.JSON(value=initial[7], label="Progreso")
        status = gr.Markdown("Listo para anotar.")
        index = gr.Slider(
            minimum=0,
            maximum=maximum_index,
            value=initial[0],
            step=1,
            label="Ventana",
        )
        title = gr.Markdown(initial[4])
        with gr.Row():
            original = gr.Video(value=initial[1], label="Video original", autoplay=False)
            pose_preview = gr.Video(value=initial[2], label="Tracking y pose", autoplay=False)
        quality = gr.JSON(value=initial[3], label="Calidad automática")
        with gr.Row():
            annotator = gr.Textbox(value=initial[5], label="Anotador")
            notes = gr.Textbox(value=initial[6], label="Notas", lines=2)
        with gr.Row():
            previous = gr.Button("← Anterior")
            next_unlabeled = gr.Button("Siguiente sin etiquetar")
            following = gr.Button("Siguiente →")
        with gr.Row():
            strike = gr.Button("Strike", variant="primary")
            no_strike = gr.Button("No strike")
            unknown = gr.Button("No se puede decidir")

        view_outputs = [
            index,
            original,
            pose_preview,
            quality,
            title,
            annotator,
            notes,
            summary,
        ]
        index.change(controller.window_view, inputs=index, outputs=view_outputs)
        previous.click(
            lambda current: controller.move(current, -1),
            inputs=index,
            outputs=view_outputs,
        )
        following.click(
            lambda current: controller.move(current, 1),
            inputs=index,
            outputs=view_outputs,
        )
        next_unlabeled.click(
            controller.next_unlabeled,
            inputs=index,
            outputs=view_outputs,
        )
        for button, label in (
            (strike, "strike"),
            (no_strike, "no_strike"),
            (unknown, "unknown_occluded"),
        ):
            button.click(
                lambda current, user, comment, selected=label: controller.save_and_advance(
                    selected,
                    current,
                    user,
                    comment,
                ),
                inputs=[index, annotator, notes],
                outputs=[*view_outputs, status],
            )
    return app
