"""Gradio application for human strike/no_strike annotation."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ufc_tracker.annotations.session import AnnotationMediaResolver, AnnotationStore


KEYBOARD_SHORTCUTS_JS = """
() => {
  if (window.strikeVisionShortcutsInstalled) return;
  window.strikeVisionShortcutsInstalled = true;
  window.strikeVisionLastShortcut = 0;
  window.strikeVisionWindowEnd = null;

  document.addEventListener("timeupdate", (event) => {
    const video = event.target;
    const end = Number(window.strikeVisionWindowEnd);
    if (
      video instanceof HTMLVideoElement
      && Number.isFinite(end)
      && video.currentTime >= end - 0.015
    ) video.pause();
  }, true);

  document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented || event.isComposing || event.repeat) return;
    const target = event.target;
    if (
      target instanceof HTMLElement
      && target.closest("input, textarea, select, [contenteditable='true']")
    ) return;

    const shortcut = (event.key || event.code || "").toLowerCase();
    const buttonId = {
      o: "annotation-strike",
      keyo: "annotation-strike",
      p: "annotation-no-strike",
      keyp: "annotation-no-strike",
    }[shortcut];
    if (!buttonId) return;

    const now = Date.now();
    if (now - window.strikeVisionLastShortcut < 300) return;
    const control = document.getElementById(buttonId);
    const button = control?.matches("button") ? control : control?.querySelector("button");
    if (!button || button.disabled) return;
    window.strikeVisionLastShortcut = now;
    event.preventDefault();
    button.click();
  }, true);
}
"""


PLAY_WINDOW_JS = """
(startSeconds, endSeconds) => {
  const start = Number(startSeconds);
  const end = Number(endSeconds);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;

  window.strikeVisionPlaybackId = (window.strikeVisionPlaybackId || 0) + 1;
  window.strikeVisionWindowEnd = end;
  const playbackId = window.strikeVisionPlaybackId;
  const selectors = ["#annotation-original video", "#annotation-pose video"];

  for (const selector of selectors) {
    const video = document.querySelector(selector);
    if (!video) continue;

    const playRange = () => {
      if (playbackId !== window.strikeVisionPlaybackId) return;
      video.currentTime = start;
      const promise = video.play();
      if (promise) promise.catch(() => {});

      if (typeof video.requestVideoFrameCallback === "function") {
        const stopAtEnd = () => {
          if (playbackId !== window.strikeVisionPlaybackId || video.paused) return;
          if (video.currentTime >= end - 0.015) video.pause();
          else video.requestVideoFrameCallback(stopAtEnd);
        };
        video.requestVideoFrameCallback(stopAtEnd);
      }
    };

    if (video.readyState >= 1) playRange();
    else video.addEventListener("loadedmetadata", playRange, { once: true });
  }
}
"""


class StrikeAnnotationApp:
    """Bind one annotation JSONL file to a local Gradio review interface."""

    def __init__(
        self,
        store: AnnotationStore,
        media: AnnotationMediaResolver,
    ) -> None:
        self.store = store
        self.media = media

    def _row(self, index: float | int) -> tuple[int, dict[str, Any]]:
        safe_index = max(0, min(int(index), len(self.store) - 1))
        return safe_index, self.store.row(safe_index)

    def _metadata(
        self, safe_index: int, row: dict[str, Any]
    ) -> tuple[int, dict[str, Any], str, str, str, dict[str, int], float, float]:
        label = row.get("label") or "Sin etiquetar"
        suggested = row.get("suggested_label") or "ninguna"
        title = (
            f"### Ventana {safe_index + 1}/{len(self.store)} · "
            f"{row['start_seconds']:.3f}s–{row['end_seconds']:.3f}s\n"
            f"`{row['window_id']}` · etiqueta actual: **{label}** · "
            f"sugerencia: **{suggested}**"
        )
        return (
            safe_index,
            row["quality"],
            title,
            row.get("annotator") or "",
            row.get("notes") or "",
            self.store.summary().to_dict(),
            float(row["start_seconds"]),
            float(row["end_seconds"]),
        )

    def window_state(self, index: float | int) -> tuple[Any, ...]:
        safe_index, row = self._row(index)
        return self._metadata(safe_index, row)

    def window_view(self, index: float | int) -> tuple[Any, ...]:
        safe_index, row = self._row(index)
        return (
            safe_index,
            str(self.media.source_for(row, "original")),
            str(self.media.source_for(row, "pose_preview")),
            *self._metadata(safe_index, row)[1:],
        )

    def save_and_advance(
        self,
        label: str,
        index: float | int,
        annotator: str | None,
        notes: str | None,
    ) -> tuple[Any, ...]:
        safe_index, _ = self._row(index)
        saved = self.store.save_label(
            safe_index,
            label,
            annotator=annotator,
            notes=notes,
        )
        next_index = self.store.next_unlabeled(safe_index + 1)
        status = f"Guardado: {saved['window_id']} → {label}."
        return (*self.window_state(next_index), status)

    def move(self, index: float | int, delta: int) -> tuple[Any, ...]:
        return self.window_state(int(index) + delta)

    def next_unlabeled(self, index: float | int) -> tuple[Any, ...]:
        return self.window_state(self.store.next_unlabeled(int(index) + 1))


def _play_after(
    dependency: gr.events.Dependency,
    window_start: gr.Number,
    window_end: gr.Number,
) -> None:
    dependency.then(
        fn=None,
        inputs=[window_start, window_end],
        outputs=None,
        js=PLAY_WINDOW_JS,
        show_progress="hidden",
    )


def build_app(store: AnnotationStore, media: AnnotationMediaResolver) -> gr.Blocks:
    """Build an isolated local annotation interface for one JSONL file."""
    controller = StrikeAnnotationApp(store, media)
    initial = controller.window_view(store.next_unlabeled())
    maximum_index = max(0, len(store) - 1)

    with gr.Blocks(title="StrikeVision Annotator", js=KEYBOARD_SHORTCUTS_JS) as app:
        gr.Markdown(
            """
            # StrikeVision · Anotador `strike/no_strike`

            Los videos completos se cargan una sola vez. Cada ventana se reproduce
            mediante búsqueda temporal en el navegador. Atajos: **O = Strike** y
            **P = No strike**.
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
            gr.Video(
                value=initial[1],
                label="Video original",
                autoplay=False,
                elem_id="annotation-original",
            )
            gr.Video(
                value=initial[2],
                label="Tracking y pose",
                autoplay=False,
                elem_id="annotation-pose",
            )
        quality = gr.JSON(value=initial[3], label="Calidad automática")
        window_start = gr.Number(value=initial[8], visible="hidden")
        window_end = gr.Number(value=initial[9], visible="hidden")
        with gr.Row():
            annotator = gr.Textbox(value=initial[5], label="Anotador")
            notes = gr.Textbox(value=initial[6], label="Notas", lines=2)
        with gr.Row():
            previous = gr.Button("← Anterior")
            next_unlabeled = gr.Button("Siguiente sin etiquetar")
            following = gr.Button("Siguiente →")
        with gr.Row():
            strike = gr.Button("Strike (O)", variant="primary", elem_id="annotation-strike")
            no_strike = gr.Button("No strike (P)", elem_id="annotation-no-strike")
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
        changed = index.change(
            controller.window_state,
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
        app.load(
            fn=None,
            inputs=[window_start, window_end],
            outputs=None,
            js=PLAY_WINDOW_JS,
            show_progress="hidden",
        )
    return app
