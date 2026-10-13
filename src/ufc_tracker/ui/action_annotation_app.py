"""Single-video action annotation, restricted to existing positive windows."""

import hashlib
import json
import threading
from pathlib import Path

import cv2
import gradio as gr
import yaml

from ufc_tracker.annotations.actions import ActionStore, LABELS
from ufc_tracker.detection.weights import project_root
from ufc_tracker.pose.pipeline import load_pose_records, render_pose_preview

ROOT = project_root(Path(__file__).resolve())
RENDER_LOCK = threading.Lock()

SHORTCUTS_JS = r"""() => {
  if (window.actionKeysInstalled) return;
  window.actionKeysInstalled = true;
  window.actionBusy = false;
  const held = new Set();
  document.addEventListener('keyup', e => held.delete(e.code), true);
  window.addEventListener('blur', () => held.clear());
  document.addEventListener('keydown', e => {
    if (e.repeat || e.isComposing || e.ctrlKey || e.altKey || e.metaKey || held.has(e.code)) return;
    held.add(e.code);
    if (e.target.closest?.('input:not([type="radio"]):not([type="checkbox"]),textarea,select,[contenteditable="true"]')) return;
    const key = e.code.startsWith('Numpad') ? e.code.slice(6) : e.key;
    if (!['7','9','4','5','6','1','3','0'].includes(key)) return;
    e.preventDefault();
    if (window.actionBusy || !document.querySelector('#action-token textarea, #action-token input')?.value) return;
    const node = document.getElementById(key === '0' ? 'action-next' : 'action-key-' + key);
    const button = node?.matches('button') ? node : node?.querySelector('button');
    if (button && !button.disabled) button.click();
  }, true);
  document.addEventListener('click', e => {
    if (!e.target.closest?.('[id^="action-key-"],#action-undo,#action-next,#action-load,#action-prev,#action-no-strike')) return;
    if (window.actionBusy) { e.preventDefault(); e.stopImmediatePropagation(); return; }
    window.actionBusy = true;
  }, true);
  document.addEventListener('loadedmetadata', e => {
    if (e.target.matches?.('#action-video video')) {
      e.target.play().catch(() => {});
    }
  }, true);
}"""
RELEASE_JS = "() => { window.actionBusy = false; }"

ACTION_RANGE_JS = r"""(state, token, attacker, start, end) => {
  if (!token) return [state, token, attacker, start, end];
  const eventId = token.slice(token.indexOf(':') + 1);
  const match = /^f(\d+)-(\d+)$/.exec(eventId);
  const video = document.querySelector('#action-video video');
  if (!match || !video || !Number.isFinite(video.currentTime)) {
    return [state, token, attacker, start, end];
  }
  video.pause();
  const eventStart = Number(match[1]);
  const eventEnd = Number(match[2]);
  if (Number(start) !== eventStart || Number(end) !== eventEnd) {
    return [state, token, attacker, start, end];
  }
  const fps = Number(state?.fps) || 30;
  const center = Math.max(eventStart, Math.min(eventEnd,
    eventStart + Math.round(video.currentTime * fps)));
  let actionStart = Math.max(eventStart, center - 12);
  let actionEnd = Math.min(eventEnd, actionStart + 23);
  actionStart = Math.max(eventStart, actionEnd - 23);
  return [state, token, attacker, actionStart, actionEnd];
}"""

GUIDE = """### Guía de anotación
Clasifica el intento visible aunque falle o sea bloqueado. El objetivo se decide por la trayectoria observable.

| Tecla | Categoría | Incluye | No incluye |
|---|---|---|---|
| 7 | punch_head | Puño hacia cabeza: recto, gancho, uppercut, volado, hammerfist/backfist | Finta, medir distancia, empujar, defensa, codo |
| 9 | punch_body | Puño hacia pecho, abdomen o costillas | Puño a cabeza, agarre o empujón |
| 4 | kick_leg | Patada a muslo, rodilla o pantorrilla | Rodillazo; levantar pierna para bloquear |
| 5 | kick_body | Patada frontal, circular o giratoria hacia torso | Rodillazo, patada alta o baja |
| 6 | kick_head | Patada hacia cabeza, incluso bloqueada por brazos | Patada al torso |
| 1 | other_strike | Codo, rodilla, hombro u otro ataque fuera de las cinco clases | Grappling, defensa o control |
| 3 | ambiguous | No se distingue ataque, objetivo o técnica por oclusión/corte | Técnica rara pero claramente identificable |

**Varias acciones:** cada tecla pausa el video, agrega una acción y mantiene abierto el fragmento en el mismo punto. Si no cambiaste los límites, se guardan automáticamente 24 frames alrededor del frame visible. Puedes editar los frames para reemplazar ese cálculo. Pulsa «Terminar fragmento» cuando no queden ataques.

**Terminar fragmento:** pulsa numpad **0** cuando ya no queden ataques por registrar.

**Atacante:** ID 1 siempre corresponde al tracking rojo e ID 2 al morado. Elige el ID mostrado; se conserva mientras sigas en el fragmento. Usa desconocido si el tracking es dudoso. Dos ataques simultáneos se guardan como filas separadas, con intervalos solapados y su atacante correspondiente.

**Posible falso strike:** usa ambiguous para revisión; no lo fuerces a una técnica.
Usa **No strike · anotación previa errónea** solo cuando confirmes que el
fragmento no contiene ningún intento ofensivo. Esta decisión se guarda como una
corrección de la primera anotación y no como una acción.
"""


class ActionController:
    def __init__(self, root=ROOT, annotation_dir=None):
        self.root = Path(root)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else self.root / 'data/annotations/strike_actions_v1'
        self.config = yaml.safe_load((self.root / 'configs/data/strike_dataset_v1.yaml').read_text(encoding='utf-8'))
        self.sources = {}
        for path in sorted((self.root / self.config['annotations_dir']).glob('*.jsonl')):
            if path.name in self.config.get('excluded_files', {}):
                continue
            try:
                store = ActionStore(path, self.annotation_dir / (path.stem + '.json'))
            except ValueError:
                continue
            video = self.root / store.video
            if video.is_file():
                self.sources.setdefault(video.parent.name, {})[video.name] = path
        if not self.sources:
            raise ValueError('No hay rounds con anotaciones humanas de strike.')

    def store(self, source):
        path = Path(source)
        if path not in [p for group in self.sources.values() for p in group.values()]:
            raise ValueError('Round inválido.')
        return ActionStore(path, self.annotation_dir / (path.stem + '.json'))

    def clip(self, store, event):
        pose_path = self.root / store.pose
        metadata = json.loads((pose_path.parent / 'run_metadata.json').read_text(encoding='utf-8'))
        if metadata.get('tracking', {}).get('component') != 'merged_person_detector':
            raise ValueError('Este round no tiene poses con merge. Prepáralas antes de anotarlo.')
        video = self.root / store.video
        identity = f'{video}:{video.stat().st_mtime_ns}:{pose_path.stat().st_mtime_ns}:{event["event_id"]}'
        output = self.root / 'outputs/action-annotation/clips' / (hashlib.sha256(identity.encode()).hexdigest() + '.mp4')
        with RENDER_LOCK:
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                records = [r for r in load_pose_records(pose_path) if event['start_frame'] <= r.frame_index <= event['end_frame']]
                if not records:
                    raise ValueError('No hay poses para este fragmento.')

                def overlay(frame, index):
                    cv2.putText(frame, f'frame {index}', (12, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)

                render_pose_preview(video, records, output, start_frame=event['start_frame'],
                                    frame_count=event['end_frame'] + 1, frame_overlay=overlay)
        return str(output)

    def view(self, store, index, message='', attacker_value='unknown'):
        data = store.read()
        index = max(0, min(index, len(store.events) - 1))
        event = store.events[index]
        done = len(data['completed']) == len(store.events)
        token = '' if done else f'{data["revision"]}:{event["event_id"]}'
        state = {'source': str(store.source), 'index': index, 'revision': data['revision'],
                 'event_id': event['event_id'], 'fps': store.fps}
        actions = [[a['start_frame'], a['end_frame'], a['attacker'], a['label']] for a in data['actions'] if a['event_id'] == event['event_id']]
        actions.extend([[c['start_frame'], c['end_frame'], '—', 'no_strike · corrección previa']
                        for c in data['source_corrections'] if c['event_id'] == event['event_id']])
        warning = ' · Tracking marcado como poco fiable: verifica IDs o usa desconocido.' if store.video_id in self.config.get('untrusted_pose_videos', {}) else ''
        status = f'{message} · Fragmento {index + 1}/{len(store.events)} · {len(data["completed"])}/{len(store.events)} terminados{warning}'
        if done:
            status = f'{message} · Round terminado. Todas las anotaciones están guardadas. Puedes deshacer la última decisión.'
        return (state, token, None if done else self.clip(store, event),
                gr.update(value=event['start_frame'], minimum=event['start_frame'], maximum=event['end_frame']),
                gr.update(value=event['end_frame'], minimum=event['start_frame'], maximum=event['end_frame']),
                attacker_value, status, actions)

    def load(self, category, video):
        try:
            store = self.store(self.sources[category][video])
            data = store.read()
            index = next((i for i,e in enumerate(store.events) if e['event_id'] not in data['completed']), 0)
            return self.view(store, index, 'Round cargado')
        except (ValueError, OSError, KeyError, RuntimeError) as error:
            return (None, '', None, gr.update(), gr.update(), 'unknown', str(error), [])

    def decide(self, state, token, attacker, start, end, *, label=None, command=None):
        try:
            if not state:
                raise ValueError('Carga un round primero.')
            store = self.store(state['source'])
            if token != f'{state["revision"]}:{state["event_id"]}' and command != 'undo':
                raise ValueError('Este round ya está terminado. Usa deshacer para corregir.')
            if command in ('prev', 'skip'):
                return self.view(store, state['index'] + (-1 if command == 'prev' else 1), 'Navegación sin etiquetar')
            data, changed_event = store.commit(
                state['revision'], state['event_id'], label=label, attacker=attacker,
                start=start, end=end,
                complete=command in ('next', 'correction'),
                undo=command == 'undo', source_correction=command == 'correction',
            )
            index = state['index']
            if command == 'undo':
                index = next(i for i,e in enumerate(store.events) if e['event_id'] == changed_event)
            elif command in ('next', 'correction'):
                index = next((i for i in list(range(index+1,len(store.events))) + list(range(index+1))
                              if store.events[i]['event_id'] not in data['completed']), index)
            if command == 'undo':
                message = 'Deshecho'
            elif label is not None:
                message = 'Acción guardada · revisa el fragmento y pulsa Terminar fragmento al finalizar'
            elif command == 'correction':
                message = 'Corrección no_strike guardada'
            else:
                message = 'Fragmento terminado'
            result = list(self.view(store, index, message, attacker if label is not None else 'unknown'))
            if label is not None:
                # Keep the existing DOM video, its paused state and its playhead. Replacing
                # the same source makes Gradio reload and autoplay the clip from the start.
                result[2] = gr.update()
            return tuple(result)
        except (ValueError, OSError, KeyError, StopIteration, RuntimeError) as error:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f'No se completó la operación: {error}', gr.update())


def build_app(root=ROOT, annotation_dir=None):
    controller = ActionController(root, annotation_dir)
    categories = list(controller.sources)
    with gr.Blocks(title='StrikeVision · Anotación de acciones', js=SHORTCUTS_JS,
                   css='#action-token {display:none} [id^="action-key-"] {min-width:110px} #action-guide th:first-child, #action-guide td:first-child {white-space:nowrap} #action-guide th:nth-child(2), #action-guide td:nth-child(2) {min-width:100px; word-break:normal; overflow-wrap:normal}') as app:
        gr.Markdown('# Anotación de acciones · Solo momentos de strike')
        state = gr.State(None)
        token = gr.Textbox(value='', elem_id='action-token', interactive=False)
        with gr.Row():
            category = gr.Dropdown(categories, value=categories[0], label='Categoría de videos')
            video = gr.Dropdown(list(controller.sources[categories[0]]), value=next(iter(controller.sources[categories[0]])), label='Pelea / round')
            load = gr.Button('Cargar strikes', elem_id='action-load', variant='primary')
        with gr.Row():
            with gr.Column(scale=3):
                preview = gr.Video(label='Fragmento strike · Pose + merge', interactive=False, elem_id='action-video', autoplay=True)
                status = gr.Markdown('Selecciona un round. No se procesarán intervalos no_strike.')
                attacker = gr.Radio(
                    [('ID 1 (rojo)', '1'), ('ID 2 (morado)', '2'), ('Desconocido', 'unknown')],
                    value='unknown',
                    label='Atacante',
                )
                with gr.Row():
                    start = gr.Number(label='Frame inicial de la acción (inclusive)', precision=0)
                    end = gr.Number(label='Frame final de la acción (inclusive)', precision=0)
                with gr.Row():
                    mark_start = gr.Button('Inicio en frame visible')
                    mark_end = gr.Button('Fin en frame visible')
                controls = {}
                for keys in (('7','9'), ('4','5','6'), ('1','3')):
                    with gr.Row():
                        for key in keys:
                            controls[key] = gr.Button(f'{key} · {LABELS[key]}', elem_id=f'action-key-{key}')
                no_strike = gr.Button(
                    'No strike · anotación previa errónea',
                    elem_id='action-no-strike',
                    variant='stop',
                )
                with gr.Row():
                    prev = gr.Button('Anterior', elem_id='action-prev')
                    undo = gr.Button('Deshacer', elem_id='action-undo')
                    finish = gr.Button('Terminar fragmento', elem_id='action-next')
                actions = gr.Dataframe(headers=['Inicio','Fin','Atacante','Categoría'], interactive=False, label='Acciones guardadas en este fragmento')
            with gr.Column(scale=2):
                gr.Markdown(GUIDE, elem_id='action-guide')
        outputs = [state, token, preview, start, end, attacker, status, actions]
        inputs = [state, token, attacker, start, end]
        category.change(lambda c: gr.update(choices=list(controller.sources[c]), value=next(iter(controller.sources[c]))), category, video)
        load.click(controller.load, [category,video], outputs, concurrency_id='actions', concurrency_limit=1).then(fn=None, js=RELEASE_JS)
        from functools import partial
        for key, button in controls.items():
            button.click(partial(controller.decide,label=LABELS[key]), inputs, outputs, concurrency_id='actions', concurrency_limit=1,
                         trigger_mode='once', js=ACTION_RANGE_JS).then(fn=None, js=RELEASE_JS)
        for button, command in ((prev,'prev'),(undo,'undo'),(finish,'next'),(no_strike,'correction')):
            button.click(partial(controller.decide,command=command), inputs, outputs, concurrency_id='actions', concurrency_limit=1).then(fn=None, js=RELEASE_JS)
        for button, field in ((mark_start,start),(mark_end,end)):
            button.click(fn=None, inputs=[state,token,start], outputs=field, js="""(state, token, fallback) => {
              const v = document.querySelector('#action-video video');
              if (!v || !token) return fallback;
              const range = token.split(':')[1].slice(1).split('-').map(Number);
              const fps = Number(state?.fps) || 30;
              return Math.min(range[1], range[0] + Math.floor(v.currentTime * fps));
            }""")
    return app
