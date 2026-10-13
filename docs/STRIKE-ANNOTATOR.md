# Anotador visual `strike/no_strike`

## Segunda etapa: clasificar acciones dentro de los strikes

```powershell
conda activate strikevision_ML
python -s scripts/annotation/run_action_annotator.py
```

La app abre normalmente en `http://127.0.0.1:7864`. Si ya está ocupado, muestra
en la terminal el siguiente puerto local disponible. Selecciona categoría y round,
y pulsa **Cargar strikes**. Se reutilizan las anotaciones humanas positivas y las poses con merge
existentes. El visor recibe solo el fragmento actual, renderizado y cacheado por
demanda; no carga el video completo. Las ventanas positivas solapadas se unen.

Elige atacante ID 1 (**rojo**), ID 2 (**morado**) o desconocido y pulsa numpad **7** punch_head, **9**
punch_body, **4** kick_leg, **5** kick_body, **6** kick_head, **1** other_strike
o **3** ambiguous. Cada pulsación agrega una acción y mantiene abierto el mismo
fragmento. **Terminar fragmento** es la única forma normal de cerrarlo y avanzar. Los
atajos también funcionan con la fila superior de números, pero se desactivan
mientras escribes en un campo de texto o editas frames.

Pulsa numpad **0** para **Terminar fragmento** sin usar el mouse.

Pulsa la categoría cuando veas el ataque. La app pausa inmediatamente el video y,
si los límites todavía cubren todo el fragmento, guarda automáticamente una ventana
de 24 frames (0,8 s) centrada alrededor del punto visible. El reproductor permanece
detenido en ese mismo punto para que puedas verificar la postura o registrar un ataque
simultáneo. Para reemplazar el cálculo, cambia los frames inicial/final o usa los
botones del frame visible antes de anotar. Los límites son inclusivos y usan los
índices originales del round.

Una combinación se guarda como varias filas. Si ambos peleadores atacan a la vez,
guarda una fila por ataque seleccionando el ID correspondiente; los intervalos pueden
solaparse. El atacante seleccionado se conserva hasta terminar el fragmento. Cuando
ya no quede ningún ataque por registrar, pulsa **Terminar fragmento**.

**Deshacer** revierte la última decisión, incluso después de cerrar y abrir la
app. Se retoma el primer fragmento pendiente al cargar el round. Las nuevas
etiquetas se guardan por round en `data/annotations/strike_actions_v1/*.json`,
con atacante, intervalo, categoría y referencias a las ventanas originales.
Las anotaciones binarias originales se conservan. Si estas cambian, el anotador
detiene la escritura para evitar mezclar versiones.

Si la primera anotación marcó un fragmento como `strike` pero al revisarlo no
hay ningún intento ofensivo, pulsa **No strike · anotación previa errónea**. La
app guarda una corrección trazable de ese fragmento y avanza. No uses
`ambiguous` para un falso positivo claro; `ambiguous` se reserva para un ataque
que no puede clasificarse con certeza.

La guía de la derecha recuerda inclusiones y exclusiones. Los rounds con poses
marcadas como poco fiables muestran un aviso: usa atacante desconocido si los
IDs no son correctos. Para probar la interfaz sin introducir etiquetas de prueba
en el dataset, usa `--annotation-dir outputs/action-annotation/ui-test`.

La app local permite etiquetar las ventanas de `strike_annotations_v1` sin editar
JSONL manualmente. Cada decisión se guarda inmediatamente y de forma atómica en
el mismo archivo de anotaciones.

## Ejecutar el piloto

```powershell
conda activate strikevision
python scripts/annotation/run_strike_annotator.py `
  --annotations data/annotations/strike_annotations_v1/topuria_gaethje_round2.jsonl
```

La app abre `http://127.0.0.1:7861`. Reutiliza los videos completos y reproduce
solo el intervalo de cada ventana en el navegador, sin generar un MP4 por
ventana. El JSONL de anotaciones queda intacto hasta que se pulsa una etiqueta.

## Preparar y anotar un split completo

Para abrir Gradio y elegir carpeta + video dentro de `data/splits`:

```powershell
python scripts/annotation/run_strike_annotator.py
```

Eso usa `data/splits` entero. En la interfaz primero eliges la subcarpeta
(`aggressive_men`, `normal_men`, etc.) y después el MP4 que hay dentro.
Al pulsar `Preparar video`, una barra informa el avance de tracking, pose y
render únicamente para ese round. Cuando termina, aparecen las ventanas de
anotación. Cada decisión se guarda en el JSONL propio del video, por lo que se
puede cambiar de carpeta o de round y volver después sin perder progreso. Los
artefactos ya preparados se reutilizan. Si el preview todavía muestra
`fighter_left` / `fighter_right`, vuelve a pulsar **Preparar video**: reescribe
solo el MP4 de preview con IDs `1` y `2` y no toca las anotaciones. Solo existe
un preview H.264 liviano por round; las ventanas no generan archivos de video.

Si quieres limitar el anotador a una sola categoría:

```powershell
python scripts/annotation/run_strike_annotator.py `
  --split-dir data/splits/aggressive_men
```

La configuración actual crea clips de 24 frames (~0,8 s) con un avance de 12
frames (~0,4 s), para reducir trabajo manual sin perder solapamiento temporal.

## Flujo de trabajo

1. Los videos original y de pose se reproducen automáticamente al cargar cada ventana.
   Usa el preview de pose como evidencia técnica.
2. Escribir el nombre del anotador una vez.
3. Añadir una nota solo si hay una situación relevante.
4. Pulsar `Strike`, `No strike` o `No se puede decidir`. Para anotar más rápido,
   usa `O` para `Strike` y `P` para `No strike`.
5. La app guarda y abre la siguiente ventana sin etiqueta.

Los atajos no se aplican mientras escribes en `Anotador` o `Notas`.

La sugerencia técnica de calidad nunca define la etiqueta final. Un caso marcado
como `unknown_occluded` puede revisarse y etiquetarse de otra forma si el video
original permite decidirlo con claridad.

## Validar antes de entrenar

```powershell
python scripts/data/validate_strike_annotations.py `
  data/annotations/strike_annotations_v1/topuria_gaethje_round2.jsonl `
  --require-labeled
```
