# Anotador visual `strike/no_strike`

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
