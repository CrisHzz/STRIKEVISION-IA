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

La app abre `http://127.0.0.1:7861` y guarda clips temporales reutilizables bajo
`outputs/annotation_clips/`. El JSONL de anotaciones queda intacto hasta que se
pulsa una de las tres etiquetas.

## Flujo de trabajo

1. Reproducir el video original; usar el preview de pose como evidencia técnica.
2. Escribir el nombre del anotador una vez.
3. Añadir una nota solo si hay una situación relevante.
4. Pulsar `Strike`, `No strike` o `No se puede decidir`.
5. La app guarda y abre la siguiente ventana sin etiqueta.

La sugerencia técnica de calidad nunca define la etiqueta final. Un caso marcado
como `unknown_occluded` puede revisarse y etiquetarse de otra forma si el video
original permite decidirlo con claridad.

## Validar antes de entrenar

```powershell
python scripts/data/validate_strike_annotations.py `
  data/annotations/strike_annotations_v1/topuria_gaethje_round2.jsonl `
  --require-labeled
```

