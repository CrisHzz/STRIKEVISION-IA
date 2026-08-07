# Contrato de anotación `strike_annotations_v1`

## Objetivo

`strike_annotations_v1` define la verdad de terreno temporal para decidir si
existe una acción ofensiva de striking dentro de una ventana corta de video.
No clasifica todavía la técnica ni exige que exista contacto.

La unidad de anotación es una ventana de 24 frames con desplazamiento de 6
frames. En video de 24 FPS representa 1 segundo de contexto y una nueva decisión
cada 0,25 segundos.

## Etiquetas

### `strike`

Existe al menos un intento ofensivo claro de golpe con brazo o pierna dentro de
la ventana. Incluye puños, patadas, rodillas, codos y golpes durante clinch o
suelo cuando la acción sea visualmente distinguible. No requiere impacto.

La fase de ejecución debe ser visible: extensión/aceleración de una extremidad
hacia el rival. La guardia o la preparación sin ejecución no bastan.

### `no_strike`

No existe una ejecución ofensiva de striking. Incluye guardia, espera,
desplazamiento, retroceso, defensa, bloqueo, esquiva, cambio de nivel, clinch sin
golpe y fintas que no llegan a una ejecución clara.

Los negativos difíciles deben conservarse; no se debe formar esta clase solo
con momentos de reposo.

### `unknown_occluded`

No puede decidirse de forma fiable por oclusión, corte de cámara, cuerpos fuera
de cuadro, una caja que contiene a ambos peleadores, pose insuficiente o una
transición ambigua de grappling. Esta clase se excluye del entrenamiento binario
inicial, pero se conserva para auditoría.

## Estado de anotación y sugerencias automáticas

Una ventana recién generada tiene `label: null`. El pipeline puede sugerir
`unknown_occluded` mediante `suggested_label`, pero nunca asigna automáticamente
`strike` ni `no_strike`.

El anotador revisa el video y decide la etiqueta final. La sugerencia automática
se basa únicamente en calidad de tracking/pose y no representa actividad de
striking.

## Esquema JSONL

Cada línea contiene una ventana:

```json
{
  "schema_version": "strike_annotations_v1",
  "window_id": "topuria_gaethje_round2__f000000-f000023",
  "video_id": "topuria_gaethje_round2",
  "video_path": "data/splits/category/round.mp4",
  "pose_path": "outputs/run/pose.jsonl",
  "pose_version": "pose_dataset_v1",
  "fps": 24.0,
  "start_frame": 0,
  "end_frame": 23,
  "start_seconds": 0.0,
  "end_seconds": 0.958333,
  "frame_count": 24,
  "label": null,
  "suggested_label": "unknown_occluded",
  "quality": {
    "status": "auto_unknown",
    "two_fighter_frame_ratio": 0.5,
    "two_pose_valid_frame_ratio": 0.5,
    "required_keypoint_ratio": 0.7,
    "reasons": ["insufficient_two_fighter_coverage"]
  },
  "annotator": null,
  "annotated_at": null,
  "notes": ""
}
```

Los intervalos de frames son inclusivos. `end_seconds` corresponde al timestamp
del último frame, no al instante posterior a la ventana.

El generador conserva únicamente ventanas completas. Si los frames finales no
alcanzan para formar 24 frames, quedan fuera de esta versión del dataset en vez
de producir una ventana de tamaño diferente.

## Reglas de calidad automática

La configuración inicial marca una ventana como candidata a
`unknown_occluded` cuando ocurre cualquiera de estas condiciones:

- Menos del 75 % de frames contiene dos peleadores visibles.
- Menos del 75 % contiene pose válida para ambos peleadores.
- Menos del 55 % de los keypoints requeridos está disponible.

Estas reglas son de control de calidad, no reglas semánticas de striking.

## Procedimiento del anotador

1. Reproducir la ventana completa y, si hace falta, revisar algunos frames antes
   y después.
2. Marcar `strike` si una ejecución ofensiva ocurre dentro de la ventana.
3. Marcar `no_strike` solo cuando exista evidencia visual suficiente de que no
   ocurrió una ejecución.
4. Marcar `unknown_occluded` cuando no sea posible decidir con confianza.
5. Registrar una nota breve en casos dudosos, clinch, corte o caja compartida.

Las ventanas solapadas se anotan independientemente. Un mismo golpe puede
producir varias ventanas positivas; el postprocesamiento unirá predicciones
consecutivas durante inferencia.

## Control de calidad del dataset

- Revisar manualmente al menos el 10 % de las ventanas anotadas.
- No permitir `label: null` en datasets usados para entrenamiento.
- No cambiar una etiqueta por la sugerencia automática sin mirar el video.
- Separar train, validation y test por `fight_id`, nunca por frames o ventanas.
- Conservar `unknown_occluded`, aunque se excluya del baseline binario.
- Versionar anotaciones y configuración con DVC antes de entrenar.
