# Preparación local del dataset strike/no_strike

Revisión: 2026-09-03. Esta etapa prepara datos; no entrena modelos ni usa Azure.

## Selección explícita

La política ejecutable está en `configs/data/strike_dataset_v1.yaml`.
Se excluyen del índice, sin borrar sus archivos:

- `yan_sandhagen__petr_yan_vs_cory_sandhagen__normal_men_round2.jsonl`:
  round sin etiquetas, descartado por decisión del usuario.
- `topuria_gaethje_round2.jsonl`: piloto histórico; se utiliza el round completo.

Quedan 54 rounds de 13 peleas: 40.480 ventanas binarias (11.946 strike y
28.534 no_strike) y 406 unknown_occluded, conservadas por separado.
Los futuros entrenamientos deben consumir `binary_windows.jsonl`, no concatenar
todos los JSONL de anotaciones por su cuenta.

## Generación y verificación

```powershell
conda activate strikevision
python scripts/data/prepare_strike_dataset.py --refresh-quality
python -m pytest -q
```

Salidas en `data/processed/datasets/strike_dataset_v1/`:

- `binary_windows.jsonl`: todos los positivos/negativos humanos incluidos,
  incluso los de baja calidad de pose.
- `review_windows.jsonl`: unknown_occluded; nunca se reasignan a no_strike.
- `audit.json`: selección, recuentos, hashes de poses/anotaciones/configuración y
  métricas de calidad recalculadas. No se ha asignado train/validation/test.

El comando valida fuentes, intervalos, FPS, duplicados por fuente/intervalo y
contratos. `--refresh-quality` modifica exclusivamente `quality` y
`suggested_label`, con copia previa y reemplazo atómico. Todos los demás campos,
incluidos label, notas, anotador, fechas, orden e intervalos, deben ser idénticos.
Sin el flag, se generan los índices derivados sin modificar anotaciones fuente.

## Poses incompletas no equivalen a etiquetas incorrectas

La calidad describe los esqueletos extraídos, no la certeza de la etiqueta humana.
`auto_unknown` es un estado automático de calidad: NO es la clase del ejemplo.
Se conservan los ejemplos binarios de baja calidad y se marcan puntos ausentes
mediante máscara y NaN, no coordenadas ficticias o ceros interpretables como pose.

## Contrato del cargador temporal

`ufc_tracker.annotations.dataset.load_pose_window`:

- Normaliza `fighter_left`/`1` a `fighter_0` y `fighter_right`/`2` a `fighter_1`.
  No reordena por coordenadas ni afirma conocer la identidad deportiva.
- Conserva 24 frames de cada intervalo por defecto, con FPS y tiempo reales.
- Distingue duración cubierta (`24/fps`) y separación entre primera/última
  muestra (`23/fps`). No modifica start_seconds/end_seconds originales.
- Devuelve posiciones en píxeles y velocidades explícitas en píxeles/segundo.
  La normalización corporal y compensación de cámara son trabajo posterior de
  features, no parte de esta reparación.
- El muestreo opcional usa solo frames existentes dentro del intervalo anotado.
  No añade contexto ni interpola puntos ausentes. Frames repetidos tienen dt=0
  y no producen una derivada válida.
- No deriva velocidad cruzando puntos ausentes o cambios de track_id.
  Esto no garantiza detectar todos los cortes de cámara o intercambios de atleta.

## Reparación de los cinco rounds Adesanya-Pereira

El flujo de reparación mantiene artefactos anteriores y anotaciones bajo
`outputs/dataset-repair-20260903/`. Los scripts son locales y reanudables:

1. `stage_pose_repairs.py --skip-preview`: snapshot de anotaciones, reprocesamiento temporal y
   comparación de frame_count/FPS/cobertura. Nunca instala automáticamente.
2. `review_pose_repairs.py --round N`: imágenes de muestra antes/después,
   dibujadas desde los JSONL sobre el video original, sin codificar videos completos.
3. Tras revisar las muestras, `install_pose_repair.py --round N --visual-approved`:
   comprueba mejora cuantitativa y ausencia de cambios concurrentes, conserva
   los artefactos anteriores en `poses_before/` y sustituye el round.
4. Volver a ejecutar `prepare_strike_dataset.py --refresh-quality` para que las
   métricas e índices correspondan a las poses instaladas.

No ejecutar la actualización mientras otra sesión está anotando. Las comprobaciones
de hash detectan cambios concurrentes, pero no sustituyen un bloqueo entre procesos.

El estado de cada round queda en `staged_comparison.json`. Las hojas de contacto
son una inspección de muestras, no una medida de exactitud contra ground truth.
Mayor cobertura de dos cuerpos no prueba por sí sola identidad correcta.

R3 y R4 no superan la inspección visual: tanto la pose antigua como la regenerada
incluyen árbitro/espectadores en varias muestras. El resultado nuevo se rechaza,
sin reemplazarlo como si estuviera corregido. `untrusted_pose_videos` marca ese
ambos rounds: sus 1.500 etiquetas siguen en el índice binario para poder utilizar el video,
pero `pose_usable=false` impide que el cargador entregue esqueletos de terceros
como datos fiables (devuelve máscaras falsas y NaN). El error de identidad del
detector/merge permanece pendiente; no se afirma haberlo solucionado.

R5 también se rechaza: la nueva recuperación introduce espectadores en clinch y
al árbitro al final. Se conserva su pose original con huecos; en las muestras
originales revisadas no se observó esa incorporación de terceros. No se aplica
el bloqueo total de pose de R3/R4: siguen vigentes las máscaras de puntos ausentes.

R1 y R2 sí se instalaron tras mejorar la cobertura y superar la revisión de muestras.
Cobertura simultánea de pose: R1 4.660/9.014 a 8.161/9.014; R2 1.711/9.068 a
6.871/9.068. Estas cifras miden disponibilidad de dos poses, NO exactitud de identidad.

Por petición del usuario, las reconstrucciones temporales R3-R5 se procesaron sin
preview completo (`preview.generated=false`). No se instalaron al fallar la
revisión visual; sus artefactos originales permanecen en su lugar. No se asocian
previews antiguos con poses nuevas.
Si se vuelve a abrir Gradio, el preview se puede generar entonces usando las poses
guardadas, sin repetir YOLO/MediaPipe. El índice de entrenamiento no exige previews.

## Trabajo deliberadamente pendiente

No se han entrenado clasificadores ni cambiado pesos YOLO/MediaPipe, aliases del
registry o recursos de nube. Antes de modelar, fijar particiones por pelea,
normalización aprendida solo en entrenamiento y métricas por calidad/categoría.
