# StrikeVision — Estado frente a `strike/no_strike`

> Última revisión: 2026-08-07  
> Base revisada: `96678ae` (`PersonDetector` registrado) → `afee8fd` (HEAD de la revisión)

## Propósito de este documento

Este documento es el contexto operativo de la siguiente misión de StrikeVision:
detectar si, dentro de una ventana temporal de un round, existe una acción de
**striking** (`strike`) o no (`no_strike`).

No intenta reconocer todavía jabs, crosses, hooks, low kicks ni impactos. La
primera salida buscada es una decisión binaria temporal, con evidencia visual y
trazabilidad de datos suficientes para entrenarla y evaluarla.

## Objetivo de producto inmediato

```text
Video de un round
  → dos peleadores detectados y seguidos
  → pose temporal de ambos
  → ventanas candidatas
  → clasificador binario
  → strike / no_strike + confianza + intervalo temporal
```

La unidad de decisión no será un frame aislado: será una ventana corta de frames
con contexto antes, durante y después del movimiento. Un golpe es movimiento,
distancia y relación entre ambos peleadores; un solo frame no contiene esa
información.

## Línea de tiempo relevante

| Fecha | Commit | Entrega | Resultado para la misión |
|---|---|---|---|
| 2026-06-26 | `dff3505` | DVC inicial y compatibilidad | El corpus de rounds puede versionarse sin guardar videos pesados en Git. |
| 2026-07-23 | `bbbb48a` | Desarrollo de person detection | Base YOLO-seg + ByteTrack para localizar personas. |
| 2026-07-24 | `5af06fb` | Utilidad productiva de person detection | Pipeline reutilizable de detección/tracking. |
| 2026-07-30 | `96678ae` | `PersonDetector` en MLflow | Punto de partida: detector UFC registrado como `PersonDetector@production`. |
| 2026-07-31 | `284ffdc` | Pose MediaPipe | Export JSONL, métricas, preview y notebook de pose. |
| 2026-08-02 | `52e94e1` | Unión de fragmentos | ByteTrack fragmentado se reconstruye en dos trayectorias. |
| 2026-08-02 | `afee8fd` | Registro de dos variantes de pose | `PoseEstimator` y `PoseEstimatorMerge` registrados en MLflow. |

## Capacidades que ya existen

### 1. Video y datos

- Los rounds de entrenamiento viven en `data/splits/` y su versionado se apoya
  en DVC.
- Existe un manifiesto de splits en `data/metadata/splits_manifest.csv`.
- Git conserva código, configuración y punteros DVC; los videos y artefactos
  pesados no deben entrar a Git.

### 2. Detección y tracking de peleadores

- `PersonDetector@production` usa YOLO-seg, ByteTrack y filtros específicos UFC
  para seleccionar peleadores y excluir al referee.
- Los registros conservan `frame_index`, `timestamp_seconds`, `track_id`, bbox,
  confianza y visibilidad.
- Antes de pose, el identificador de un fragmento es
  `fighter_track_<bytetrack_id>`.

### 3. Pose MediaPipe

- MediaPipe Pose Landmarker está aprobado como backend de pose.
- Se estima pose sobre el crop de cada peleador y los keypoints se devuelven en
  coordenadas del frame completo.
- Keypoints no visibles se almacenan como `null`, no como coordenadas falsas.
- Se producen `tracking.jsonl`, `pose.jsonl`, `pose_preview.mp4`,
  `pose_metrics.json` y `run_metadata.json`.
- El pipeline genérico está en `src/ufc_tracker/pose/pipeline.py` y se puede
  ejecutar con `scripts/ml/run_pose_pipeline.py`.

### 4. Unión de fragmentos de ByteTrack

La variante merge encadena fragmentos cortos cuando son temporal y espacialmente
compatibles. Produce dos slots temporales: `fighter_left` y `fighter_right`.

Estos nombres son etiquetas técnicas de continuidad, no identidad real, esquina
roja/azul ni asignación definitiva del atleta. Deben revisarse en videos con
cruces, clinch, oclusión y cortes de cámara para asegurar que el slot no cambia
de cuerpo.

### 5. MLflow

El registro local contiene, al momento de esta revisión:

| Modelo | Alias | Versión | Uso |
|---|---:|---:|---|
| `PersonDetector` | `production` | 3 | Detección y ByteTrack UFC. |
| `PoseEstimator` | `production` | 2 | Pose sin unión de fragmentos. |
| `PoseEstimatorMerge` | `production` | 2 | Pose con unión de fragmentos. |

Los modelos de pose registran el peso MediaPipe, parámetros y contrato de
artefactos. La inferencia completa sigue ejecutándose mediante los entry points
del pipeline; el `pyfunc.predict()` del registro documenta/valida el contrato y
la ruta esperada, pero no ejecuta el video completo por sí mismo.

## Evidencia actual

En los primeros 1.000 frames de Fiziev vs. Bahamondes:

| Variante | Trayectorias/fragmentos | Fighter-frames visibles | Cobertura de pose |
|---|---:|---:|---:|
| Pose estándar | 3 fragmentos | 1,835 | 97.93% |
| Pose con merge | 2 slots | 1,873 | 97.44% |

La variante merge mantiene una cobertura equivalente y recupera la propiedad
necesaria para clasificación temporal: dos secuencias en vez de varios IDs de
un mismo peleador.

Validación de código realizada en la revisión:

- `pytest -q`: 5 pruebas superadas.
- `ruff check src scripts tests`: sin errores.
- `dvc status`: datos y pipelines actuales sin cambios pendientes.

## Estado frente a la meta

```text
V0  Video reproducible/anotado visualmente                         COMPLETO
V1  Detección + tracking de los dos peleadores                     COMPLETO
V1.5 Pose temporal y manejo de fragmentos                          COMPLETO CON VALIDACIÓN PENDIENTE
V2  Contrato y dataset anotado de ventanas strike/no_strike        NO INICIADO
V3  Features/candidatos temporales                                 NO INICIADO
V4  Clasificador binario y evaluación por pelea                    NO INICIADO
V5  Registro/promoción de StrikeClassifier                         NO INICIADO
```

La conclusión es directa: el cuello de botella ya no es localizar a los
peleadores. Ahora es crear una verdad de terreno temporal y evitar que el
clasificador aprenda fondos, peleas o frames vecinos en lugar de striking.

## Deuda priorizada

### P0 — necesaria antes de anotar

1. **Aprobar una sola variante de pose.**
   Revisar `PoseEstimatorMerge` en varios rounds de ataque, defensa, distancia,
   clinch breve, oclusión y cortes. Si supera la revisión, debe ser el único
   modo de producción para esta misión.

2. **Validar continuidad de identidad.**
   El merge usa proximidad, gap temporal y posición horizontal media. No tiene
   re-identificación visual del atleta. Debe comprobarse que no intercambie
   peleadores en los intercambios más cerrados.

3. **Versionar resultados aprobados de pose con DVC.**
   Hoy los artefactos generados bajo `data/processed/` y `outputs/` están
   ignorados por Git y no hay punteros DVC de pose en `data/metadata/`. Que
   `dvc status` esté limpio no versiona automáticamente estos nuevos outputs.

4. **Definir el contrato de anotación.**
   Debe fijar el tamaño de ventana, unidad temporal, clase, regla de inicio/fin,
   etiqueta `unknown/occluded`, anotador, versión del video y versión de pose.

### P1 — necesaria para el primer dataset útil

5. **Construir candidatos de alta cobertura.**
   Usar velocidad/aceleración normalizada de muñecas, codos, rodillas y tobillos,
   orientación corporal, distancia entre peleadores y visibilidad. Los
   candidatos reducen trabajo humano; no son aún la etiqueta final.

6. **Crear negativos difíciles.**
   `no_strike` debe incluir guardia, fintas, pasos, defensa, retroceso, clinch
   breve y desplazamientos. Si solo contiene reposo, el primer modelo será
   artificialmente bueno.

7. **Separar train/validation/test por pelea.**
   Nunca por frames aleatorios: frames vecinos del mismo golpe producirían data
   leakage.

### P2 — endurecimiento de ingeniería

8. Añadir pruebas unitarias de `tracking/merge.py`, incluyendo cruces, gaps y
   rechazo de un tercer cuerpo.
9. Añadir prueba de integración del pipeline merge con datos simulados y
   verificación de los contratos JSONL.
10. Cuando exista el clasificador, registrar un `StrikeClassifier` cuyo
    `predict()` realice inferencia, no solo describa una ruta de artefactos.

## Ruta propuesta hacia el primer modelo

### Fase A — congelar la entrada de pose

**Objetivo:** seleccionar y versionar la entrada que alimentará anotación.

1. Ejecutar `PoseEstimatorMerge` sobre 5–8 rounds representativos.
2. Revisar sus previews y registrar fallos de continuidad/pose.
3. Definir criterios de aprobación: dos slots coherentes, extremidades útiles
   en la mayoría de fighter-frames visibles y ausencia de intercambio evidente.
4. Añadir los artefactos aprobados a DVC.

**Salida:** `pose_dataset_v1` reproducible y un solo pipeline de pose aprobado.

### Fase B — contrato y anotación binaria

**Objetivo:** crear verdad de terreno para actividad de striking.

1. Definir una ventana inicial fija y una convención para ventanas solapadas.
2. Etiquetar `strike`, `no_strike` y `unknown/occluded`.
3. Guardar anotaciones por video, intervalo temporal, IDs técnicos de ambos
   peleadores, versión de pose y anotador.
4. Hacer una revisión de calidad de una muestra antes de escalar anotación.

**Salida:** `strike_annotations_v1` versionado con DVC.

### Fase C — candidatos y features temporales

**Objetivo:** transformar la pose en señales comparables entre cámaras y tamaños
de peleador.

1. Normalizar coordenadas por el tamaño de bbox/torso.
2. Calcular trayectorias, velocidad y aceleración por extremidad.
3. Calcular distancia entre peleadores y aproximación de extremidad al cuerpo
   rival.
4. Generar ventanas candidatas de alta sensibilidad y unirlas a etiquetas.

**Salida:** `strike_features_v1` y un dataset de ventanas listo para modelar.

### Fase D — baseline `strike/no_strike`

**Objetivo:** validar que las señales aportan información antes de intentar
reconocer técnicas específicas.

1. Entrenar un baseline temporal sencillo sobre ventanas etiquetadas.
2. Medir precisión, recall, F1, matriz de confusión y errores por categoría de
   escena.
3. Evaluar por pelea completa y conservar un conjunto de prueba no visto.
4. Comparar contra un baseline de heurísticas para saber si el modelo aprende
   algo adicional.

**Salida:** `StrikeClassifier` candidato, métricas y previews de errores.

### Fase E — promoción controlada

**Objetivo:** registrar un modelo que pueda entrar al producto sin contaminar
la evaluación.

1. Registrar datos, parámetros, features, métricas y artefactos en MLflow.
2. Promover a `staging` tras revisión de videos de error.
3. Promover a `production` solo después de una segunda evaluación sobre peleas
   no usadas para entrenar.

## Límites de la misión actual

- No asignar identidades deportivas permanentes ni esquina roja/azul.
- No clasificar técnica específica todavía.
- No interpretar contacto/impacto como requisito para marcar `strike`.
- No integrar el clasificador en Gradio o Streamlit hasta validar el dataset y
  el baseline fuera de la interfaz.

## Próxima acción recomendada

La siguiente tarea concreta es diseñar el contrato de anotación
`strike_annotations_v1` y el protocolo de revisión de `PoseEstimatorMerge`.
Ambos deben aprobarse antes de etiquetar masivamente o entrenar un clasificador.
