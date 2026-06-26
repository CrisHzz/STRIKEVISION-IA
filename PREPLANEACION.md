# UFC Striking Tracker - Preplaneacion temporal

Este documento es para uso personal durante el desarrollo. No esta pensado todavia como documentacion publica del portafolio, sino como una guia para mantener claro el flujo del proyecto, el alcance y los siguientes pasos.

## Objetivo del proyecto

Crear un sistema de computer vision para analizar videos de peleas de UFC o striking, enfocado solamente en golpes de pie. El sistema debe permitir subir un video de un round y devolver el mismo video anotado con:

- Tracking de los dos peleadores.
- Tracking del referee.
- Identidad persistente para cada persona detectada.
- Deteccion y conteo de golpes de striking:
  - Jab
  - Cross
  - Hook
  - Uppercut
  - Low kick
  - Body kick
  - Head kick

El proyecto no va a cubrir grappling, clinch prolongado, wrestling, submissions ni control en piso durante la primera version.

## Alcance actual

La unidad principal de analisis sera un round individual de maximo 5 minutos.

Esto significa:

- El sistema recibe un video que representa un solo round.
- La duracion maxima esperada es de 5 minutos.
- El video debe tener dos peleadores y un referee.
- El foco inicial es contar golpes lanzados, no golpes conectados.
- Se excluyen replays, pausas largas, esquinas, entrevistas y cortes fuera del round.

La fuente del dataset puede ser una pelea completa, incluso de 5 rounds. Pero esas peleas deben segmentarse antes:

```text
fight_001_full.mp4
 -> fight_001_round_01.mp4
 -> fight_001_round_02.mp4
 -> fight_001_round_03.mp4
 -> fight_001_round_04.mp4
 -> fight_001_round_05.mp4
```

Decision importante:

```text
Unidad de recoleccion de datos: pelea completa
Unidad de entrenamiento/evaluacion: round o clip derivado de un round
Unidad de entrada del producto: un round de maximo 5 minutos
```

## Tipo de proyecto de machine learning

Este proyecto no es un solo modelo aislado. Es un sistema de computer vision compuesto por varios modelos y modulos que trabajan juntos.

Clasificacion del proyecto:

```text
Area general: Computer Vision
Tecnica principal: Deep Learning
Paradigma principal: Supervised Learning
Tipo de problema: Video Understanding / Action Recognition / Temporal Action Classification
```

No es principalmente un proyecto de unsupervised learning. Para clasificar golpes como jab, cross, hook, uppercut o patadas, necesito ejemplos etiquetados.

Ejemplo:

```text
clip_001 -> jab
clip_002 -> cross
clip_003 -> low_kick
clip_004 -> no_strike
```

La descripcion profesional del proyecto seria:

```text
Sistema supervisado de computer vision y deep learning para analisis de striking en video, usando deteccion de personas, multi-object tracking, pose estimation y clasificacion temporal de acciones.
```

## Arquitectura ML general

La arquitectura debe ser modular. No conviene intentar resolver todo con un modelo gigante que reciba el video completo y prediga todos los golpes directamente.

Arquitectura objetivo:

```text
Video
 -> Person Detection
 -> Multi Object Tracking
 -> Role Assignment
 -> Pose Estimation
 -> Feature Engineering
 -> Strike Candidate Detection
 -> Strike Classification
 -> Event Post-processing
 -> Annotated Video + JSON
```

Componentes:

| Componente | Puede usar ML | Funcion |
|---|---:|---|
| Detector de personas | Si | Encuentra personas en cada frame |
| Tracker | No necesariamente | Mantiene IDs entre frames |
| Role assignment | Reglas o ML | Decide quien es peleador y quien referee |
| Pose estimation | Si | Extrae keypoints corporales |
| Strike candidate detector | Reglas o ML | Encuentra ventanas sospechosas de golpe |
| Strike classifier | Si | Clasifica jab, cross, hook, uppercut o patada |
| Post-processing | No | Limpia duplicados y une eventos |

El modelo que probablemente entrenare de forma mas especifica sera el clasificador temporal de golpes. Los detectores de persona, pose y tracking pueden iniciar con modelos o algoritmos preentrenados.

## Flujo esperado del sistema

```text
Usuario sube video de un round
 -> API recibe el archivo
 -> se crea un job de procesamiento
 -> se extraen frames del video
 -> se detectan personas
 -> se separan roles: fighter_red, fighter_blue, referee
 -> se mantiene tracking frame a frame
 -> se estima pose de los peleadores
 -> se detectan posibles eventos de striking
 -> se clasifican los golpes
 -> se agregan anotaciones visuales al video
 -> se genera video final procesado
 -> se devuelve video anotado + resumen JSON
```

## Salida esperada

El sistema debe retornar dos cosas:

1. Un video anotado.
2. Un archivo JSON con eventos y resumen.

Ejemplo de salida JSON:

```json
{
  "video_id": "fight_001",
  "annotated_video_path": "outputs/fight_001_annotated.mp4",
  "summary": {
    "fighter_red": {
      "jab": 12,
      "cross": 8,
      "hook": 5,
      "uppercut": 1,
      "low_kick": 7,
      "body_kick": 2,
      "head_kick": 1
    },
    "fighter_blue": {
      "jab": 9,
      "cross": 6,
      "hook": 4,
      "uppercut": 0,
      "low_kick": 5,
      "body_kick": 1,
      "head_kick": 0
    }
  },
  "events": [
    {
      "frame_start": 1240,
      "frame_end": 1265,
      "timestamp": 41.3,
      "fighter_id": "fighter_red",
      "event": "jab",
      "confidence": 0.87
    }
  ]
}
```

## Modulos principales

### 1. Ingestion de video

Responsable de recibir el video, guardarlo y crear un job de procesamiento.

Tecnologias posibles:

- FastAPI
- PostgreSQL
- almacenamiento local al inicio
- Azure Blob Storage mas adelante

Lo que debo aprender:

- Manejo de archivos grandes en FastAPI.
- Jobs asincronos.
- Separar request HTTP del procesamiento pesado.

### 2. Extraccion de frames

Responsable de convertir el video en frames procesables.

Tecnologias posibles:

- OpenCV
- FFmpeg

Lo que debo aprender:

- FPS, resolucion, codecs y timestamps.
- Como reconstruir un video desde frames.
- Como mantener sincronizados frames y eventos.

### 3. Deteccion de personas

Responsable de detectar las personas visibles en cada frame.

El sistema debe detectar:

- Peleador 1
- Peleador 2
- Referee

Lo que debo aprender:

- Object detection.
- Bounding boxes.
- Confianza de deteccion.
- Diferencia entre detectar personas y asignar roles.

### 4. Asignacion de roles

No basta con detectar tres personas. El sistema debe decidir quien es cada uno:

- `fighter_red`
- `fighter_blue`
- `referee`

Reglas iniciales posibles:

- Los dos peleadores suelen estar mas cerca entre si.
- El referee normalmente no participa en intercambios de golpes.
- Los peleadores tienen movimiento ofensivo/defensivo mas frecuente.
- Si hay colores de esquina o shorts, pueden ayudar a mantener identidad.

Lo que debo aprender:

- Tracking por identidad.
- Re-identification.
- Heuristicas temporales.
- Errores comunes de identity switch.

### 5. Tracking

Responsable de mantener la identidad de cada persona durante todo el video.

Salida esperada:

```text
frame 100:
  fighter_red -> bbox, mask, pose
  fighter_blue -> bbox, mask, pose
  referee -> bbox
```

Tecnologias posibles:

- ByteTrack
- DeepSORT
- SORT como baseline

Lo que debo aprender:

- Track IDs.
- Asociacion entre frames.
- Oclusiones.
- Cambios de camara.
- Metricas como IDF1 y MOTA.

### 6. Pose estimation

Responsable de extraer puntos corporales de los peleadores.

Puntos importantes:

- hombros
- codos
- munecas
- cadera
- rodillas
- tobillos

Tecnologias posibles:

- MediaPipe Pose
- MoveNet
- modelos pose compatibles con TensorFlow

Lo que debo aprender:

- Keypoints.
- Normalizacion por escala del cuerpo.
- Angulos articulares.
- Velocidad y aceleracion de extremidades.

### 7. Deteccion inicial de golpes

Primera version: contador basico basado en reglas.

Ejemplos:

- Muneca avanza rapido hacia el rival y vuelve: posible golpe de mano.
- Trayectoria recta: posible jab o cross.
- Trayectoria lateral: posible hook.
- Trayectoria ascendente: posible uppercut.
- Tobillo acelera hacia el rival: posible patada.
- Altura de la pierna respecto al rival:
  - baja: low kick
  - media: body kick
  - alta: head kick

Este baseline no tiene que ser perfecto. Sirve para validar el flujo completo:

```text
subir video -> procesar -> anotar -> contar -> devolver resultado
```

Lo que debo aprender:

- Features temporales.
- Ventanas de frames.
- Falsos positivos.
- Falsos negativos.
- Post-procesamiento para no contar el mismo golpe varias veces.

### 8. Clasificacion ML de golpes

Segunda version: modelo entrenado para clasificar golpes.

Entrada posible:

- Secuencias de keypoints.
- Velocidades.
- Angulos.
- Distancias relativas entre peleadores.
- Ventanas de 16, 32 o 64 frames.

Modelos candidatos:

- 1D CNN
- Temporal Convolutional Network
- LSTM
- GRU
- Transformer pequeno para secuencias

Clases iniciales:

- no_strike
- jab
- cross
- hook
- uppercut
- low_kick
- body_kick
- head_kick

Lo que debo aprender:

- Preparacion de datasets secuenciales.
- Clasificacion multiclase.
- Desbalance de clases.
- Evaluacion por evento, no solo por frame.
- TensorFlow/Keras.

Nota tecnica:

Para el primer modelo serio conviene empezar con 1D CNN o Temporal Convolutional Network sobre keypoints/features. No conviene empezar con clasificacion directa sobre video RGB completo porque requiere mas datos, mas GPU y es menos explicable.

### 9. Render del video anotado

Responsable de generar el video final.

Debe dibujar:

- bounding boxes
- IDs
- skeleton de pose
- etiqueta del golpe cuando ocurra
- contador por peleador
- confianza del evento si aplica

Ejemplo visual esperado:

```text
FIGHTER_RED [bbox azul]
JAB 0.87

FIGHTER_BLUE [bbox rojo]

REFEREE [bbox amarillo]

Contadores:
RED: Jab 4 | Cross 2 | Hook 1 | Low kick 3
BLUE: Jab 3 | Cross 1 | Hook 2 | Head kick 1
```

Lo que debo aprender:

- Dibujo con OpenCV.
- Codificacion de video.
- Sincronizacion de anotaciones con frames.
- Diseno visual legible sobre video.

## MLOps

El proyecto debe tener control de experimentos, modelos y calidad.

### MLflow

Usar para:

- registrar experimentos
- guardar metricas
- guardar parametros
- guardar modelos
- comparar runs
- manejar model registry

Ejemplos de datos a registrar:

- version del dataset
- modelo usado
- learning rate
- batch size
- accuracy
- precision
- recall
- F1 por clase
- matriz de confusion
- ejemplos fallidos

### Airflow

Usar para orquestar pipelines:

```text
ingest_dataset
 -> validate_annotations
 -> preprocess_keypoints
 -> train_model
 -> evaluate_model
 -> register_model
```

Airflow no debe usarse para procesar cada request de usuario en tiempo real. Para eso conviene usar workers o jobs asincronos.

### Docker

Usar para:

- reproducibilidad
- levantar API
- levantar MLflow
- levantar Airflow
- levantar base de datos
- facilitar despliegue

### Azure

Usar cuando el flujo local este claro.

Servicios posibles:

- Azure Blob Storage para videos, datasets y artefactos.
- Azure Machine Learning para experimentos y entrenamiento.
- Azure Container Registry para imagenes Docker.
- Azure Container Apps o App Service para desplegar la API.
- Application Insights para logs y monitoreo.

Evitar al inicio:

- AKS
- arquitecturas demasiado complejas
- entrenamientos GPU costosos sin dataset listo

## Roadmap personal

### Etapa 1 - Base del proyecto

Meta:

- Tener estructura profesional del repo.
- Tener FastAPI funcionando.
- Poder subir un video.
- Guardar metadata del video.

Resultado esperado:

```text
POST /videos
GET /jobs/{job_id}
```

### Etapa 2 - Video processing local

Meta:

- Extraer frames.
- Leer video con OpenCV.
- Reconstruir video anotado simple.

Resultado esperado:

```text
input.mp4 -> output_annotated.mp4
```

Aunque solo tenga texto o frame number dibujado al inicio.

### Etapa 3 - Deteccion y tracking

Meta:

- Detectar personas.
- Mantener IDs.
- Separar dos peleadores y referee.

Resultado esperado:

```text
video con cajas:
fighter_red
fighter_blue
referee
```

### Etapa 4 - Pose estimation

Meta:

- Extraer keypoints de los peleadores.
- Dibujar skeleton.
- Guardar keypoints por frame.

Resultado esperado:

```text
frames + tracks + pose_data.json
```

### Etapa 5 - Contador basico

Meta:

- Crear reglas iniciales para detectar posibles golpes.
- Evitar contar varias veces el mismo golpe.
- Mostrar conteos en el video.

Resultado esperado:

```text
video anotado con eventos y contadores basicos
```

### Etapa 6 - Dataset etiquetado

Meta:

- Etiquetar clips reales.
- Crear guia de etiquetado.
- Separar train, validation y test por pelea/video.
- Usar peleas completas como fuente, pero segmentarlas por rounds.
- Evitar mezclar rounds de la misma pelea entre train y test.

Resultado esperado:

```text
dataset versionado con eventos:
jab, cross, hook, uppercut, low_kick, body_kick, head_kick, no_strike
```

Estructura sugerida:

```text
data/raw/
  fight_001/full_video.mp4

data/rounds/
  fight_001_round_01.mp4
  fight_001_round_02.mp4
  fight_001_round_03.mp4
  fight_001_round_04.mp4
  fight_001_round_05.mp4

data/annotations/
  fight_001_round_01_events.json
  fight_001_round_02_events.json
```

Regla de evaluacion:

```text
Correcto:
  train -> fight_001, fight_002, fight_003
  validation -> fight_004
  test -> fight_005

Evitar:
  train -> round 1 de fight_001
  test -> round 3 de fight_001
```

La separacion debe hacerse por pelea completa para evitar metricas infladas por ver los mismos peleadores, camara, iluminacion y estilo en train y test.

### Etapa 7 - Modelo ML

Meta:

- Entrenar primer clasificador temporal.
- Compararlo contra el baseline de reglas.
- Registrar resultados en MLflow.

Resultado esperado:

```text
modelo TensorFlow versionado en MLflow
```

### Etapa 8 - Evaluacion seria

Meta:

- Medir precision por tipo de golpe.
- Medir error de conteo por video.
- Medir latencia.
- Documentar limitaciones.

Resultado esperado:

```text
evaluation_report.md
confusion_matrix.png
examples_false_positives/
examples_false_negatives/
```

### Etapa 9 - Despliegue

Meta:

- Dockerizar API.
- Servir modelo registrado.
- Subir artefactos a Azure.
- Desplegar API en Azure.

Resultado esperado:

```text
API desplegada + demo funcional
```

## Prioridades tecnicas

Orden recomendado:

1. Hacer que el video entre y salga anotado.
2. Resolver tracking de personas.
3. Resolver identidad de peleadores y referee.
4. Extraer pose.
5. Crear contador basico.
6. Crear dataset etiquetado.
7. Entrenar modelo.
8. Integrar MLflow y Airflow.
9. Desplegar en Azure.

## Decisiones de alcance tomadas

- El input del producto sera un round individual de maximo 5 minutos.
- Las peleas completas de 3 o 5 rounds se pueden usar como fuente de dataset.
- Las peleas completas deben segmentarse en rounds antes de entrenar/evaluar.
- El sistema no sera un solo modelo, sino un pipeline de modelos y modulos.
- El enfoque principal de ML sera supervised deep learning aplicado a computer vision.
- El clasificador principal sera de acciones temporales basado inicialmente en pose/keypoints.
- Primero se contaran golpes lanzados; golpes conectados, bloqueados o fallados quedan para una version posterior.

## Riesgos del proyecto

- Dataset dificil de conseguir.
- Videos de UFC tienen derechos de autor.
- Cambios de camara complican el tracking.
- Oclusiones entre peleadores.
- Referee puede confundirse con peleadores.
- Golpes rapidos pueden durar pocos frames.
- Algunos golpes son ambiguos incluso para humanos.
- La diferencia entre jab y cross depende de postura e identidad de mano adelantada.
- Contar golpes conectados es mas dificil que contar golpes lanzados.

## Decision importante

Primero contar golpes lanzados. Despues, si el sistema madura, agregar:

- golpe conectado
- golpe bloqueado
- golpe fallado
- combinaciones
- distancia de pelea
- presion ofensiva
- ritmo por round
- precision estimada

## Definicion de MVP

El MVP queda completo cuando pueda:

- Subir un video de un round de maximo 5 minutos.
- Procesarlo asincronicamente.
- Detectar y trackear dos peleadores y referee.
- Dibujar anotaciones en el video.
- Detectar al menos eventos basicos de striking.
- Mostrar contador por peleador.
- Retornar video anotado.
- Retornar JSON con eventos.

## Nota personal

No debo intentar resolver todo desde el primer modelo. El valor del proyecto esta en construir un sistema completo, medible y profesional:

- pipeline claro
- dataset versionado
- evaluacion honesta
- MLOps real
- API funcional
- demo visual
- documentacion de decisiones

La precision mejorara por iteraciones. Primero necesito que todo el flujo funcione de extremo a extremo.
