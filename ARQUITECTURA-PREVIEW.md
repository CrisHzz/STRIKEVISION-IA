# Arquitectura Preview - Machine Learning

Este documento guarda la vision tecnica inicial de la arquitectura de machine learning del proyecto. Es una referencia personal para tener claro que no se trata de un unico modelo, sino de un sistema modular de computer vision.

## Definicion tecnica del proyecto

El proyecto se puede describir asi:

```text
Sistema de computer vision y deep learning supervisado para analisis de striking en video, usando deteccion de personas, multi-object tracking, pose estimation y clasificacion temporal de acciones.
```

Categorias principales:

```text
Area general: Computer Vision
Tecnica principal: Deep Learning
Paradigma principal: Supervised Learning
Tipo de problema: Video Understanding / Action Recognition / Temporal Action Classification
```

No es principalmente unsupervised learning. Para diferenciar golpes como jab, cross, hook, uppercut, low kick, body kick y head kick, el sistema necesitara ejemplos etiquetados.

## Decision arquitectonica principal

No conviene entrenar un modelo gigante que reciba el video completo y devuelva directamente todos los golpes.

Esa opcion seria:

```text
video completo -> modelo unico -> eventos de golpes
```

Problemas de ese enfoque:

- Requiere muchisimos datos etiquetados.
- Es costoso en GPU.
- Es dificil de explicar.
- Es dificil saber donde falla.
- Es mas complejo de evaluar por partes.
- Mezcla deteccion, tracking, pose y clasificacion en un solo problema.

La arquitectura recomendada es modular:

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

## Idea central

La funcionalidad final es una sola:

```text
Subir video de un round -> devolver video anotado con tracking y golpes
```

Pero internamente se resuelve con varios modelos, algoritmos y reglas:

```text
1. Detectar personas.
2. Mantener identidad de cada persona.
3. Separar peleador rojo, peleador azul y referee.
4. Extraer pose de los peleadores.
5. Generar features temporales.
6. Encontrar ventanas donde posiblemente hubo golpe.
7. Clasificar el golpe.
8. Limpiar eventos duplicados.
9. Renderizar el video anotado.
```

## Componentes del sistema ML

| Componente | Usa ML | Funcion |
|---|---:|---|
| Person Detection | Si | Detectar personas visibles en cada frame |
| Multi Object Tracking | No necesariamente | Mantener IDs entre frames |
| Role Assignment | Reglas o ML | Decidir quien es fighter_red, fighter_blue y referee |
| Pose Estimation | Si | Extraer keypoints corporales |
| Feature Engineering | No necesariamente | Calcular velocidades, angulos y distancias |
| Strike Candidate Detection | Reglas o ML | Encontrar posibles ventanas de golpe |
| Strike Classification | Si | Clasificar el tipo de golpe |
| Event Post-processing | No | Unir eventos, eliminar duplicados y aplicar thresholds |
| Video Renderer | No | Dibujar cajas, skeleton, etiquetas y contadores |

## 1. Person Detection

Objetivo:

Detectar todas las personas visibles en cada frame.

Entrada:

```text
frame de video
```

Salida:

```json
{
  "frame": 100,
  "detections": [
    {
      "bbox": [120, 80, 380, 720],
      "confidence": 0.94,
      "class": "person"
    }
  ]
}
```

Opciones tecnicas:

- YOLO como opcion practica.
- TensorFlow Object Detection API si quiero mantener el stack mas cercano a TensorFlow.
- EfficientDet si quiero trabajar dentro del ecosistema TensorFlow.

Decision inicial:

Usar un detector preentrenado. No entrenar desde cero en la primera version.

## 2. Multi Object Tracking

Objetivo:

Mantener la identidad de cada persona a lo largo del video.

Ejemplo:

```text
track_1 -> misma persona durante varios frames
track_2 -> misma persona durante varios frames
track_3 -> misma persona durante varios frames
```

Salida esperada:

```json
{
  "frame": 100,
  "tracks": [
    {
      "track_id": 1,
      "bbox": [120, 80, 380, 720],
      "confidence": 0.91
    },
    {
      "track_id": 2,
      "bbox": [500, 90, 760, 720],
      "confidence": 0.89
    }
  ]
}
```

Opciones tecnicas:

- SORT como baseline simple.
- DeepSORT si necesito embeddings de apariencia.
- ByteTrack como opcion fuerte para mantener tracks con detecciones de alta y baja confianza.

Metricas a investigar:

- IDF1.
- MOTA.
- identity switches.
- track fragmentation.

## 3. Role Assignment

Objetivo:

Convertir tracks genericos en roles del dominio:

```text
track_1 -> fighter_red
track_2 -> fighter_blue
track_3 -> referee
```

Este modulo es critico porque el clasificador de golpes solo debe correr sobre los peleadores, no sobre el referee.

Reglas iniciales posibles:

- Los dos peleadores suelen estar mas cerca entre si.
- Los peleadores tienen mas interaccion ofensiva/defensiva.
- El referee se mueve distinto y no lanza golpes.
- El referee suele mantener distancia del intercambio.
- Los colores de short, esquina o guantes pueden ayudar si son visibles.

Salida:

```json
{
  "track_id": 1,
  "role": "fighter_red"
}
```

Version inicial:

Reglas temporales y heuristicas.

Version futura:

Clasificador de rol entrenado con features de movimiento, posicion y apariencia.

## 4. Pose Estimation

Objetivo:

Extraer puntos corporales de cada peleador.

Keypoints importantes:

- hombros
- codos
- munecas
- cadera
- rodillas
- tobillos

Entrada:

```text
crop del peleador o frame + bbox
```

Salida:

```json
{
  "fighter_id": "fighter_red",
  "frame": 100,
  "keypoints": {
    "left_shoulder": [230, 180, 0.93],
    "left_elbow": [210, 260, 0.88],
    "left_wrist": [190, 330, 0.84],
    "right_shoulder": [300, 180, 0.94],
    "right_elbow": [330, 260, 0.86],
    "right_wrist": [360, 330, 0.81]
  }
}
```

Opciones tecnicas:

- MediaPipe Pose.
- MoveNet.
- YOLO Pose.
- modelo compatible con TensorFlow.

Decision inicial:

Usar pose estimation preentrenado. El foco propio del proyecto sera el clasificador temporal de golpes.

## 5. Feature Engineering

Objetivo:

Transformar keypoints crudos en senales utiles para detectar acciones.

Features posibles:

- posicion normalizada de keypoints
- velocidad de munecas
- velocidad de tobillos
- aceleracion de extremidades
- angulos de codo
- angulos de rodilla
- orientacion del torso
- distancia entre peleadores
- direccion del movimiento hacia el rival
- altura relativa del pie o mano respecto al oponente
- cambio de distancia durante la accion

Ejemplo de tensor:

```text
[T=32 frames, F=120 features]
```

Donde:

- `T` es la longitud temporal de la ventana.
- `F` es la cantidad de features por frame.

Esta parte es importante porque hace el modelo mas explicable y mas barato que usar video RGB completo.

## 6. Strike Candidate Detection

Objetivo:

Detectar ventanas donde posiblemente hubo un golpe.

No conviene clasificar todo el video frame por frame sin filtro. Primero se generan candidatos.

Ejemplo:

```text
frames 1200-1232 -> posible golpe de mano
frames 1510-1545 -> posible patada
```

Reglas iniciales:

- Muneca acelera hacia el rival.
- Brazo se extiende y vuelve.
- Tobillo acelera hacia el rival.
- Pierna sube a altura baja, media o alta.
- Distancia entre peleadores cambia durante la accion.

Salida:

```json
{
  "candidate_id": "cand_001",
  "fighter_id": "fighter_red",
  "start_frame": 1200,
  "end_frame": 1232,
  "candidate_type": "hand_strike",
  "score": 0.76
}
```

Version inicial:

Heuristica basada en pose y movimiento.

Version futura:

Modelo binario para detectar `strike` vs `no_strike`.

## 7. Strike Classification

Objetivo:

Clasificar que tipo de golpe ocurrio dentro de una ventana candidata.

Clases iniciales:

```text
no_strike
jab
cross
hook
uppercut
low_kick
body_kick
head_kick
```

Entrada:

```text
secuencia temporal de keypoints + features
```

Ejemplo:

```text
[32 frames, 120 features] -> modelo -> jab
```

Arquitectura inicial recomendada:

```text
Input [T, F]
 -> Normalization
 -> Conv1D
 -> Conv1D
 -> Dropout
 -> GlobalAveragePooling1D
 -> Dense
 -> Softmax
```

Modelos candidatos:

- 1D CNN.
- Temporal Convolutional Network.
- LSTM.
- GRU.
- Transformer temporal pequeno.

Decision inicial:

Empezar con 1D CNN o Temporal Convolutional Network sobre keypoints/features.

Razon:

- Menos costoso que usar video RGB.
- Mas facil de entrenar con pocos datos.
- Mas interpretable.
- Mejor para un MVP profesional.
- Permite depurar errores por movimiento corporal.

## 8. Event Post-processing

Objetivo:

Limpiar predicciones para convertirlas en eventos reales.

Problema:

El modelo puede predecir el mismo golpe en muchos frames o ventanas cercanas.

Ejemplo crudo:

```text
frame 1200 -> jab
frame 1201 -> jab
frame 1202 -> jab
frame 1203 -> jab
```

Evento limpio:

```json
{
  "fighter_id": "fighter_red",
  "event": "jab",
  "start_frame": 1200,
  "end_frame": 1220,
  "confidence": 0.87
}
```

Reglas posibles:

- unir eventos cercanos del mismo tipo
- aplicar threshold de confianza
- aplicar cooldown minimo entre golpes
- filtrar eventos demasiado cortos
- filtrar eventos demasiado largos
- usar non-maximum suppression temporal

## 9. Annotated Video Renderer

Objetivo:

Generar el video final procesado.

Debe dibujar:

- bounding boxes
- IDs persistentes
- roles: fighter_red, fighter_blue, referee
- skeleton de pose para peleadores
- etiqueta del golpe cuando ocurra
- confianza del evento
- contador por peleador

Ejemplo visual:

```text
FIGHTER_RED [bbox azul]
JAB 0.87

FIGHTER_BLUE [bbox rojo]

REFEREE [bbox amarillo]

RED: Jab 4 | Cross 2 | Hook 1 | Low kick 3
BLUE: Jab 3 | Cross 1 | Hook 2 | Head kick 1
```

## Contrato de salida del sistema

El sistema debe retornar:

```text
1. Video anotado.
2. JSON de eventos.
3. Resumen de conteos.
```

Ejemplo JSON:

```json
{
  "video_id": "fight_001_round_01",
  "annotated_video_path": "outputs/fight_001_round_01_annotated.mp4",
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

## Datos y entrenamiento

La entrada del producto sera un round de maximo 5 minutos, pero la recoleccion de datos puede venir de peleas completas.

Estrategia:

```text
pelea completa
 -> segmentar por rounds
 -> extraer clips/eventos
 -> etiquetar acciones
 -> entrenar/evaluar con splits por pelea
```

Regla importante:

No mezclar rounds de la misma pelea entre train y test.

Correcto:

```text
train: fight_001, fight_002, fight_003
validation: fight_004
test: fight_005
```

Evitar:

```text
train: fight_001_round_01
test: fight_001_round_03
```

Razon:

Si el modelo ve la misma pelea, peleadores, camara e iluminacion en train y test, las metricas pueden quedar infladas.

## Metricas por modulo

Person Detection:

- mAP
- precision
- recall

Tracking:

- IDF1
- MOTA
- identity switches

Pose Estimation:

- keypoint confidence
- estabilidad temporal
- porcentaje de frames con pose valida

Strike Classification:

- precision por clase
- recall por clase
- F1 por clase
- matriz de confusion

Conteo final:

- error absoluto por tipo de golpe
- error absoluto por peleador
- diferencia entre conteo real y conteo predicho

Evento temporal:

- acierto si el evento predicho cae dentro de una ventana permitida, por ejemplo +/- 10 frames del evento real

## Arquitectura de modelos esperada

En la primera version seria razonable tener:

```text
Modelo 1: detector de personas preentrenado
Modelo 2: pose estimation preentrenado
Modelo 3: clasificador temporal de golpes entrenado por mi
Algoritmo 1: tracker
Modulo 1: role assignment
Modulo 2: post-processing
Modulo 3: renderer
```

Con el tiempo podria evolucionar a:

```text
Modelo 1: detector/segmentador ajustado al dominio
Modelo 2: pose estimation o pose refinement
Modelo 3: role classifier
Modelo 4: strike candidate detector
Modelo 5: strike classifier
Modelo 6: impact/result classifier
```

## Roadmap tecnico de ML

### Version 0 - Pipeline visual

Objetivo:

- Leer video.
- Extraer frames.
- Dibujar anotaciones simples.
- Reconstruir video.

No requiere ML avanzado.

### Version 1 - Detection + Tracking

Objetivo:

- Detectar personas.
- Mantener track IDs.
- Dibujar cajas e identidades.

Resultado:

```text
video con fighter_red, fighter_blue y referee
```

### Version 2 - Pose

Objetivo:

- Extraer keypoints de peleadores.
- Dibujar skeleton.
- Guardar pose por frame.

Resultado:

```text
tracks + pose_data.json
```

### Version 3 - Baseline de golpes

Objetivo:

- Usar reglas sobre pose para detectar posibles golpes.
- Crear primer contador basico.
- Mostrar eventos en video.

Resultado:

```text
video anotado con conteos iniciales
```

### Version 4 - Clasificador temporal

Objetivo:

- Crear dataset etiquetado.
- Entrenar modelo TensorFlow.
- Registrar experimentos en MLflow.
- Comparar contra baseline.

Resultado:

```text
strike_classifier_v1
```

### Version 5 - Evaluacion y MLOps

Objetivo:

- Evaluar por clase.
- Evaluar por evento.
- Evaluar conteo final.
- Registrar modelo en MLflow Model Registry.
- Automatizar pipeline con Airflow.

Resultado:

```text
modelo versionado, evaluado y listo para integracion con API
```

## Riesgos tecnicos

- Distinguir jab vs cross depende de postura y mano adelantada.
- Hooks y crosses pueden parecer similares desde ciertos angulos.
- Uppercuts pueden ocultarse por oclusion.
- Patadas pueden durar pocos frames.
- El referee puede confundirse con peleadores.
- Cambios de camara rompen tracking.
- Oclusiones generan perdida de pose.
- La calidad del video afecta deteccion y pose.
- Contar golpes conectados es mucho mas dificil que contar golpes lanzados.

## Decision final por ahora

La arquitectura base sera:

```text
Detection + Tracking + Role Assignment + Pose Estimation + Temporal Strike Classification
```

El foco principal de entrenamiento propio sera:

```text
Clasificador temporal supervisado de golpes basado en pose/keypoints.
```

Los demas componentes pueden iniciar con modelos preentrenados, algoritmos existentes y reglas. Esto permite construir un sistema completo, medible y profesional antes de intentar entrenar todos los componentes desde cero.
