# StrikeVision

StrikeVision es un proyecto preliminar de computer vision y machine learning enfocado en analizar rounds de peleas de MMA/UFC desde el striking.

La idea principal es permitir cargar un video de un round y devolver el mismo video anotado con tracking de los dos peleadores, el referee y eventos de golpe como jabs, crosses, hooks, uppercuts, low kicks, body kicks y head kicks.

El proyecto busca construir un sistema modular que combine deteccion de personas, tracking, pose estimation, clasificacion temporal de acciones y MLOps para versionar datos, experimentos y modelos.

Por ahora el alcance esta cerrado a striking. Grappling, clinch prolongado, control en piso y analisis completo de peleas quedan fuera de la primera version.

## Laboratorio web local

El proyecto incluye dos páginas locales para subir un round y probar progresivamente el pipeline:

- **Gradio**: interfaz principal para experimentación con video y modelos ML.
- **Streamlit**: alternativa para dashboards, métricas y exploración de resultados.

Actualmente implementan inspección de metadata y el Pipeline visual V0, que lee, anota y reconstruye
el video. Detection, tracking, pose y clasificación se conectarán progresivamente al mismo servicio
compartido.

Ejecutar Gradio:

```powershell
conda activate strikevision
python -m ufc_tracker.ui.gradio_app
```

Ejecutar Streamlit:

```powershell
conda activate strikevision
streamlit run src/ufc_tracker/ui/streamlit_app.py
```

Consulta la guía completa en [`docs/INTERFACES-WEB.md`](docs/INTERFACES-WEB.md).

## Pose estimation

La etapa de pose usa **MediaPipe Pose Landmarker** sobre los crops generados por el
`PersonDetector@production` existente. El detector YOLO-seg + ByteTrack identifica los
peleadores y excluye al referee; MediaPipe agrega hombros, codos, muñecas, caderas,
rodillas y tobillos en coordenadas del frame original.

Flujo:

```text
Round de video
  -> PersonDetector UFC + ByteTrack
  -> fragmentos fighter_track_<id>
  -> MediaPipe Pose Landmarker
  -> tracking.jsonl + pose.jsonl + pose_preview.mp4 + métricas
```

El notebook de desarrollo es
[`notebooks/pose-estimation/pose-estimation.ipynb`](notebooks/pose-estimation/pose-estimation.ipynb).
Permite seleccionar cualquier round desde `splits_manifest.csv`, ejecutar primero 300 frames,
inspeccionar métricas y después procesar el round completo.

Ejecutar el pipeline genérico desde PowerShell:

```powershell
conda activate strikevision
python -m pip install -e .

# Prueba corta
python scripts/ml/run_pose_pipeline.py `
  --video "data/splits/<categoria>/<round>.mp4" `
  --max-frames 300

# Round completo
python scripts/ml/run_pose_pipeline.py `
  --video "data/splits/<categoria>/<round>.mp4"
```

Por defecto los artefactos se guardan en:

```text
data/processed/poses/<video_stem>/
  tracking.jsonl
  pose.jsonl
  pose_preview.mp4
  pose_metrics.json
  run_metadata.json
```

Interpretación de métricas:

- `pose_coverage`: proporción de observaciones visibles con al menos cuatro keypoints esenciales.
- `required_keypoint_availability`: disponibilidad individual de muñecas, codos, rodillas y tobillos.
- `required_keypoint_temporal_continuity`: continuidad de cada articulación entre frames consecutivos.
- `mean_inference_ms_per_visible_frame`: latencia media de pose por peleador visible.

Después de revisar el preview, el resultado puede versionarse directamente con DVC:

```powershell
python scripts/ml/run_pose_pipeline.py `
  --video "data/splits/<categoria>/<round>.mp4" `
  --dvc-add
```

El script crea un puntero en `data/metadata/pose_<video_stem>.dvc`. Git guarda el puntero,
la configuración y el código; DVC guarda los JSONL, métricas y video derivados.

## Entorno de desarrollo

El entorno principal del proyecto esta definido con Conda + pip y esta limitado al
pipeline local que se usa actualmente:

- `environment.yml`: crea el entorno `strikevision` con Python 3.11, pip, ffmpeg, git y graphviz.
- `requirements.txt`: instala computer vision, ML clasico, MLflow, DVC, interfaces locales y testing.
- `requirements-azure.txt`: agrega Azure Blob Storage y Azure Machine Learning solo al preparar el despliegue.

### Crear el entorno Conda

Primero instala Miniconda, Anaconda o Miniforge. Luego, desde la raiz del proyecto:

```powershell
conda env create -f environment.yml
conda activate strikevision
python -m pip check
```

Si ya existe el entorno y quieres actualizarlo:

```powershell
conda env update -f environment.yml --prune
conda activate strikevision
python -m pip check
```

### Instalar el paquete del proyecto

El codigo fuente vive en `src/ufc_tracker` y se instala en modo editable para que los imports (`from ufc_tracker... import ...`) funcionen desde scripts, notebooks y tests:

```powershell
python -m pip install -e .
```

Esto se hace una sola vez por entorno. Si creas el entorno Conda de nuevo, vuelve a ejecutarlo.

### Verificar librerias principales

```powershell
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import mlflow; print('MLflow:', mlflow.__version__)"
python -c "import dvc.version; print('DVC:', dvc.version.__version__)"
```

### Relacion con DVC, MLflow y Azure

Uso esperado:

```text
DVC -> versionar videos, rounds, features y datasets.
MLflow -> registrar experimentos, metricas, parametros y modelos.
Azure Blob Storage -> remote futuro de DVC.
Azure ML / MLflow -> tracking y model registry futuro.
```

El entorno local usa DVC/MLflow sin instalar SDK de nube. Cuando llegue el despliegue,
las integraciones se agregan sobre el mismo entorno:

```powershell
python -m pip install -r requirements-azure.txt
```

Asi Azure sigue siendo el destino previsto, pero no aumenta el peso ni la complejidad
del desarrollo local antes de necesitarlo.

## Versionado local de datos con DVC

Los videos de `data/splits` estan versionados con DVC por categoria de pelea:

```text
data/splits/aggressive_men.dvc
data/splits/aggressive_women.dvc
data/splits/low_kicks_men.dvc
data/splits/low_kicks_women.dvc
data/splits/normal_men.dvc
data/splits/passive_men.dvc
data/splits/passive_women.dvc
data/splits/striking_light_grappling_men.dvc
```

El remote local temporal esta en:

```text
C:\Users\Cristian\Documents\dvc-remotes\strikevision
```

La metadata del dataset se genera con:

```powershell
conda activate strikevision
python scripts/data/build_splits_manifest.py
```

Archivos generados:

```text
data/metadata/splits_manifest.csv
data/metadata/splits_manifest.json
data/metadata/splits_summary_by_category.csv
```

Estos archivos registran categoria, etiqueta, pelea, round, duracion, FPS, resolucion, codec, tamano, ruta y bandera de calidad de duracion.

Para actualizar DVC despues de agregar o cambiar videos:

```powershell
conda activate strikevision
python scripts/data/build_splits_manifest.py

dvc add data/splits/aggressive_men data/splits/aggressive_women data/splits/low_kicks_men data/splits/low_kicks_women data/splits/normal_men data/splits/passive_men data/splits/passive_women data/splits/striking_light_grappling_men
dvc push
```

Luego guardar en Git solo los punteros y metadata:

```powershell
git add .dvc .dvcignore .gitignore data/splits/*.dvc data/splits/.gitignore data/metadata scripts/data/build_splits_manifest.py
git commit -m "Version dataset splits with DVC"
```

Los `.mp4` no deben entrar a Git. Git guarda punteros `.dvc`; DVC guarda los archivos pesados en el remote local y mas adelante en Azure Blob Storage.
