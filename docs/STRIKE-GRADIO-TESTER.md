# Probador local de `strike/no_strike`

Para visualizar también cajas, IDs unidos por merge tracking y poses de los
peleadores, usa el probador combinado (mismos controles y atajos a 30 FPS):

```powershell
conda activate strikevision_ML
python -s -m pip install -r requirements-pose-strike.txt
python -s scripts/ml/run_pose_strike_inference_app.py
```

Abre `http://127.0.0.1:7863`. Combina YOLO11n-seg, ByteTrack con unión de
fragmentos, MediaPipe Pose y el clasificador `head-v1`. Strike/no strike describe
la escena completa; no asigna golpes a los IDs. Ambos modelos reciben el video
original. Los resultados combinados se guardan en
`outputs/strike-detection/gradio-pose-strike/`, incluyendo MP4, JSON, poses y tracks.

La interfaz permite probar el baseline X3D-S `head-v1` con un video local sin
procesar el round completo. Elige el número de frames iniciales que se desea
analizar; el límite es 9.000 frames (cinco minutos a 30 FPS) y el valor inicial
es 240.

```powershell
conda activate strikevision_ML
python -s scripts/ml/run_strike_inference_app.py
```

Abre `http://127.0.0.1:7862`, carga el video e indica los frames. El modelo
analiza ventanas de 24 frames con salto de 12 frames, igual que el dataset. En
cada ventana toma 13 muestras uniformes, preserva el aspect ratio y aplica el
preprocesamiento usado en entrenamiento.

Los videos de este proyecto se procesan a 30 FPS: 9.000 frames cubren
exactamente un round de cinco minutos. La aplicación conserva esa cadencia en
el MP4 de vista previa; no duplica frames ni convierte el video a 60 FPS.

La tabla muestra `p(strike)` y la clase con el umbral fijo de validación 0,33.
La vista previa sólo contiene el prefijo solicitado y dibuja la predicción de
la ventana solapada con mayor `p(strike)` para facilitar la revisión. El JSON y
el MP4 quedan en `outputs/strike-detection/gradio-inference/`.

Las ventanas se solapan, por lo que el número de filas marcadas como `strike`
no equivale a un conteo de golpes. Esta utilidad es para evaluar visualmente el
modelo actual, cuyo F1 de strike en el test reservado es 0,606.
