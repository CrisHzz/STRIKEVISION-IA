# Probador local de `strike/no_strike`

La interfaz permite probar el baseline X3D-S `head-v1` con un video local sin
procesar el round completo. Elige el número de frames iniciales que se desea
analizar; el límite es 3.600 frames y el valor inicial es 240.

```powershell
conda activate strikevision_ML
python -s scripts/ml/run_strike_inference_app.py
```

Abre `http://127.0.0.1:7862`, carga el video e indica los frames. El modelo
analiza ventanas de 24 frames con salto de 12 frames, igual que el dataset. En
cada ventana toma 13 muestras uniformes, preserva el aspect ratio y aplica el
preprocesamiento usado en entrenamiento.

La tabla muestra `p(strike)` y la clase con el umbral fijo de validación 0,33.
La vista previa sólo contiene el prefijo solicitado y dibuja la predicción de
la ventana solapada con mayor `p(strike)` para facilitar la revisión. El JSON y
el MP4 quedan en `outputs/strike-detection/gradio-inference/`.

Las ventanas se solapan, por lo que el número de filas marcadas como `strike`
no equivale a un conteo de golpes. Esta utilidad es para evaluar visualmente el
modelo actual, cuyo F1 de strike en el test reservado es 0,606.
