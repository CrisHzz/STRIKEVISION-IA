# Entorno independiente StrikeVision ML

El entorno Conda `strikevision_ML` corresponde a las fases 1–2 de X3D-S RGB.
El entorno anterior `strikevision` conserva pose y anotación.

```powershell
conda activate strikevision_ML
python -s -m pip check
python -s -m pytest tests/unit/test_strike_rgb_dataset.py tests/unit/test_strike_detection.py -q
```

En Jupyter/VS Code, seleccionar **Python (strikevision_ML)** y abrir
`notebooks/StrikeDetection/strike-detection-training.ipynb`.
La variable `PYTHONNOUSERSITE=1` está guardada en el entorno y en el kernel.
Al invocar el ejecutable por ruta absoluta sin activar Conda, conservar `-s`.

## Recreación

Desde la raíz del proyecto, para un equipo donde todavía no exista el entorno:

```powershell
conda env create -f environment-strike-ml.yml
conda activate strikevision_ML
python -s -m pip install -r requirements-strike-ml.lock.txt
python -s -m pip install --no-deps -e .
python -s -m ipykernel install --user --name strikevision_ML --display-name "Python (strikevision_ML)" --env PYTHONNOUSERSITE 1
```

El lock registra las versiones efectivamente instaladas para Windows/Python 3.11.
Las dependencias directas y el índice de ruedas CPU están en
`requirements-strike-ml.txt`; no se necesita instalar YOLO, MediaPipe o MLflow
para esta etapa. PyAV usa su FFmpeg incluido. Se importa el constructor de
modelos de PyTorchVideo; no se usan sus transformaciones históricas de torchvision.

Pesos originales:
`https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth`.
Ubicación local: `models/strike_x3d_s/X3D_S.pyth`.
SHA-256 esperado:
`26b95f1605d49650b54049db40ba3a56e023b86b58c3b3e0e10e0992a9c8682f`.
Se rechaza cualquier archivo con un hash distinto y no se descarga código online.

## Probar el modelo actual

```powershell
python -s scripts/ml/run_strike_inference_app.py
```

La interfaz local permite cargar un video y analizar únicamente la cantidad de
frames indicada. Consulta `docs/STRIKE-GRADIO-TESTER.md` para el uso y los
límites de las predicciones por ventana.

## Estado y decisiones pendientes

La partición 9/2/2 fue aprobada por el usuario y se registra en
`configs/data/strike_split_v1.json`. El usuario ejecutará manualmente la
extracción y el entrenamiento; el agente no debe iniciarlos. La configuración
y los resultados de fases 1–2 se conservan como snapshot del benchmark original.

Implementados: contratos RGB, validación de fuentes, partición aprobada,
modelo congelado, pruebas técnicas y notebook de entrenamiento.
Ya están implementados el cache reanudable, entrenamiento/validación de cabeza
y comandos manuales: consultar `docs/STRIKE-ML-MANUAL.md`.
El baseline `head-v1` ya fue entrenado, evaluado y conectado a Gradio.

Las pruebas de FPS sintéticos incluyen 24, 30, 60, 30000/1001 y 60000/1001.
Las cabeceras se verifican en los 54 videos; los PTS se verifican solo a medida
que se decodifican. El benchmark recorre prefijos de nueve videos, por lo que no
demuestra todavía que todos los rounds completos tengan timestamps uniformes.
