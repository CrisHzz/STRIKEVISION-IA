# Interfaces web locales de StrikeVision

StrikeVision incluye dos interfaces que consumen el mismo servicio de video:

- **Gradio**: interfaz principal recomendada para probar funciones de ML.
- **Streamlit**: alternativa orientada a dashboards, métricas y exploración.

Las interfaces no contienen lógica de computer vision. Ambas llaman a
`ufc_tracker.services.video_lab`, que valida el video, crea un job local y ejecuta la etapa
seleccionada. De esta manera, detection, tracking, pose y clasificación podrán agregarse una sola
vez y aparecer en las dos páginas.

## Instalación con Miniconda

Desde la raíz del repositorio:

```powershell
conda activate strikevision
conda env update -f environment.yml --prune
python -m pip install -e .
python -m pip check
```

`environment.yml` instala `requirements.txt`; allí están declarados Gradio y Streamlit. No es
necesario instalarlos manualmente por separado.

El proyecto mantiene Gradio en la serie 5.49 por compatibilidad con SceneDetect 0.6: ambas ramas
pueden compartir una versión compatible de `click`. Subir Gradio a la serie 6 requerirá revisar o
actualizar primero esa dependencia de procesamiento de video.

## Ejecutar Gradio

```powershell
conda activate strikevision
python -m ufc_tracker.ui.gradio_app
```

La aplicación abre el navegador automáticamente. Normalmente utiliza
`http://127.0.0.1:7860`.

## Ejecutar Streamlit

```powershell
conda activate strikevision
streamlit run src/ufc_tracker/ui/streamlit_app.py
```

Streamlit normalmente abre `http://localhost:8501`.

## Flujo interno

```text
Navegador
  -> Gradio o Streamlit
  -> run_video_lab(...)
  -> validar formato, metadata y duración
  -> crear outputs/jobs/<job_id>/
  -> ejecutar etapa seleccionada
  -> devolver video, metadata y archivos descargables
```

Los archivos de cada ejecución quedan en `outputs/jobs/<job_id>/`. `outputs/` está ignorado por
Git para evitar subir videos pesados al repositorio.

## Etapas disponibles

### Inspección del video

Guarda una copia local y devuelve duración, FPS, frames, resolución, codec y tamaño. Aplica el
límite actual de cinco minutos por round.

### Pipeline visual (V0)

Lee todos los frames con OpenCV, dibuja el identificador de StrikeVision junto con frame y
timestamp, y reconstruye un MP4. Esta etapa prueba el circuito completo de entrada y salida antes
de conectar modelos.

La salida V0 no conserva audio. El audio no es necesario para el pipeline de computer vision y se
podrá remultiplexar con FFmpeg en una versión posterior si se requiere.

## Evolución prevista

Las etapas se incorporarán al servicio compartido en este orden:

1. Person detection.
2. Multi-object tracking.
3. Role assignment.
4. Pose estimation.
5. Strike candidate detection.
6. Strike classification.
7. Post-processing y render final.

Para procesamiento local una ejecución síncrona es suficiente. Cuando los jobs sean largos o haya
varios usuarios, FastAPI podrá recibir la solicitud y Celery/Redis ejecutar el pipeline en segundo
plano; Gradio puede montarse posteriormente en la misma aplicación FastAPI.
