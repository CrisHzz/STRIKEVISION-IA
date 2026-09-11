# Ejecución manual — extracción y primer entrenamiento

La partición aprobada se lee de `configs/data/strike_split_v1.json`. Estas
instrucciones corresponden a la fase 3: extractor X3D-S congelado y cabeza
binaria. El agente no ejecuta la extracción ni el entrenamiento completos.

Puedes usar los comandos de esta guía o el notebook
`notebooks/StrikeDetection/strike-detection-training.ipynb`. Escoge un recorrido;
no ejecutes dos extracciones/entrenamientos a la vez. El bloqueo de archivos
rechaza un segundo escritor y se libera al terminar el proceso, incluso si falla.

## 1. Activar el entorno y verificar

Abre una terminal Conda/PowerShell en la raíz del proyecto:

```powershell
conda activate strikevision_ML
python -s scripts/ml/run_strike_detection.py check
```

`check` valida etiquetas, exclusiones, partición, pesos y preprocesamiento.
Muestra 25.592 ventanas de train, 7.396 de validación y 7.492 de test.
No decodifica videos ni entrena. El test solo se inventaría y permanece cerrado.

## 2. Extraer características — ejecución larga iniciada por ti

```powershell
python -s scripts/ml/run_strike_detection.py extract --split both
```

La proyección previa es aproximadamente 2 h 15 min para train y validación,
sin garantía: deriva de 108 ventanas y no incluye todo el coste de hashes y disco
añadido por el cache reanudable. Este paso usa CPU y no cambia pesos.

El comando lee secuencialmente cada video, conserva el encuadre RGB y guarda
embeddings float32 de 2.048 dimensiones con IDs, etiquetas humanas, índices y
PTS. El progreso informa conjunto, video y ventanas guardadas. No genera clips
MP4 ni imágenes por ventana. No filtra por calidad de pose.

Cada 128 ventanas se confirma un bloque mediante escritura temporal y reemplazo
atómico. Se guarda bajo `data/processed/features/strike_x3d_s/<hash>/`.
El hash identifica etiquetas, partición, pesos, código, versiones y transformación;
cada video tiene además SHA-256 de fuente y cada bloque su hash de integridad.

**Pausar:** Ctrl+C. **Continuar:** ejecutar exactamente el mismo comando.
Se verifican y reutilizan los bloques confirmados. Como máximo se repite el bloque
sin confirmar. Para mantener índices exactos se vuelve a decodificar el prefijo
del video interrumpido, pero no se vuelve a ejecutar X3D sobre sus ventanas
confirmadas. No hace falta borrar un archivo `writer.lock` que permanezca en disco.

Si cambia el código del extractor/preprocesamiento o sus versiones, se selecciona
otro cache; se preserva el anterior. Si cambia un video o se corrompe un bloque,
el proceso se detiene y explica la causa; no mezcla características incompatibles.
Consulta el error antes de borrar o reemplazar datos.

### Timestamps variables o frames temporalmente ausentes

Las etiquetas siguen referidas al índice de frame decodificado. El cargador guarda
la correspondencia explícita entre cada uno de los 13 índices muestreados y su PTS
real. Un salto positivo de PTS queda registrado y no desplaza, interpola ni cambia
la ventana humana. Se rechazan PTS repetidos o hacia atrás porque no admiten una
correspondencia inequívoca con esta política.

El round O'Malley–Zahabi R2 contiene un salto al último frame: desde el frame
15.850 al 15.851 transcurren aproximadamente 50 ms, en lugar de 16,68 ms. El frame
15.851 se conserva como último frame anotado y su PTS real queda mapeado. La
corrección migró a un cache nuevo los bloques anteriores después de verificar
hashes de video, shards, IDs, etiquetas, índices y PTS. El cache anterior se
preserva como procedencia y no se usa para escrituras nuevas.

Puedes separar la extracción en `--split train` y `--split validation`.
No existe opción `--split test` en esta etapa. Mantén suficientes GB libres para
entorno, cache, temporales y resultados; los embeddings de train+validación ocupan
aproximadamente 258 MiB, más IDs, índices, timestamps y metadatos.

## 3. Entrenar la cabeza — después de completar ambos caches

```powershell
python -s scripts/ml/run_strike_detection.py train --run head-v1
```

Antes de entrenar, el comando exige ambos caches completos y verifica la
alineación exacta de características y etiquetas. No vuelve a procesar RGB.

La receta en `configs/models/strike_training_v1.yaml` compara dos opciones:

- CrossEntropyLoss estándar.
- CrossEntropyLoss con pesos de clase calculados exclusivamente desde train.

Ambas parten de la misma semilla. Solo hay 4.098 parámetros entrenables.
La configuración inicial usa AdamW, learning rate 0,001, batch 256, un máximo
de 50 épocas y paciencia de 8 por opción. No se combinan pesos con sobremuestreo.
Se imprime el tiempo real por época; todavía no se ha medido sobre todo el cache.

En validación se seleccionan conjuntamente checkpoint y umbral por F1 de strike.
La cuadrícula de umbrales va de 0,05 a 0,95 con paso 0,01. En empates de F1 entre
checkpoints se prefiere menor pérdida de validación; en empates de umbrales,
el más cercano a 0,5. El test no interviene en esas decisiones.

**Pausar:** Ctrl+C. **Continuar:** repetir el comando con `--run head-v1`.
Se recuperan cabeza, optimizador, historial, mejor checkpoint y parada temprana
desde la última época terminada. Una época interrumpida se repite. Cambiar la
receta/datos/código requiere un nombre nuevo, por ejemplo `--run head-v2`.

## 4. Verificar recarga en otro proceso

Después de que `train` termine correctamente:

```powershell
python -s scripts/ml/verify_strike_candidate.py --run-dir outputs/strike-detection/head-v1
```

Debe mostrar `head_reload_ok: true`. Comprueba probabilidades de referencia con
tolerancia de 1e-6. Esta verificación corresponde a la cabeza sobre embeddings,
no a un artefacto RGB completo de producción.

## 5. Revisar los resultados antes del test

Resultados en `outputs/strike-detection/head-v1/`:

| Archivo | Contenido |
|---|---|
| `run.json` | Configuración, partición y procedencia |
| `standard/history.json`, `class_weighted/history.json` | Pérdidas, F1, umbral y tiempo por época |
| `standard/last.pt`, `class_weighted/last.pt` | Estado completo para reanudar |
| `candidate.pt` | Cabeza seleccionada, umbral y contrato del extractor |
| `validation.json` | Métricas y comparación con clasificadores constantes |
| `validation_predictions.jsonl` | Predicciones independientes y etiquetas humanas de referencia |
| `reload_check.json` | Resultado de recarga en proceso limpio |

`validation.json` incluye precision/recall/F1 de strike, macro-F1, balanced
accuracy, PR-AUC calculada como average precision y matriz de confusión. También
desglosa por pelea, categoría, FPS, duración y disponibilidad de pose como
diagnóstico; no convierte las ventanas solapadas en golpes independientes.

Al terminar, comparte los resultados de validación y los historiales para
decidir si el baseline merece pasar al test o necesita ajustes. No se ha fijado
una meta de precisión que permita promoverlo automáticamente. El resultado aún
no atribuye acciones a peleadores, cuenta impactos ni está calibrado.

Pendientes tras esta fase: revisión de errores, test final reservado,
empaquetado RGB completo y Gradio. Antes de ejecutar los comandos manuales, las
pruebas de implementación usan datos sintéticos pequeños; el archivo
`validation.json` confirma cuándo existe un candidato real.

## 6. Test final reservado — ejecutar una sola vez tras revisar validación

El candidato `head-v1` quedó seleccionado por validación: pérdida ponderada,
época 8 y umbral 0,33. El test permanece sin evaluar hasta que el usuario decide
abrirlo. Al ejecutarlo, no se permite cambiar la receta, el checkpoint o umbral.

```powershell
python -s scripts/ml/run_strike_detection.py test --run head-v1
```

El comando extrae y guarda primero las características de las 7.492 ventanas de
test, con reanudación por bloques igual a train/validación. La proyección previa
es aproximadamente 31 minutos de procesamiento de video; es una estimación. Una
interrupción durante esa parte se reanuda repitiendo el mismo comando. Después de
escribir `test.json`, el test queda cerrado y el comando se niega a re-ejecutarlo.

`test.json` y `test_predictions.jsonl` contienen las métricas finales y las
predicciones por ventana. Revisar el informe antes de empaquetar el modelo RGB y
construir Gradio.

El test de `head-v1` ya fue ejecutado. El candidato queda como baseline
reproducible y no se vuelve a evaluar para ajustar decisiones.

El baseline `head-v1` se conserva como el modelo de referencia reproducible.
Para probarlo sobre videos nuevos sin procesar un round completo, usa
`scripts/ml/run_strike_inference_app.py` y consulta
`docs/STRIKE-GRADIO-TESTER.md`.
