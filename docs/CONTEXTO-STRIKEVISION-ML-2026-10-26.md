# Contexto de StrikeVision ML

Fecha del estado: 27 de octubre de 2026.

Este documento sirve para continuar el trabajo en otra conversación sin perder
las decisiones, resultados y restricciones importantes. La fuente detallada del
reconocimiento de acciones sigue siendo `docs/PLAN-ACTION-RECOGNITION.md`.

## Instrucciones del usuario que se deben respetar

- El entorno Conda de trabajo es `strikevision_ML` y no debe borrarse.
- El usuario ejecuta manualmente todos los entrenamientos largos. El agente puede
  preparar código, configuraciones, datos, diagnósticos e inferencias cortas, pero
  no debe iniciar entrenamientos sin una orden explícita.
- Consultar antes de cambiar categorías, corregir anotaciones humanas en volumen,
  adoptar otra arquitectura principal o tomar una decisión destructiva.
- No sobrescribir anotaciones originales ni resultados históricos.
- El video del proyecto se considera de 30 FPS. Cinco minutos son 9.000 frames.
- `ambiguous` no es una clase que deba predecir el modelo. Significa que no fue
  posible decidir la acción, hubo problemas de pose/tracking o la escena no fue
  suficientemente clara.
- Durante una sesión de anotación autorizada, continuar hasta completar el bloque
  acordado o hasta que el usuario indique que se debe parar. No interrumpir por
  iniciativa propia para presentar avances parciales, pedir confirmaciones que no
  sean necesarias o cambiar de método. Una imposibilidad técnica real sí se debe
  informar con precisión, sin declarar como terminadas anotaciones no revisadas.

## Objetivo del proyecto

Procesar un round de UFC para:

1. Detectar y seguir a los dos peleadores.
2. Mantener ID 1 rojo e ID 2 morado.
3. Estimar la pose de cada peleador.
4. Detectar momentos globales `strike`/`no_strike`.
5. Atribuir una técnica a cada peleador:
   `punch_head`, `punch_body`, `kick_leg`, `kick_body`, `kick_head` u
   `other_strike`.
6. Mostrar los eventos de manera clara, corta y cercana al momento real, con una
   presentación tipo videojuego.

Se detectan intentos/momentos de striking. Un golpe fallado o bloqueado puede
contar como strike; no se está estimando si conectó limpiamente.

## Arquitectura que existe

La cadena visual actual combina componentes independientes:

- Detección de personas: YOLO11n-seg.
- Tracking: ByteTrack más unión de fragmentos.
- Pose: MediaPipe Pose sobre los dos peleadores elegidos.
- Strike/no-strike global: X3D-S congelado más cabeza `head-v1`.
- Técnica por peleador: TCN sobre secuencias de pose del atacante y rival.

McBytePlusPlus fue estudiado y descartado. La carpeta
`experiments/McBytePlusPlus` se eliminó. No participa en las aplicaciones ni en
inferencia. El entorno `strikevision_ML` se conservó.

## Estado de strike/no-strike

El candidato binario de referencia es:

```text
outputs/strike-detection/head-v1
```

- Umbral fijo de validación: 0,33.
- F1 de `strike` en el test reservado: 0,606.
- Trabaja con ventanas de 24 frames y salto de 12 frames.
- Su salida describe la escena global; por sí sola no identifica cuál peleador
  lanzó el ataque ni cuenta golpes individuales.

Aplicaciones:

```powershell
conda activate strikevision_ML

# Strike/no-strike sin pose
python -s scripts/ml/run_strike_inference_app.py

# YOLO + tracking unido + pose + strike/no-strike
python -s scripts/ml/run_pose_strike_inference_app.py
```

El límite de ambas aplicaciones se adaptó a 9.000 frames y se añadieron atajos
de 30 s, 1 min, 1:30 min, 2 min, 2:30 min y 3 min. La salida conserva 30 FPS.

## Anotación de técnicas

Aplicación:

```powershell
conda activate strikevision_ML
python -s scripts/annotation/run_action_annotator.py
```

Si el puerto 7864 está ocupado, el launcher busca automáticamente el siguiente
puerto libre.

El anotador carga únicamente fragmentos previamente marcados como `strike`; no
obliga a recorrer de nuevo todo el round. Permite registrar varias acciones y
ambos peleadores dentro del mismo fragmento. Los rangos de cada acción se
guardan por separado.

Regla de identidad:

- ID 1 = tracking rojo.
- ID 2 = tracking morado.
- Si el tracking no permite determinar al atacante, usar `unknown`.

Atajos de numpad:

| Tecla | Etiqueta |
|---:|---|
| 7 | `punch_head` |
| 9 | `punch_body` |
| 4 | `kick_leg` |
| 5 | `kick_body` |
| 6 | `kick_head` |
| 1 | `other_strike` |
| 3 | `ambiguous` |
| 0 | Terminar fragmento |

También existe el botón para corregir como `no_strike` una detección binaria
previa errónea.

Reglas de etiquetado relevantes:

- Cada fragmento debe reproducirse y analizarse durante todo su intervalo de
  tiempo. La miniatura, el primer frame o un único instante no representan el
  fragmento completo. En un clip de dos segundos, la acción puede comenzar o
  culminar cerca de 0,5 s, 1,5 s o al final del clip.
- Recorrer el clip completo al menos una vez a velocidad normal y hacer una
  segunda pasada con pausa, barra temporal o avance cuadro a cuadro alrededor de
  cada movimiento candidato. No terminar el fragmento hasta revisar también los
  últimos frames.
- La técnica se decide por la trayectoria finalmente ejecutada.
- Una finta de low kick que termina en patada alta es `kick_head`.
- Si un puño apuntaba a la cabeza pero el ajuste final del atacante cambia la
  trayectoria al cuerpo, se usa `punch_body`.
- Si la trayectoria permanece hacia la cabeza y el rival se mueve, sigue siendo
  `punch_head`, salvo que la imagen no permita decidirlo.
- `punch_head` se debe anotar con un criterio inclusivo pero basado en evidencia:
  basta un intento real dirigido a la cabeza, aunque falle, sea bloqueado, roce o
  termine corto. No se exige impacto limpio. Debe existir una fase ofensiva
  reconocible —salida/aceleración del brazo, trayectoria hacia la cabeza y
  extensión o culminación coherente— y no solamente guardia, mano extendida,
  tanteo, finta incompleta, contacto de clinch o preparación.
- No volver excesivamente conservadora la selección de `punch_head`. Los jabs,
  rectos, cross, hooks y uppercuts rápidos cuentan cuando su trayectoria ofensiva
  se puede sostener al revisar el movimiento completo, incluso si un frame aislado
  no muestra la extensión máxima. A la vez, el volumen esperado nunca reemplaza
  la evidencia visual: no se debe forzar una cuota por round.
- Clinch, rodillazos, codazos o acciones fuera de las cinco categorías principales
  pueden ir a `other_strike` cuando sean visibles.
- Usar `ambiguous` por oclusión, identidad dudosa o pose/tracking insuficiente;
  no usarlo como equivalente general de `no_strike`.
- Para combinaciones, registrar cada acción visible por separado, incluso si hay
  acciones simultáneas de los dos peleadores.
- En intercambios simultáneos, evaluar por separado al ID 1 y al ID 2. Una acción
  clara de un peleador no permite ignorar la del otro; pueden existir dos o más
  registros superpuestos dentro del mismo fragmento.
- El rango temporal de cada acción debe abarcar su ejecución observable, sin
  convertir todo el fragmento en una sola acción ni recortar solamente el frame
  de impacto. En combinaciones, separar golpes distintos cuando visualmente se
  distingan sus ciclos ofensivos.
- Antes de atribuir una acción, comprobar que la caja y el esqueleto pertenecen al
  peleador correcto durante el rango elegido. Si la pose salta al rival, árbitro,
  público u otra persona, no usar esos frames como evidencia del atacante.
- Si la acción es visible en el video pero el esqueleto correcto no queda capturado
  de forma utilizable, emplear `ambiguous` o `unknown` según corresponda a la
  incertidumbre de técnica o identidad. No asignar una técnica segura al esqueleto
  equivocado.
- Marcar `no_strike` solamente después de revisar el fragmento completo y confirmar
  que no existe ningún intento ofensivo registrable de ninguno de los dos
  peleadores.

### Protocolo operativo consolidado para Gradio

Este protocolo es obligatorio para las próximas sesiones y sustituye la práctica
de decidir a partir de una portada o de muestras aisladas:

1. Confirmar pelea, round, número total de momentos y fragmento actual antes de
   iniciar el bloque.
2. Reproducir el fragmento completo de principio a fin. Observar a los dos
   peleadores, no solamente al que inicia el intercambio.
3. Hacer una segunda pasada temporal. Detenerse alrededor de cada cambio brusco de
   hombros, cadera, brazo o pierna y revisar la preparación, trayectoria y
   culminación de la acción.
4. Elaborar mentalmente una lista de acciones candidatas por peleador y en orden
   temporal. Incluir simultáneas, respuestas y combinaciones; no contar dos veces
   el mismo ciclo ofensivo.
5. Verificar durante cada candidata la continuidad de ID, caja y esqueleto. La
   identidad visual del peleador prevalece sobre una pose que se haya enganchado a
   la persona equivocada.
6. Elegir la clase por la trayectoria ejecutada y el objetivo aparente. Para
   `punch_head`, aplicar el criterio inclusivo acordado: intento real dirigido a la
   cabeza, sin exigir conexión, pero descartando guardia, fintas incompletas y
   movimientos preparatorios.
7. Ajustar inicio y fin de cada acción dentro del fragmento. Registrar cada ataque
   visible por separado y permitir rangos solapados cuando ambos peleadores atacan
   al mismo tiempo.
8. Usar `ambiguous` si la revisión completa no resuelve la técnica, la identidad o
   la validez de la pose. Usar `unknown` cuando la técnica sea identificable pero
   no se pueda atribuir con seguridad a un ID.
9. Solo después de revisar todo el intervalo, terminar el fragmento o corregirlo
   como `no_strike`. Comprobar que el contador de terminados avanzó exactamente una
   unidad y que se conservaron todas las acciones añadidas.
10. Continuar con el siguiente fragmento sin cerrar el anotador, cambiar de flujo
    ni detener la sesión hasta completar el bloque indicado por el usuario o
    recibir una orden explícita de parar.

### Control de calidad durante y después de cada bloque

- Trabajar en bloques numerados con meta explícita, por ejemplo, fragmentos 1–40
  o hasta 81/81. El contador de Gradio es la fuente de verdad del progreso.
- Hacer una revisión rápida de consistencia después de cada 10–20 fragmentos sin
  abandonar la sesión: buscar omisiones, duplicados, rangos demasiado largos,
  identidades intercambiadas y uso indebido de `no_strike` o `ambiguous`.
- Al completar un round, revisar una muestra estratificada de todas las clases y
  dar prioridad a `punch_head`, simultáneas, acciones muy rápidas y casos con pose
  inestable. Si el usuario solicita una auditoría total de una clase, revisar todos
  sus ejemplos recorriendo nuevamente cada clip completo.
- Reportar al final los conteos por clase, por ID, `ambiguous`, `unknown` y
  correcciones `no_strike`. Separar claramente acciones de fragmentos: un
  fragmento puede contener cero, una o varias acciones.
- No inferir calidad por el volumen. Un round activo puede producir muchas
  acciones y uno táctico menos; el criterio visual debe mantenerse constante.
- No crear fragmentaciones, hojas de contacto, PDFs u otros artefactos auxiliares
  durante la anotación salvo petición expresa. Usar el reproductor y los controles
  temporales del anotador como flujo principal.
- No afirmar que un round está completo ni entregar conteos finales si el contador
  no alcanzó la meta o si quedaron fragmentos sin revisar en toda su duración.

### Referencia de calibración acordada para `punch_head`

La revisión manual del round 2 de Zhang Weili vs Joanna Jedrzejczyk sirvió para
ajustar el criterio. Una primera lectura demasiado permisiva produjo 83
`punch_head`; una corrección excesivamente conservadora lo redujo a 26. La
revisión intermedia aceptada dejó 48 ejemplos después de volver a mirar la
trayectoria completa de los puños.

Este resultado define el tipo de equilibrio buscado, no una cifra objetivo para
otros rounds. Se deben conservar ataques reales a la cabeza aunque fallen o sean
bloqueados, y retirar guardias, tanteos, fintas sin culminación, preparaciones y
movimientos cuya trayectoria no sustente un ataque. Nunca se debe subir o bajar un
conteo para aproximarlo artificialmente a 48.

## Dataset de acciones disponible

Snapshot del corpus V2 generado el 22 de octubre:

- 34 rounds con anotación de técnicas terminada.
- 8 peleas.
- 1.777 muestras positivas.
- 32.190 muestras `no_strike` confirmadas.
- 837 registros `ambiguous` excluidos del entrenamiento.
- 25 acciones con atacante desconocido excluidas.
- 1.751/1.777 acciones con supervisión completa.
- 26 acciones con supervisión parcial por ambigüedad/solapamiento.

Conteos humanos por etiqueta:

| Etiqueta | Cantidad |
|---|---:|
| `punch_head` | 1.155 |
| `kick_leg` | 265 |
| `punch_body` | 158 |
| `kick_body` | 123 |
| `kick_head` | 90 |
| `other_strike` | 11 |
| `ambiguous` | 837 |

El desequilibrio es severo. `punch_head` domina y las clases corporales y
`kick_head` tienen pocos ejemplos. `other_strike` es demasiado heterogénea y
escasa para esperar buen rendimiento por ahora.

Archivos derivados:

```text
data/processed/datasets/action_recognition_v2/report.json
data/processed/datasets/action_recognition_v2/samples.jsonl
data/processed/datasets/action_recognition_v2/audit_records.jsonl
```

Existen 56 rounds con anotación binaria y 34 con anotación de técnicas, por lo
que quedan 22 pendientes. Los grupos útiles pendientes incluyen:

- Joanna Jedrzejczyk vs Jessica Andrade: 5 rounds, `low_kicks_women`.
- Zhang Weili vs Joanna Jedrzejczyk: 5 rounds, `aggressive_women`.
- Amanda Nunes vs Valentina Shevchenko 2: 5 rounds, `passive_women`.
- Rafael Fiziev vs Bobby Green: 3 rounds, `low_kicks_men`.
- Sean O'Malley vs Aiemann Zahabi: 2 rounds, `low_kicks_men`.
- Dos archivos aislados pendientes de revisar por nombre/duplicación:
  `topuria_gaethje_round2` y Yan–Sandhagen R2.

## Torneo de reconocimiento de técnicas

Se probaron regresión logística, Extra Trees, TCN, ST-GCN compacto y X3D-S con
cabeza lineal, usando folds separados por pelea.

Resultado inicial del torneo:

- X3D-S: macro-F1 principal 0,567, pero excluyó cerca del 20 % de los clips al
  no conseguir recortes completos del atacante.
- TCN: macro-F1 principal bruto 0,487.
- En exactamente los clips que X3D pudo procesar, TCN obtuvo 0,718 frente a
  0,567 de X3D.

Por cobertura, coste y comparación equivalente, TCN fue seleccionado como
familia principal. Los valores son de desarrollo, no acreditan producción.

La sintonización TCN de fold 0 produjo históricamente 0,575 de macro-F1 bajo la
política V1 de máscaras parciales. Ese valor no se debe comparar directamente
con los experimentos V2 de supervisión completa.

Candidato histórico usado en Gradio:

```text
outputs/action-recognition/tcn-optuna-fold0/trials/trial-018
```

## Reparación de inferencia y HUD

La primera versión de Gradio generaba etiquetas con retraso visual, eventos de
varios segundos, varias técnicas superpuestas para un mismo ID y exceso de
`punch_head`/`other_strike`.

La inferencia fue reparada para:

- Anclar la decisión al centro de la ventana, no a su final.
- Limitar los eventos visuales a aproximadamente 0,9 s.
- Exigir probabilidad mínima 0,50 y cobertura de pose mínima 0,65.
- Mostrar una sola técnica principal por peleador en cada instante.
- Evitar superposiciones de técnicas diferentes para el mismo ID.
- Mostrar tarjetas cerca de la caja del peleador, ID 1 rojo/ID 2 morado.
- Guardar candidaturas crudas y una línea temporal de diagnóstico en
  `action_timeline.csv`.

En los primeros 900 frames (30 s) de:

```text
tests/videos_tests/DuPlessisvsUsman _UFCOklahomaCityCUTTED.mp4
```

el TCN V1 reparado produjo 18 eventos cortos, con duración máxima aproximada de
0,83 s. La lectura visual mejoró claramente, pero todavía se pierden algunos
jabs/rectos/cross y persisten confusiones entre puños y patadas.

Probador de acciones:

```powershell
conda activate strikevision_ML
python -s scripts/ml/run_action_inference_app.py
```

El selector de Gradio muestra candidatos históricos y el candidato
`TCN etiquetas V2` para comparación manual.

## Experimentos V2 y conclusiones

### TCN jerárquico con puerta acción/no-acción

Se creó una variante con puerta explícita acción/no-acción, atención temporal y
clasificación posterior de técnica. Se ejecutaron cinco ensayos solamente en
fold 0.

- Score original de Optuna: 0,340; mezclaba métricas separadas.
- Macro-F1 real de técnicas después de la puerta recalibrada: 0,251.
- TCN V1 sobre exactamente las mismas etiquetas V2: 0,317 con umbrales guardados
  y 0,328 con umbrales recalibrados.
- En 30 s del video de prueba produjo 42 eventos frente a 18 del V1, con exceso
  de `punch_head`.

Decisión: variante jerárquica rechazada. No ejecutar folds 1 y 2. La selección
de futuros modelos jerárquicos ya fue corregida para usar el F1 final posterior
a la puerta.

Artefactos:

```text
outputs/action-recognition-v2/hierarchical_tcn-optuna-fold0_v2
```

### TCN tradicional entrenado con etiquetas V2 completas

Se reutilizó la arquitectura TCN y los mejores hiperparámetros V1, cambiando
solamente la política de supervisión:

```text
outputs/action-recognition-v2/tcn-labels-v2-fold0
```

Resultados:

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| `punch_head` | 0,44 | 0,60 | 0,51 |
| `punch_body` | 0,12 | 0,45 | 0,19 |
| `kick_leg` | 0,31 | 0,22 | 0,26 |
| `kick_body` | 0,44 | 0,41 | 0,42 |
| `kick_head` | 0,44 | 0,19 | 0,27 |
| `other_strike` | 0,00 | 0,00 | 0,00 |

- Macro-F1 principal: 0,331.
- TCN V1 recalibrado sobre las mismas etiquetas: 0,328.
- Diferencia: +0,003, insuficiente para reemplazar el modelo vigente.
- En 900 frames produjo 22 eventos: 20 `punch_head` y 2 `kick_leg`.
- Recuperó algunos jabs claros, pero añadió un falso positivo, dividió una
  secuencia prolongada y confundió un puño con `kick_leg`.

Decisión: conservarlo como candidato A/B en Gradio, sin promoverlo todavía.

Vista previa guardada:

```text
outputs/action-recognition/gradio/31b65d4eae164a4b9183c4e2bb110162/action_preview.mp4
```

## Qué funciona y qué no está resuelto

Funciona:

- Carga de rounds y selección de prefijo temporal hasta 9.000 frames.
- Detección de peleadores, tracking unido y pose.
- Strike/no-strike global con resultado visual útil.
- Anotación de múltiples técnicas y atacantes por fragmento.
- Modelo TCN de técnicas ejecutable en Gradio.
- HUD más corto y legible, sin múltiples etiquetas simultáneas ilegibles para
  el mismo ID.
- Exportación de JSON, probabilidades, eventos y CSV temporal.

Pendiente:

- Mejorar precisión y recall de clases minoritarias.
- Reducir falsos `punch_head` durante fintas, guardia y preparación.
- Recuperar jabs, rectos y cross rápidos sin saturar la salida.
- Corregir confusiones puño/patada.
- Evitar división o unión incorrecta de combinaciones.
- Evaluar atribución correcta al ID y fallos por árbitro/oclusiones.
- Medir falsos eventos por minuto, desfase temporal y rendimiento por pelea,
  sexo y estilo.
- Entrenar un modelo final congelado y evaluarlo sobre peleas completamente
  independientes antes de hablar de producción.

## Plan de desarrollo fortalecido

No repetir el torneo ni entrenar inmediatamente. Primero se debe aumentar y
auditar el corpus con el protocolo temporal consolidado. Cada etapa tiene una
condición de salida para evitar entrenar con anotaciones incompletas o aplicar
criterios distintos entre rounds.

### Etapa 1: cerrar la prueba de anotación actual

1. Completar los cinco rounds de Zhang vs Joanna con Gradio, respetando los
   bloques que indique el usuario y sin saltar fragmentos.
2. Reproducir completos todos los fragmentos, registrar acciones simultáneas y
   validar caja, pose e identidad antes de terminar cada uno.
3. Al finalizar cada round, exportar el conteo por clase e ID y revisar muestras
   estratificadas. Auditar de forma reforzada `punch_head` con la calibración
   acordada del round 2.
4. Considerar cerrado un round solamente cuando Gradio muestre todos sus momentos
   terminados y no existan fragmentos revisados únicamente por portada.

### Etapa 2: ampliar clases débiles y negativos difíciles

Orden actualizado recomendado:

1. Terminar Zhang vs Joanna, cinco rounds, ya iniciada como prueba de criterio.
2. Joanna vs Andrade, cinco rounds, con prioridad adicional para `kick_leg` y
   ataques al cuerpo.
3. Nunes vs Shevchenko 2, cinco rounds; reservar la pelea completa como test
   independiente y excluirla de entrenamiento y calibración.

Metas orientativas del bloque, nunca cuotas que justifiquen etiquetas dudosas:

- Añadir 80–120 `punch_body`.
- Añadir 80–120 `kick_body`.
- Añadir 80–120 `kick_head`.
- Añadir al menos 100 correcciones `no_strike` difíciles: fintas, amagos,
  cambios de guardia, pasos y entradas sin golpe.
- Conservar `punch_head` reales bajo el criterio acordado, sin inflar la clase
  dominante ni eliminar jabs y rectos válidos para equilibrar artificialmente.

### Etapa 3: auditoría antes del corpus V3

1. Generar un reporte por pelea, round, clase, atacante y duración de los rangos.
2. Buscar duplicados, solapamientos anómalos, rangos que ocupen todo el fragmento,
   IDs intercambiados y fragmentos con muchas acciones o ninguna acción.
3. Auditar una muestra aleatoria y otra dirigida por clase minoritaria, además de
   todos los casos `unknown` y una muestra amplia de `ambiguous`.
4. Comparar distribuciones entre anotadores o sesiones para detectar deriva de
   criterio. Usar la auditoría del round 2 como referencia cualitativa de
   `punch_head`, no como objetivo numérico.
5. Corregir en Gradio los errores confirmados y preservar una copia/versionado de
   las anotaciones previas antes de cualquier transformación masiva.

La etapa termina cuando el usuario apruebe la calidad de las muestras y los
conteos sean reproducibles desde los JSON guardados.

### Etapa 4: construir V3 y entrenar un candidato controlado

1. Congelar el manifiesto del corpus V3 y documentar exclusiones.
2. Mantener Nunes vs Shevchenko 2 completamente fuera del entrenamiento.
3. Entrenar un solo candidato TCN controlado con semillas, configuración y
   umbrales registrados. Los entrenamientos largos los ejecuta el usuario.
4. No lanzar otro Optuna ni un torneo general sin evidencia de que el nuevo corpus
   mejora cobertura y consistencia.

### Etapa 5: evaluación de promoción

- Medir macro-F1 y F1 por clase, pero también cobertura, falsos eventos por
  minuto, desfase temporal, atribución de ID y rendimiento por pelea/estilo.
- Revisar visualmente errores de `punch_head`, clases corporales, patadas y
  acciones simultáneas sobre peleas no usadas en entrenamiento.
- Comparar contra el candidato vigente en exactamente los mismos clips.
- Promover un modelo solo si mejora métricas y calidad visual sin aumentar de
  manera inaceptable falsos positivos o pérdida de acciones rápidas.

## Vía alternativa cuando la interacción de Gradio no permite anotar con fiabilidad

Esta vía se puede usar cuando Gradio carga el round pero el control del video,
el avance cuadro a cuadro o la automatización de los componentes falla. No es
una excusa para omitir la revisión visual: cambia el mecanismo de escritura, no
el criterio de anotación.

### Revisión visual fuera del reproductor

1. Generar hojas ordenadas con **todos los cuadros consecutivos** de cada
   fragmento, conservando en cada cuadro el número absoluto de frame, el número
   de fragmento y la superposición de pose/tracking. En la revisión del round 3
   de Zhang vs Joanna se usaron las hojas `tmp_round3_audit/full_<evento>_<hoja>.jpg`.
2. Revisar cada fragmento completo, desde el primer hasta el último cuadro. Para
   cada intento se debe localizar preparación/inicio, extensión o contacto y
   recuperación. No decidir únicamente con la primera imagen ni con la hoja de
   resumen `event_<n>.jpg`.
3. Revisar a las dos peleadoras en cada secuencia. Si atacan simultáneamente,
   guardar dos filas con rangos solapados y su atacante correspondiente.
4. Cuando el tracker salte al árbitro, al público o intercambie colores, usar la
   identidad visual de las peleadoras para decidir quién atacó. Si ni siquiera
   con los cuadros consecutivos se puede resolver atacante, técnica u objetivo,
   usar `ambiguous`/`unknown`; no inventar una clase.
5. Marcar una corrección `no_strike` sólo después de comprobar el intervalo
   completo y confirmar que no hay ningún intento ofensivo.

### Escritura mediante el mismo backend de Gradio

El anotador de Gradio guarda mediante
`src/ufc_tracker/annotations/actions.py::ActionStore`. Si la UI no permite
registrar de forma estable muchas decisiones, se puede usar directamente esa
clase sobre el JSON de salida separado. No se debe editar el JSON a mano.

Para cada acción, llamar `ActionStore.commit()` con la revisión vigente,
`event_id`, atacante, frames inclusivos y clase. Registrar primero todas las
acciones del fragmento y después cerrarlo con `complete=True`. Para un falso
positivo confirmado, usar `source_correction=True` y `complete=True`. Leer de
nuevo el estado antes de cada commit conserva el control de revisión optimista y
evita clics/operaciones duplicadas.

Controles obligatorios antes de declarar el round terminado:

- trabajar sobre un directorio de salida separado y no sobrescribir las
  anotaciones originales;
- comprobar que cada rango está dentro de los límites del evento;
- comprobar duplicados exactos, acciones simultáneas y correcciones
  `no_strike`;
- comparar `len(completed)` con el número de eventos de `ActionStore.events`;
- recargar el mismo round en Gradio y confirmar el mensaje **Round terminado.
  Todas las anotaciones están guardadas**;
- calcular los conteos desde el JSON persistido, no desde memoria ni desde una
  estimación visual.

### Aplicación en Zhang vs Joanna, round 3

Para completar los fragmentos 50-81 se revisaron hojas de cuadros consecutivos,
se escribieron las decisiones con `ActionStore` y luego se recargó el round en
Gradio. El resultado persistido fue 81/81 fragmentos, 138 acciones y 16
correcciones `no_strike`. El script de sesión fue
`tmp_round3_finish_50_81.py`; debe tratarse como herramienta de auditoría de esa
sesión, no como anotación genérica reutilizable sin volver a mirar los cuadros.

Los conteos reproducibles del round completo quedaron así:

| Clase | Registros |
|---|---:|
| `punch_head` | 96 |
| `kick_body` | 19 |
| `kick_leg` | 12 |
| `ambiguous` | 6 |
| `kick_head` | 2 |
| `other_strike` | 2 |
| `punch_body` | 1 |
| **Total** | **138** |

Desde el fragmento 40 hasta el 81 hay 81 acciones: 56 `punch_head`, 10
`kick_leg`, 9 `kick_body`, 3 `ambiguous`, 2 `other_strike` y 1 `punch_body`.
Además hay siete correcciones `no_strike` en ese tramo; estas correcciones no se
incluyen en el total de acciones.

## Archivos principales

```text
docs/PLAN-ACTION-RECOGNITION.md
docs/STRIKE-GRADIO-TESTER.md
docs/STRIKE-ML-MANUAL.md
configs/models/action_recognition_v1.yaml
configs/models/action_recognition_v2.yaml
configs/models/action_training_v1.yaml
configs/models/action_training_v2.yaml
scripts/annotation/run_action_annotator.py
scripts/ml/run_action_inference_app.py
scripts/ml/run_action_recognition.py
scripts/ml/run_action_tournament.ps1
src/ufc_tracker/action_recognition/
src/ufc_tracker/ui/action_annotation_app.py
src/ufc_tracker/ui/action_inference_app.py
```

## Estado del repositorio

La implementación de action recognition y varios documentos/configuraciones
siguen sin commit y aparecen como archivos nuevos en `git status`. También hay
cambios en `.gitignore` y `src/ufc_tracker/ui/action_annotation_app.py`. Antes de
borrar, resetear o preparar un commit, revisar el diff y preservar anotaciones,
modelos y artefactos que el usuario haya decidido conservar.

## Actualización del 28 de octubre: primera ampliación con mujeres

Se terminaron los cinco rounds de Zhang Weili vs Joanna Jedrzejczyk en
`data/annotations/strike_actions_zhang_joanna_trial`. Todos los fragmentos están
completos y los hashes corresponden a sus fuentes binarias.

La configuración V3 combina esa carpeta con `strike_actions_v1` sin copiar ni
sobrescribir anotaciones. Conserva el test anterior y los folds de desarrollo;
`zhang_joanna_1` entra únicamente al entrenamiento. Esto permite comparar contra
V2 sobre exactamente las mismas 1.149 muestras de validación del fold 0.

El entrenamiento fold 0 crece de 943 a 1.486 positivos y de 2.829 a 4.458
muestras después del muestreo de negativos. Aporte de etiquetas positivas al
entrenamiento: +430 `punch_head`, +62 `kick_body`, +50 `kick_leg`, +6
`kick_head`, +4 `punch_body` y +3 `other_strike`. La ampliación aporta variedad
femenina y `kick_body`, pero aumenta el desequilibrio hacia `punch_head` y casi
no refuerza `punch_body`/`kick_head`.

Archivos V3:

```text
configs/models/action_recognition_v3.yaml
configs/models/action_training_v3.yaml
data/processed/datasets/action_recognition_v3/
```

### Corrección crítica del caché temporal

El 28 de octubre se detectó que `build_pose_feature_cache` agrupaba muestras por
archivo de pose, mientras el entrenamiento temporal asumía que el arreglo seguía
en orden `train + validation`. Esto mezcló ambos subconjuntos en los TCN
anteriores y produjo reportes/comparaciones inválidos. Los valores históricos de
TCN no deben utilizarse para seleccionar un modelo hasta reentrenar V2 y V3 con
el orden corregido. El caché ahora conserva el orden solicitado, existe una
prueba de regresión y el comparador rechaza artefactos creados con el contrato
antiguo.

### Comparación limpia V2 vs V3 después de corregir el caché

Se reentrenaron ambos candidatos desde cero con la misma arquitectura,
hiperparámetros, semilla y las mismas 1.149 muestras de validación:

- V2 limpio: macro-F1 0,3146; macro-AP 0,2332.
- V3 con Zhang–Joanna: macro-F1 0,3452; macro-AP 0,2481.
- Mejora absoluta: +0,0305 macro-F1; mejora relativa aproximada: +9,7 %.

F1 V2 -> V3: `punch_head` 0,55 -> 0,58; `punch_body` 0,23 -> 0,25;
`kick_leg` 0,36 -> 0,35; `kick_body` 0,12 -> 0,16; `kick_head` 0,32 ->
0,38. `other_strike` permanece en 0 por su soporte escaso.

En los primeros 900 frames de Du Plessis–Usman, V2 limpio produjo 24 eventos
y V3 produjo 30. V3 recupera acciones y elimina alguna activación dudosa, pero
también añade falsos/eventos discutibles. La ampliación de datos ayuda, aunque
el modelo todavía no está listo para producción. Continuar con anotación
dirigida a clases minoritarias y negativos difíciles; evitar otra tanda dominada
por `punch_head`.

## Regla de entrega por round (29 de octubre de 2026)

Cada round anotado debe cerrarse con dos entregables obligatorios:

1. Resultados y tabla de auditoría: fragmentos completados, acciones por clase,
   acciones por peleador, correcciones `no_strike`, simultaneidades y detalle por
   fragmento.
2. PDF visual de control con los fragmentos catalogados por clase. Debe incluir
   una imagen representativa de cada acción, sus frames, atacante y fragmento;
   además debe mostrar en una sección propia los fragmentos corregidos como
   `no_strike`.

El PDF se renderiza a imágenes y se inspecciona antes de entregarlo. Esta regla
aplica de ahora en adelante a cada round, aunque el usuario no vuelva a pedir el
PDF explícitamente.

## Actualización del 30 de octubre: V4 con Joanna–Andrade

Se incorporaron como entrenamiento los cinco rounds completos de Joanna
Jedrzejczyk vs Jessica Andrade desde
`data/annotations/strike_actions_joanna_andrade_trial`. El preparador oficial
validó completitud, correspondencia con las fuentes binarias y contratos de los
cinco rounds.

La pelea aporta 409 acciones anotadas: 292 `punch_head`, 46 `kick_leg`, 42
`kick_body`, 25 `other_strike`, 3 `kick_head` y 1 `punch_body`, además de 79
correcciones explícitas a `no_strike`. El corpus V4 queda con 44 rounds, 10
peleas, 2.729 positivos, 40.688 muestras confirmadas `no_strike` y 894
`ambiguous` excluidos del aprendizaje.

Archivos V4:

```text
configs/models/action_recognition_v4.yaml
configs/models/action_training_v4.yaml
data/processed/datasets/action_recognition_v4/
```

En fold 0, V4 utiliza 5.685 muestras de entrenamiento frente a 4.458 en V3.
Las 1.149 muestras de validación son exactamente las mismas y tienen el mismo
SHA-256 de IDs (`b3550bb841911798a22219c61809ea17be6e855dfc9cda7e21c4e6160f286996`).
`joanna_andrade` y `zhang_joanna_1` son exclusivamente de entrenamiento. La
pelea Nunes–Shevchenko 2 debe permanecer independiente y nunca entrar al
entrenamiento; se reserva para evaluación externa posterior.

### Comparación controlada V3 vs V4

V4 se reentrenó con el mismo TCN y la misma sintonización ganadora de V3 (96
canales, 4 bloques, `dropout=0.1930`, `batch_size=16`, misma semilla) y sobre
las mismas 1.149 muestras de validación. Por tanto, esta es una comparación
válida del aporte de Joanna–Andrade:

- V3: macro-F1 0,3452; macro-AP 0,2481.
- V4: macro-F1 0,3528; macro-AP 0,2784.
- Diferencia: +0,0076 macro-F1 (+2,2 %) y +0,0304 macro-AP (+12,2 %).

Por clase, F1 V3 -> V4: `punch_head` 0,581 -> 0,571; `punch_body` 0,255 ->
0,224; `kick_leg` 0,350 -> 0,409; `kick_body` 0,161 -> 0,197; `kick_head`
0,378 -> 0,364. Sin embargo, el AP de `punch_head` y `kick_head` también
mejora (0,548 -> 0,592 y 0,208 -> 0,256); sus pequeños descensos de F1 son de
umbral, no una evidencia de que el ordenamiento de confianza haya empeorado.

El candidato líder actual es
`outputs/action-recognition-v4/tcn-women-v4-fold0-cachefix`. La pelea restante
se debe anotar y mantener completamente fuera del entrenamiento y de la
sintonización: es la primera prueba externa honesta del candidato V4.


## Continuidad de anotacion directa: Volkanovski–Holloway I, round 1

Instruccion explicita vigente del usuario: completar el round sin cerrar respuestas por tandas arbitrarias. Guardar puntos de control durante el trabajo y continuar. Una actualizacion de progreso no es un motivo para finalizar el turno. Ante compactacion, recuperar el punto de control y seguir; no reiniciar ni inventar revision. No es posible cambiar el tamano de la ventana de contexto mediante este archivo.

Estado comprobado: hojas 1–475 de 562 revisadas y guardadas; siguiente hoja 476 (frame 7600). Quedan 87 hojas. Fuente de verdad: JSON en data/annotations/action_strikes_direct_v1 y sus reviewed_ranges; los reportes anteriores pueden estar desactualizados.

Descartar las detecciones de pose del arbitro; no detener la anotacion por ellas, no asignarlas a atletas y no fabricarlas como poses validas. Conservar acciones simultaneas como eventos independientes. Revisar todos los cuadros de cada hoja. No usar el clasificador v4 para inventar etiquetas. No declarar listo para entrenamiento sin verificar el adaptador y la geometria. Entregar conteo final por clase solamente tras completar y validar cobertura hasta frame 8990. Sin PDF en esta tanda por preferencia vigente.

Actualizacion: revision visual completada hasta hoja 562, frame 8990. Fuente y conteos comprobados en output/annotation_audits/volkanovski_holloway_i/round1_reporte_final.json. Integracion de poses para entrenamiento pendiente.

### Continuidad: Volkanovski-Holloway I, round 2 (2026-11-05)

- Tanda solicitada completada: hojas 1-144 de 576, cuadros originales 0-2303 inclusive, 2304 cuadros revisados visualmente en hojas consecutivas de 16 imagenes.
- Siguiente hoja: 145 (frame 2304). Quedan 432 hojas del round; no declarar el round completo.
- Fuente: data/annotations/action_strikes_direct_v1/alexander_volkanovski_vs_max_holloway_i_ufc_245__t_5G7bGTPxs__round2.json, revision 177.
- Conteo: punch_head 16, punch_body 3, kick_leg 7, kick_body 1, kick_head 0, other_strike 0; ambiguous 5. Total 32 registros, 27 acciones confirmadas.
- Reportes y punto de reanudacion: output/annotation_audits/volkanovski_holloway_i/round2_001_144_reporte.md, round2_001_144_reporte.json y round2_001_144_progress.json.
- Conservar acciones simultaneas como registros separados. No transformar medicion con palma abierta o amagos en golpes. No forzar objetivo de patadas fuera del encuadre.
- Las hojas de esta tanda son RGB sin pose superpuesta. track_slot=null: enlace geometrico y validacion de poses pendientes, training_ready=false. Descartar detecciones del arbitro al integrar poses, segun regla del usuario. No afirmar que los errores de tracking estan corregidos o auditados sobre estas hojas sin overlay.
- Validado: 144 rangos de hoja sin huecos 0-2303, sin acciones duplicadas exactas y evidencia existente. Sin PDF en esta tanda.


## Continuidad: Volkanovski-Holloway I R2, hojas 145-284

Tanda de 140 hojas completada con revision visual de todos sus 2240 cuadros (2304-4543). Acumulado 284/576; continuar desde hoja 285, quedan 292. Reporte: output/annotation_audits/volkanovski_holloway_i/round2_145_284_reporte.md y .json. Revision 373. Conteos acumulados: {"punch_head": 56, "punch_body": 6, "kick_leg": 14, "kick_body": 1, "kick_head": 0, "other_strike": 0, "ambiguous": 10}. Golpes simultaneos en registros separados. Sin PDF en esta tanda. Poses aun sin vincular ni validar geometricamente: training_ready=false, track_slot null; descartar poses del arbitro.


## Continuidad 2026-11-06 - Volkanovski Holloway I R2, hojas 305-444

Tanda de 140 hojas completada mediante revision visual de todos los 16 cuadros por hoja. Cobertura acumulada 444/576 (frames 0-7103); continuar en hoja 445, frame 7104. Pendientes 132 hojas. Reporte: `output/annotation_audits/volkanovski_holloway_i/round2_305_444_reporte.md` y JSON equivalente. Progreso: `round2_305_444_progress.json`. Sin PDF por instruccion vigente de tandas.

Conteos tanda: {'kick_leg': 2, 'punch_head': 26, 'ambiguous': 3, 'kick_body': 1, 'punch_body': 1}; acumulados: {'punch_head': 84, 'ambiguous': 14, 'kick_leg': 16, 'punch_body': 7, 'kick_body': 2}.

Hojas 305-444 revisadas en RGB por identidad fisica; sin overlay de pose. Vinculacion de track_slot y auditoria de pose pendientes; descartar capturas del arbitro. No declarar datos listos para entrenamiento v4.


## Continuidad 2026-11-06: Volkanovski-Holloway I, round 2 completo en RGB

Revision visual completada hasta la hoja 576/576 (9201 cuadros). Tanda final 445-576: 132 hojas. Totales: punch_head=113, punch_body=10, kick_leg=22, kick_body=3, kick_head=0, other_strike=1, ambiguous=14. Reporte: `output/annotation_audits/volkanovski_holloway_i/round2_completo_reporte.md`; JSON de acciones en `data/annotations/action_strikes_direct_v1/alexander_volkanovski_vs_max_holloway_i_ufc_245__t_5G7bGTPxs__round2.json`. Acciones simultaneas separadas por identidad fisica. Pendiente vinculacion a poses y validacion de tracking: training_ready=false; no afirmar listo para v4. Descartar poses del arbitro. No se inicio round 3.


## Avance 2026-11-06: Volkanovski-Holloway I, round 3, primeras 150 hojas

Revision visual completada de hojas 1-150/570, cuadros 0-2399 (2400 cuadros). Restan 420 hojas; continuar en hoja 151, cuadro 2400. Conteos: punch_head=22, punch_body=2, kick_leg=6, kick_body=2, kick_head=0, other_strike=0, ambiguous=2. Total 34 registros, 32 acciones clasificadas. Acciones simultaneas independientes; sin PDF para esta tanda. Reporte: output/annotation_audits/volkanovski_holloway_i/round3_001_150_reporte.md y .json. Guardado mediante DirectActionStore, revision 185. Cobertura continua, evidencia existente y ausencia de duplicados exactos comprobadas.

IMPORTANTE: anotacion visual RGB; track_slot=null y training_ready=false. Pendiente vincular/verificar poses por identidad para entrenamiento v4; descartar poses del arbitro. No afirmar que tracking o keypoints esten validados. No usar modelo v4 para generar estas etiquetas.


## Volkanovski-Holloway I R3: hojas 151-290 completadas

Revision visual de 140 hojas adicionales y sus 2240 cuadros (2400-4639). Avance acumulado 290/570; restan 280; siguiente hoja 291, cuadro 4640. Conteos de tanda: {"punch_head": 28, "punch_body": 0, "kick_leg": 4, "kick_body": 1, "kick_head": 0, "other_strike": 0, "ambiguous": 4}; acumulados: {"punch_head": 50, "punch_body": 2, "kick_leg": 10, "kick_body": 3, "kick_head": 0, "other_strike": 0, "ambiguous": 6}. Reporte output/annotation_audits/volkanovski_holloway_i/round3_151_290_reporte.md y .json. Cobertura continua y evidencia comprobadas; sin duplicados exactos. Sin PDF por preferencia de tandas. Poses pendientes, track_slot=null, training_ready=false; descartar poses del arbitro al vincular.


### 2026-11-06 — Volkanovski–Holloway I, R3: hojas 291–430

Se revisaron visualmente las 140 hojas completas (cuadros 4640–6879), conservando acciones simultáneas por identidad física. Acumulado 430/570 hojas; continuar en 431, cuadro 6880; faltan 140 hojas.

Conteos tanda: {"punch_head": 39, "punch_body": 2, "kick_leg": 4, "kick_body": 1, "kick_head": 0, "other_strike": 1, "ambiguous": 3}. Acumulado: {"punch_head": 89, "punch_body": 4, "kick_leg": 14, "kick_body": 4, "kick_head": 0, "other_strike": 1, "ambiguous": 9}. Revisión 554.

Reporte: output/annotation_audits/volkanovski_holloway_i/round3_291_430_reporte.md y .json. Sin PDF para esta tanda. Poses no visibles: track_slot null, training_ready false; vinculación de poses y validación de tracking pendientes. Descartar poses del árbitro. No convertir ambigüedad en positivo.


## 2026-11-06 - Volkanovski-Holloway I R3: revision RGB completa

Hojas 431-570 terminadas; acumulado 570/570 (cuadros 0-9109). 176 registros: punch_head 129, punch_body 6, kick_leg 22, kick_body 6, kick_head 1, other_strike 2, ambiguous 10. 166 acciones positivas. Simultaneas separadas. Reporte: output/annotation_audits/volkanovski_holloway_i/round3_completo_reporte.md y JSON. JSON canonico mediante DirectActionStore. Asociacion con poses y validacion de tracking pendientes: track_slot null, training_ready false. Descartar poses del arbitro. Revision RGB completa no equivale a dataset v4 listo.


## 2026-11-06 - Volkanovski-Holloway I R4: hojas 1-140 completas

Revision visual de las primeras 140/579 hojas y sus 2240 cuadros (0-2239). Continuar desde hoja 141, cuadro 2240; restan 439 hojas. Conteos: {"punch_head": 20, "punch_body": 1, "kick_leg": 5, "kick_body": 2, "kick_head": 1, "other_strike": 0, "ambiguous": 1}. 29 acciones positivas y 1 ambigua excluida. Simultaneas separadas. Reporte: output/annotation_audits/volkanovski_holloway_i/round4_001_140_reporte.md y JSON. Revision 171. Cobertura continua, evidencia existente y cero duplicados exactos comprobados. Sin PDF para esta tanda. RGB sin overlay: track_slot=null, training_ready=false; asociacion de poses y auditoria de tracking pendientes. Descartar poses del arbitro.


## 2026-11-06 - Volkanovski-Holloway I, R4: hojas 141-280 completadas

Revision visual de 140 hojas completas, frames 2240-4479. Cobertura acumulada 280/579 hojas (frames 0-4479). Proxima hoja 281, frame 4480; restan 299.

| Clase | Esta tanda | Acumulado |
|---|---:|---:|
| punch_head | 32 | 52 |
| punch_body | 0 | 1 |
| kick_leg | 4 | 9 |
| kick_body | 1 | 3 |
| kick_head | 0 | 1 |
| other_strike | 0 | 0 |
| ambiguous | 2 | 3 |


39 registros nuevos: 37 positivos y 2 ambiguos. Acumulado: 69 registros, 66 positivos y 3 ambiguos. Acciones simultaneas separadas por identidad fisica. No contar medicion, paradas, amagos ni pummeling como golpes.

JSON canonico revision 351. Cobertura continua, evidencias verificadas y cero duplicados exactos. Asociacion con poses y validacion de tracking pendientes, track_slot=null y training_ready=false; descartar poses del arbitro. No producir negativos automaticos.

Reportes: output/annotation_audits/volkanovski_holloway_i/round4_141_280_reporte.md y .json. Sin PDF por tratarse de tanda parcial.


## 2026-11-06 - Volkanovski-Holloway I R4: hojas 281-420 completas

Revision visual de 140 hojas completas, 2240 cuadros (4480-6719). Acumulado 420/579; continuar en hoja 421, frame 6720. Restan 159 hojas.

| Clase | Nuevas | Acumuladas |
|---|---:|---:|
| punch_head | 43 | 95 |
| punch_body | 1 | 2 |
| kick_leg | 5 | 14 |
| kick_body | 1 | 4 |
| kick_head | 0 | 1 |
| other_strike | 0 | 0 |
| ambiguous | 2 | 5 |


51? Registros verificados en reporte JSON; revision 544. Acciones simultaneas separadas. Cobertura continua y evidencia verificadas, cero duplicados exactos. RGB sin overlay: track_slot=null, training_ready=false; poses y tracking pendientes. Descartar poses del arbitro. Sin PDF para tanda parcial. Reportes: output/annotation_audits/volkanovski_holloway_i/round4_281_420_reporte.md y .json.


## 2026-11-06 - Volkanovski-Holloway I R4 completo, 579/579

Ultimas 159 hojas (421-579), cuadros 6720-9260, revisadas visualmente. Cobertura RGB total 9261 cuadros. Restan 0 hojas.

| Clase | Ultimas 159 hojas | Total round |
|---|---:|---:|
| punch_head | 42 | 137 |
| punch_body | 5 | 7 |
| kick_leg | 3 | 17 |
| kick_body | 1 | 5 |
| kick_head | 0 | 1 |
| other_strike | 0 | 0 |
| ambiguous | 5 | 10 |

177 registros, 167 positivos y 10 ambiguos. Revision 760. Simultaneas separadas; evidencia y cobertura verificadas, sin duplicados exactos. Poses y tracking pendientes, track_slot=null, training_ready=false; descartar poses del arbitro. Reporte: output/annotation_audits/volkanovski_holloway_i/round4_completo_reporte.md y .json. PDF clasificado: output/pdf/volkanovski_holloway_i_round4_clasificado.pdf. No iniciar R5 sin orden.


## 2026-11-06 - Volkanovski-Holloway I R5: hojas 5-144

Round 5: hojas 5-144 revisadas (140 hojas, frames 64-2303). Acumulado 144/562. Siguiente hoja 145.

- kick_leg: 7
- kick_body: 1
- punch_head: 32
- punch_body: 4
- kick_head: 1
- ambiguous: 2

Pose y tracking pendientes, training_ready=false.


## 2026-11-06 - Volkanovski-Holloway I R5: hojas 145-284

140 hojas completas; 2240 cuadros consecutivos (2304-4543). Acumulado: 284/562 hojas. Siguiente: 285. Restan 278.

| Clase | Tanda | Acumulado round |
|---|---:|---:|
| punch_head | 42 | 74 |
| punch_body | 3 | 7 |
| kick_leg | 3 | 10 |
| kick_body | 2 | 3 |
| kick_head | 0 | 1 |
| other_strike | 1 | 1 |
| ambiguous | 3 | 5 |

Tanda: 54 registros, 51 positivos y 3 ambiguos. Acumulado: 101 registros, 96 positivos y 5 ambiguos.

Pose y tracking pendientes; track_slot=null, training_ready=false. Simultaneas separadas. Cobertura continua y evidencia verificadas; cero duplicados exactos. Reportes: output/annotation_audits/volkanovski_holloway_i/round5_145_284_reporte.md y .json. Continuar en hoja 285, frame 4544.


## 2026-11-06 - Volkanovski-Holloway I R5: hojas 285-424

140 hojas completas; 2240 cuadros consecutivos (4544-6783). Acumulado: 424/562 hojas. Siguiente: 425. Restan 138.

| Clase | Tanda | Acumulado round |
|---|---:|---:|
| punch_head | 32 | 106 |
| punch_body | 3 | 10 |
| kick_leg | 2 | 12 |
| kick_body | 0 | 3 |
| kick_head | 0 | 1 |
| other_strike | 1 | 2 |
| ambiguous | 3 | 8 |

Tanda: 41 registros, 38 positivos y 3 ambiguos. Acumulado: 142 registros, 134 positivos y 8 ambiguos.

Pose y tracking pendientes; track_slot=null, training_ready=false. Simultaneas separadas. Cobertura continua y evidencia verificadas; cero duplicados exactos. Reportes: output/annotation_audits/volkanovski_holloway_i/round5_285_424_reporte.md y .json. Continuar en hoja 425, frame 6784.


## 2026-11-06 - Volkanovski-Holloway I R5: hojas 425-562

138 hojas completas; 2207 cuadros consecutivos (6784-8990). Acumulado: 562/562 hojas. Round completo en revision visual RGB.

| Clase | Tanda | Acumulado round |
|---|---:|---:|
| punch_head | 34 | 140 |
| punch_body | 4 | 14 |
| kick_leg | 2 | 14 |
| kick_body | 2 | 5 |
| kick_head | 0 | 1 |
| other_strike | 2 | 4 |
| ambiguous | 2 | 10 |

Tanda: 46 registros, 44 positivos y 2 ambiguos. Acumulado: 188 registros, 178 positivos y 10 ambiguos.

Pose y tracking pendientes; track_slot=null, training_ready=false. Simultaneas separadas. Cobertura continua y evidencia verificadas; cero duplicados exactos. Reportes: output/annotation_audits/volkanovski_holloway_i/round5_425_562_reporte.md y .json. Revision RGB del round terminada: 562/562 hojas. Sin PDF por instruccion del usuario.


## 2026-11-06 - Round 1: auditoria geometrica automatica

Generadas mascaras para 155 acciones en output/annotation_audits/volkanovski_holloway_i/round1_geometry_masks. 107 candidatas con cobertura articular del atacante >=0.65. Controles de confianza, caja, limites, segmentos y saltos temporales. Revision automatica, no aprobacion visual. training_mask=false y training_ready=false; falta revision manual y adaptador v4 con contexto auditado. Se excluyeron tramos dudosos del sidecar de las ultimas 26 acciones: no inferir identidad global por track_id entre planos. Script reproducible scripts/annotation/build_round1_geometry_masks.py. Originales intactos; sin PDF.

## 2026-11-06 - Volkanovski-Holloway I R3: refinamiento post autoanotación

- RGB de origen en revisión 750, con 570/570 hojas, 9.110 cuadros y 176 registros (166 clasificados y 10 ambiguos). El inventario detectó 49 pares de acciones clasificadas superpuestas entre ambos atletas.
- Extracción de tracking y poses completa en 534,48 s. Tracking en RTX 3060 (303,47 s), poses en CPU (230,24 s). Los JSONL y la caché están en `data/processed/poses_direct_actions/alexander_volkanovski_vs_max_holloway_i_ufc_245__t_5G7bGTPxs__round3/`.
- En `output/annotation_audits/volkanovski_holloway_i/round3_refinement/` están las 269 hojas de evidencia, los inventarios, las decisiones de identidad con filas de pose, el reporte y las máscaras. La auditoría visual de identidad abarca 463/3.332 cuadros únicos de contexto y 24/176 ventanas completas (17 `punch_head`, 4 `kick_leg`, 1 `punch_body` y 2 `ambiguous`). Una ventana completa puede contener cuadros excluidos.
- Se verificaron 856 referencias por peleador y cuadro. Se excluyeron 53 por persona incorrecta, 12 por falta de detección del atleta y 5 por identidad sin resolver. En el intercambio de los cuadros 1211–1219 se descartaron detecciones del árbitro y se registraron los cambios de ranura de Volkanovski.
- Las máscaras geométricas filtran únicamente poses con identidad verificada. `frame_mask` indica al menos una articulación retenida y `training_mask` permanece falso. La aprobación visual de articulaciones y la validación del adaptador v4 siguen pendientes. El JSON de anotaciones conserva su revisión de origen.
- Continuar la auditoría en la ventana **1847–1882**, acción `kick_body` de cuadros **1859–1870**. Guardar decisiones con `scripts/annotation/save_refinement_identity_spans.py --round 3` y regenerar el reporte y el NPZ con la misma ejecución.

### Punto de continuación actualizado del round 3

Auditoría de identidad ampliada a **44/176 ventanas y 876/3.332 cuadros**. Desde la ventana 1847–1882 se completaron 20 ventanas y se añadieron 413 cuadros únicos. Las decisiones nuevas contienen 724 referencias verificadas, 98 exclusiones por persona incorrecta (árbitro, público o rival) y 4 casos sin resolver. El intercambio simultáneo en 2020–2032 mantiene sus dos etiquetas RGB, pero la pose de Holloway se excluyó en ese intervalo por seguimiento del público.

Se regeneró el NPZ y se comprobó la coincidencia de cada referencia con las decisiones de identidad y la ausencia de puntos activos en cuadros excluidos o pendientes. `training_ready=false`. Aún faltan la revisión articular y la integración con la entrada temporal de v4, que usa 32 pasos y normalización con el cuerpo del atacante. Retomar en **2934–2968** (`punch_head`, 2946–2956).

## 2026-11-07 - Volkanovski-Holloway I R2 y R3: cierre de identidad y máscaras

La revisión de identidad por cuadro cubre el contexto completo de las 339 ventanas. Las poses del árbitro, del público o del peleador equivocado se descartaron. Las poses ausentes o mezcladas quedaron sin referencia cuando no se pudo asignar una fila fiable. Se conservan por separado los registros RGB simultáneos.

| Round | Ventanas | Cuadros únicos | Referencias verificadas | Persona incorrecta | Ausentes | No resueltas | Pares simultáneos |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 163/163 | 2.834/2.834 | 5.002 | 322 | 181 | 163 | 62 |
| 3 | 176/176 | 3.332/3.332 | 5.939 | 467 | 167 | 91 | 49 |

Se regeneraron `geometry_report.json` y `geometry_masks.npz` en los directorios `round2_refinement` y `round3_refinement`. Hay 102 y 132 positivos, respectivamente, con cobertura articular del atacante de al menos 65 %. El script `scripts/annotation/validate_refinement_v4.py 2 3` valida las filas referenciadas y ejercita el extractor temporal y el TCN sin entrenar. La velocidad se invalida si falta una transición auditada entre cuadros muestreados.

| Clase | R2 RGB | R2 prueba v4 | R3 RGB | R3 prueba v4 |
|---|---:|---:|---:|---:|
| punch_head | 113 | 105 | 129 | 123 |
| punch_body | 10 | 9 | 6 | 6 |
| kick_leg | 22 | 22 | 22 | 21 |
| kick_body | 3 | 3 | 6 | 6 |
| kick_head | 0 | 0 | 1 | 1 |
| other_strike | 1 | 1 | 2 | 2 |
| ambiguous | 14 | 0 | 10 | 0 |
| Total | 163 | 140 | 176 | 159 |

La prueba técnica excluyó 9 positivos en R2 y 7 en R3 por ausencia de pose del atacante utilizable durante el golpe. Los ambiguos quedan fuera de la supervisión.

`scripts/annotation/render_refinement_masks.py 2 3` generó 515 hojas para R2 y 551 para R3 bajo `masked_overlays/<SHA-256 del NPZ>/`. Cada manifiesto vincula hojas por `action_id` y rango de cuadros. Verde indica atacante y azul rival. La revisión puntual de las hojas enmascaradas produjo exclusiones manuales para ambas muñecas de Volkanovski en R2, cuadro 8165, y el tobillo derecho de Holloway durante la patada alta en R3, cuadros 8712–8722. `joint_exclusions.json` conserva los motivos y la evidencia de cada exclusión. Se regeneraron las máscaras y la prueba v4 mantuvo los mismos totales.

**Punto de continuación.** La identidad está cerrada. Queda revisar exhaustivamente las articulaciones que muestran las hojas de máscaras, registrar sus correcciones y conectar los ejemplos aprobados al constructor de corpus de v4. Ambos rounds siguen con `training_ready=false`. Los 299 casos que pasan la prueba técnica aún no son ejemplos aprobados.

## 2026-11-07 - Volkanovski-Holloway I R4 y R5: refinamiento post autoanotación

Se verificaron los inventarios RGB de origen antes de extraer tracking y pose. El round 4 tiene 579 hojas, 9.261 cuadros, 177 registros y 53 pares de positivos simultáneos. El round 5 tiene 562 hojas, 8.991 cuadros, 188 registros y 57 pares. Los manifiestos separan las etiquetas ambiguas y guardan la procedencia por SHA-256.

La extracción se ejecuta con `configs/app/pose_pipeline_direct_actions.yaml`, GPU RTX 3060 para detección y tracking, MediaPipe CPU para pose, sin vista previa y con caché verificada. En este equipo `strikevision_ML` tiene PyTorch CPU. Se usó el Python de `strikevision_McByte` para cargar su PyTorch CUDA, con las dependencias de pose ya instaladas en `strikevision_ML` añadidas al final de la ruta de importación del proceso. Una prueba aislada de 16 cuadros confirmó las etapas completas antes de procesar ambos videos.

La auditoría utiliza las hojas `overlays/` y `save_refinement_identity_spans.py --round 4` o `--round 5`. El comando registra el próximo cuadro pendiente. Las máscaras y la prueba v4 distinguen auditoría de identidad, elegibilidad técnica y aprobación visual.

### Cierre de identidad y validación técnica

| Round | Ventanas auditadas | Cuadros únicos de contexto | Referencias verificadas | Persona incorrecta | Ausentes | No resueltas | Duplicadas |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 177/177 | 3.315/3.315 | 5.862 | 534 | 163 | 65 | 6 |
| 5 | 188/188 | 3.438/3.438 | 5.844 | 708 | 262 | 51 | 11 |

Cada cuadro tiene una decisión por atleta. Se descartaron las detecciones del árbitro y del público. El mapa conserva cambios de ranura y cada acción simultánea mantiene su propio atacante. Los vínculos exactos están en `identity_decisions.json` y en el eje atacante/rival de `geometry_masks.npz`.

| Clase | R4 RGB | R4 pasa v4 | R5 RGB | R5 pasa v4 |
|---|---:|---:|---:|---:|
| punch_head | 137 | 134 | 140 | 134 |
| punch_body | 7 | 7 | 14 | 14 |
| kick_leg | 17 | 17 | 14 | 14 |
| kick_body | 5 | 5 | 5 | 5 |
| kick_head | 1 | 1 | 1 | 1 |
| other_strike | 0 | 0 | 4 | 3 |
| ambiguous | 10 | 0 | 10 | 0 |
| **Total** | **177** | **164** | **188** | **171** |

`validate_refinement_v4.py 4 5` verificó referencias, máscaras, continuidad de velocidades y paso por el extractor de 32 pasos y el TCN sin entrenar. Tres positivos de R4 y siete de R5 quedaron fuera por falta de una pose utilizable del atacante durante el golpe. Hubo 125 y 135 positivos, respectivamente, con cobertura articular del atacante de al menos 65 % en los cuadros de acción.

Se inspeccionaron las ventanas enmascaradas de ambas patadas altas y se excluyó el tobillo derecho de Volkanovski en R4 (1591–1595) y el de Holloway en R5 (927–930). `joint_exclusions.json` registra los cuadros y las imágenes que motivaron la corrección. La prueba de v4 mantuvo sus totales tras regenerar las máscaras. Para localizar la revisión vigente, usar el SHA-256 de `v4_validation.json` bajo `masked_overlays/` en cada directorio `round4_refinement` y `round5_refinement`.

Durante esa revisión aparecieron dos posibles golpes de mano fuera de los registros RGB vigentes. Revisar el de Volkanovski en R4 (1577–1583) y el de Holloway al inicio de la patada alta en R5 (885–890) si se reabre la anotación RGB. Estas observaciones quedan fuera del conteo hasta resolverlas.

**Punto de continuación.** Identidad, vínculos, geometría y prueba técnica cerrados para R4 y R5. Quedan la revisión exhaustiva de las articulaciones retenidas y la incorporación de ejemplos aprobados al constructor de corpus. `training_ready=false` en ambos rounds.

## 2026-11-09 - Round 1: preparación de auditoría de contexto v4

Se verificó la cobertura RGB del round 1 filtrando los 562 rangos con estado `reviewed`. Los 30 rangos adicionales corresponden a decisiones `no_strike` o `ambiguous`, por lo que no deben contarse como hojas adicionales. Se generaron 247 hojas de tracking y pose en `round1_refinement/overlays/` usando la extracción con caché verificada de `poses_direct_actions`.

La nueva auditoría por cuadro y atleta cubre 178/3.128 cuadros de contexto y 6/155 ventanas. En 600–643 la segunda ranura sigue al árbitro y se descarta para Holloway; desde el corte en 644 se identificaron ambos peleadores. Las decisiones y máscaras parciales están en `round1_refinement/identity_decisions.json` y `geometry_masks.npz`. Continuar en el cuadro 997, ventana 997–1029. El contexto del round 1 aún no está auditado completamente, la aprobación visual de articulaciones de los cinco rounds y la integración al corpus v4 siguen pendientes. `training_ready=false`.

La auditoría avanzó a 36/155 ventanas y 788/3.128 cuadros únicos. Se excluyeron capturas del árbitro y del público, incluso cuando la ranura cambió durante el mismo intercambio. Las poses del otro atleta se conservaron en los cuadros donde pudieron verificarse. El siguiente cuadro pendiente es el 2654, en la ventana 2654–2691.

La auditoría avanzó a 76/155 ventanas y 1.564/3.128 cuadros únicos. Se guardaron las reasignaciones durante los cruces y se excluyeron las capturas del árbitro y del público. Las máscaras parciales se regeneraron con las decisiones nuevas. El siguiente cuadro pendiente es el 4769, en la ventana 4769–4803.

### Cierre de la auditoría de identidad del round 1

Se completaron las 79 ventanas restantes. La cobertura final es **155/155 ventanas y 3.128/3.128 cuadros únicos de contexto**. Se guardaron 5.426 referencias verificadas y se excluyeron 522 detecciones de persona incorrecta, 155 ausencias y 153 casos sin identidad resoluble. Las detecciones del árbitro y del público quedaron excluidas. En el cierre del round, los cuadros con poses mezcladas se dejaron sin referencia cuando no se pudo separar a los atletas con claridad. La revisión RGB 874 conserva sus 51 pares de acciones positivas simultáneas.

| Clase | Registros RGB | Pasan prueba técnica v4 |
|---|---:|---:|
| punch_head | 120 | 114 |
| punch_body | 3 | 3 |
| kick_leg | 14 | 14 |
| kick_body | 2 | 2 |
| kick_head | 1 | 1 |
| other_strike | 2 | 2 |
| ambiguous | 13 | 0 |
| **Total** | **155** | **136** |

Seis positivos `punch_head` quedaron fuera de la prueba por falta de una pose utilizable del atacante durante el golpe. Las máscaras geométricas dejan 112 positivos con cobertura articular del atacante de al menos 65 %. Se ejecutó `validate_refinement_v4.py 1` con el extractor de 32 pasos y un TCN sin entrenar. `render_refinement_masks.py 1` generó 503 hojas de evidencia en `round1_refinement/masked_overlays/<SHA-256 de geometry_masks.npz>/`. El SHA vigente y el resultado por acción están en `round1_refinement/v4_validation.json`.

**Punto de continuación.** Falta la revisión visual de las articulaciones retenidas y la integración de los ejemplos aprobados al corpus v4. El estado permanece en `training_ready=false`.


### Primera tanda de revisión visual de articulaciones del round 1

Revisadas cuadro por cuadro cinco ventanas (18 hojas y 156 cuadros únicos de contexto). Se priorizaron las dos patadas al cuerpo y la patada a la cabeza, el golpe simultáneo de Volkanovski y el golpe siguiente en el clinch. Las ventanas del manifiesto son 119 (6712–6756), 152 (8856–8902), 153 (8926–8968), 154 (8937–8968) y 155 (8957–8989).

`round1_refinement/joint_exclusions.json` registra las exclusiones por atleta, articulación y cuadro, con motivo y evidencia. Entre ellas están la rodilla y el tobillo de Holloway proyectados sobre Volkanovski en 8943–8945, y puntos del cuerpo de Volkanovski colapsados sobre su cabeza cuando el árbitro tapa el resto en 8878. Las exclusiones se aplican también cuando ese mismo cuadro aparece en otra ventana. El renderizador deja las leyendas fuera del cuadro para que los pies y los puntos de la parte inferior se puedan revisar.

La prueba técnica de v4 tras regenerar las máscaras se mantiene en 136 acciones, seis positivos excluidos y 13 ambiguos. **Revisión visual en curso.** Retomar por la ventana 1 (600–636), con las hojas del manifiesto identificado por el SHA-256 vigente de `v4_validation.json`. Quedan 150 ventanas por pasar por esta revisión y la integración al corpus. `training_ready=false`.


### Piloto del round 1 conectado al corpus de v4

La configuración de v4 incorpora el JSON de anotación directa del round 1 con una fuente explícita para el manifiesto, los vínculos de pose y las máscaras. El constructor comprueba procedencia y cobertura, y el cargador de entrenamiento reconstruye la secuencia con las articulaciones retenidas y la continuidad auditada. La pelea queda solo en entrenamiento en los tres folds. Se generó un corpus piloto separado bajo `round1_refinement/v4_corpus_pilot/`.

El cargador real recorrió **136 acciones en 5 lotes**, con 32 pasos y 206 características por ejemplo. Conteo por clase principal del evento: 114 `punch_head`, 3 `punch_body`, 14 `kick_leg`, 2 `kick_body`, 1 `kick_head` y 2 `other_strike`. Quedaron fuera 6 `punch_head` sin pose utilizable durante el golpe y 13 ambiguos. De los 51 pares simultáneos anotados, 48 conservan ambos ejemplos en el piloto. Los objetivos son positivos solamente. Los intervalos explícitos sin golpe requieren una auditoría de poses propia para ingresar como negativos.

El informe `round1_loader_validation.json` distingue conteos por evento y por objetivo multietiqueta. Esta validación habilita el subconjunto piloto con el criterio de identidad auditada, geometría y pose del atacante durante la acción. La revisión visual exhaustiva de articulaciones sigue registrada por separado. Los rounds 2 a 5 todavía no se incorporaron al corpus. No se ejecutó entrenamiento.


### Round 2 incorporado al corpus piloto de v4

Se incorporó el round 2 de Volkanovski–Holloway I a la configuración de anotación directa. El corpus piloto bajo `round2_refinement/v4_corpus_pilot/` conserva los 136 ejemplos del round 1 e incorpora 140 del round 2. Los 276 ejemplos de la pelea quedaron exclusivamente en entrenamiento en los tres folds.

El cargador de entrenamiento leyó los 140 ejemplos nuevos en cinco lotes, con 32 pasos y 206 características por ejemplo. Conteo por acción: 105 `punch_head`, 9 `punch_body`, 22 `kick_leg`, 3 `kick_body`, 0 `kick_head` y 1 `other_strike`. Quedaron fuera 14 ambiguos y nueve acciones sin pose utilizable del atacante (ocho `punch_head` y una `punch_body`). Se conservan ambos ejemplos en 54 de los 62 pares simultáneos del round. El informe reproducido con el cargador está en `round2_loader_validation.json`. El piloto incorpora objetivos positivos del round 2 y no se ejecutó entrenamiento.


### Round 3 incorporado al corpus piloto de v4

El corpus piloto de `round3_refinement/v4_corpus_pilot/` añade 159 ejemplos del round 3 y conserva los 276 de los rounds 1 y 2. Los 435 ejemplos de la pelea se seleccionan exclusivamente para entrenamiento en los tres folds.

El cargador de entrenamiento leyó los 159 ejemplos nuevos en cinco lotes de secuencias de 32 pasos y 206 características. Conteo por acción: 123 `punch_head`, 6 `punch_body`, 21 `kick_leg`, 6 `kick_body`, 1 `kick_head` y 2 `other_strike`. Quedaron fuera 10 ambiguos y siete acciones sin pose utilizable del atacante (seis `punch_head` y una `kick_leg`). Se conservaron ambos ejemplos en 44 de los 49 pares simultáneos. `round3_loader_validation.json` contiene los identificadores de los pares conservados y los hashes de procedencia. El piloto añade solo positivos del round 3. No se ejecutó entrenamiento.


### Round 4 incorporado al corpus piloto de v4

El corpus piloto de `round4_refinement/v4_corpus_pilot/` añade 164 ejemplos positivos del round 4 y conserva los 435 de los rounds 1 a 3. Los 599 ejemplos de la pelea se seleccionan exclusivamente para entrenamiento en los tres folds.

El cargador de entrenamiento leyó los 164 ejemplos nuevos en seis lotes de secuencias de 32 pasos y 206 características. Conteo por acción: 134 `punch_head`, 7 `punch_body`, 17 `kick_leg`, 5 `kick_body`, 1 `kick_head` y 0 `other_strike`. Quedaron fuera 10 ambiguos y tres `punch_head` sin pose utilizable del atacante. Se conservaron ambos ejemplos en los 53 pares simultáneos. `round4_loader_validation.json` contiene los identificadores de los pares y los hashes de procedencia. No se ejecutó entrenamiento.


### Volkanovski–Holloway I completo en el corpus piloto de v4

El corpus consolidado de `round5_refinement/v4_corpus_pilot/` conserva los 599 positivos de los rounds 1 a 4 y añade 171 del round 5. El cargador real recorrió los 770 ejemplos en 25 lotes de secuencias de 32 pasos y 206 características. Se verificó que los cinco rounds se seleccionan exclusivamente para entrenamiento en los tres folds.

| Clase | Round 5 | Pelea completa |
|---|---:|---:|
| punch_head | 134 | 610 |
| punch_body | 14 | 39 |
| kick_leg | 14 | 88 |
| kick_body | 5 | 21 |
| kick_head | 1 | 4 |
| other_strike | 3 | 8 |
| **Total** | **171** | **770** |

Se conservaron ambos ejemplos en 51 de los 57 pares simultáneos del round 5 y en 250 de 272 en toda la pelea. Se excluyeron diez ambiguos y siete acciones sin pose utilizable del atacante en el round 5. En total quedaron fuera 57 ambiguos y 32 acciones sin pose utilizable. El piloto incorpora únicamente positivos de esta pelea. `fight_loader_validation.json` contiene conteos por round, identificadores de pares y hashes de procedencia. No se ejecutó entrenamiento.


### Corpus definitivo de v4 publicado con Volkanovski–Holloway I

El comando `scripts/ml/run_action_recognition.py prepare --config configs/models/action_training_v4.yaml` publicó los cinco rounds en `data/processed/datasets/action_recognition_v4`. El corpus pasó de 43.417 a 44.187 ejemplos. Se comprobó la igualdad de los 43.417 ejemplos previos y que los archivos de muestras y auditoría publicados coinciden con los del piloto. El respaldo previo está en `round5_refinement/v4_previous_corpus_895b8225305b/`.

Desde la ruta oficial, el selector de entrenamiento incluyó las 770 acciones de la pelea en los tres folds y las excluyó de validación y prueba. El cargador volvió a leer los 770 positivos en 25 lotes y se verificaron los 250 pares simultáneos conservados. `round5_refinement/v4_official_loader_validation.json` registra los hashes y el conteo. No se ejecutó entrenamiento.


### Volkanovski–Holloway I aporta solo punch_head al corpus de v4

Los cinco rounds configuran `supervised_classes: [punch_head]`. El corpus oficial contiene 44.027 ejemplos, incluidos 610 positivos de esta pelea con solo el objetivo `punch_head` activo. Los otros 160 ejemplos previamente utilizables se conservan en la auditoría con `excluded_by_class_scope` y el piloto completo preserva sus muestras. La publicación se verificó contra el corpus de prueba. Las 43.417 muestras de las demás peleas son idénticas a las de la versión anterior.

El cargador oficial leyó los 610 positivos en 20 lotes. El fold 0 selecciona 6.295 muestras de entrenamiento y las mismas 1.149 muestras de validación de la referencia `tcn-women-v4-fold0-cachefix`. Nunes–Shevchenko II permanece fuera del corpus. El reporte está en `round5_refinement/v4_punch_head_only/official_loader_validation.json`. No se ejecutó entrenamiento.

