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

