# Plan de desarrollo y operación: anotación directa de acciones

Creado: 5 de noviembre de 2026. Actualizado: 6 de noviembre de 2026. Estado: metodología operativa aplicada a Volkanovski-Holloway I.

Este es el plan de autoanotación directa vigente. La clasificación se realiza mediante revisión visual de cuadros consecutivos por el asistente; no significa etiquetado automático por el modelo v4.

## 1. Objetivo y control del usuario

Anotar las peleas de `data/splits/fights_remaining_mixed` directamente con
acciones por atacante, técnica y tiempo, usando imágenes consecutivas y escritura
validada en backend. Se elimina la tarea humana previa de etiquetar ventanas
binarias. La ruta operativa actual revisa todos los cuadros del round; no depende de un filtro strike/no-strike ni de predicciones de v4.

El usuario autoriza la pelea y dirige el avance por rounds y tandas. Retomar el estado guardado antes de empezar; nunca reiniciar una tanda ya completada por falta de memoria conversacional.

Reglas operativas obligatorias:

1. Una sola pelea activa. Procesar sus rounds en orden numérico.
2. Trabajar un round a la vez y completar la tanda solicitada. Al cerrar el round, entregar resultados y PDF salvo que el usuario indique omitirlo. Respetar el control del usuario sobre el siguiente round o tanda; una instrucción posterior tiene prioridad.
3. Al terminar la pelea, entregar el consolidado y detenerse. La siguiente pelea
   requiere una orden expresa del usuario. Tampoco preparar sus videos por lote
   mientras se espera esa orden.
4. Conservar originales y anotaciones existentes. Cada ejecución tendrá fuentes,
   configuración, hashes y salida separados; retomar desde el último estado válido.
5. No entrenar ni promover modelos como parte de este trabajo de anotación.

Pelea piloto ejecutada: **Alexander Volkanovski vs Max Holloway I, UFC 245**, rounds 1-5. Sus anotaciones RGB se completaron por hojas y tandas. El refinamiento de poses de R1 se inició después: no confundir anotación visual terminada con dataset listo para entrenar.

## 2. Inventario verificado

Se inspeccionaron cabeceras de los MP4 locales mediante OpenCV. Son **134 rounds
de 32 peleas**, **1.134.636 cuadros**, aproximadamente **10,71 horas** de video.
Los rounds contienen entre **975 y 12.887 cuadros**. Ningún archivo tuvo FPS o
conteo de cuadros nulos en esta comprobación; falta comprobar decodificación
completa e integridad al preparar cada round.

| Orden propuesto | Pelea / evento identificado por archivo | Rounds disponibles |
|---:|---|---|
| 1 | Volkanovski–Holloway I, UFC 245 | 1–5 |
| 2 | Volkanovski–Holloway II, UFC 251 | 1–5 |
| 3 | Condit–Diaz, UFC 143 | 1–5 |
| 4 | Cerrone–Iaquinta, UFC Ottawa | 1–5 |
| 5 | Poirier–Hooker, UFC Vegas 4 | 1–5 |
| 6 | Poirier–Holloway II, UFC 236 | 1–5 |
| 7 | Barboza–Hooker, UFC Milwaukee | 1–3 |
| 8 | Barboza–Felder I, UFC on Fox 16 | 1–3 |
| 9 | Barboza–Felder II, UFC 242 | 1–3 |
| 10 | Barboza–Yusuff, UFC Vegas 81 | 1–5 |
| 11 | Adesanya–Gastelum, UFC 236 | 1–5 |
| 12 | Joanna–Kowalkiewicz, UFC 205 | 1–5 |
| 13 | Joanna–Waterson, UFC Tampa | 1–5 |
| 14 | Joanna–Letourneau, UFC 193 | 1–5 |
| 15 | Calderwood–Chookagian, UFC 238 | 1–3 |
| 16 | Aldo–Stephens, UFC Calgary | 1 |
| 17 | Aldo–Munhoz, UFC 265 | 1–3 |
| 18 | Rosa–Aldana, UFC 296 | 1–3 |
| 19 | Holloway–Pettis, UFC 206 | 1–3 |
| 20 | Holloway–Allen, UFC Kansas City | 1–5 |
| 21 | Holloway–Ortega, UFC 231 | 1–4 |
| 22 | Holloway–Kattar, UFC on ABC 1 | 1–5 |
| 23 | Holloway–Poirier III, UFC 318 | 1–5 |
| 24 | Holloway–Rodriguez, UFC Vegas 42 | 1–5 |
| 25 | Costa–Vettori, UFC Vegas 41 | 1–5 |
| 26 | Fiziev–Diakiese, UFC Fight Island 2 | 1–3 |
| 27 | Whittaker–Romero II, UFC 225 | 1–5 |
| 28 | Strickland–Costa, UFC 302 | 1–5 |
| 29 | Thompson–Holland, UFC Orlando | 1–4 |
| 30 | Dillashaw–Barao II, UFC on Fox 16 | 1–4 |
| 31 | Shevchenko–Joanna, UFC 231 | 1–5 |
| 32 | Rodriguez–Emmett, UFC 284 | 1–2 |

El orden es una cola propuesta por nombre de archivo y el usuario puede cambiarlo.
Se conservará el identificador completo, incluido el ID de origen, para distinguir
revancha, evento y round. No se asumirá que toda pelea tiene cinco rounds.

## 3. Flujo operativo: round completo por hojas y lotes

`MP4 original → inventario y decodificación → todas las hojas consecutivas → total informado al usuario → lotes de 140-160 hojas → acciones guardadas y progreso → cierre RGB → refinamiento posterior`.

### 3.1 Preparar e informar el total

1. Verificar identificación de pelea/round, video, resolución, FPS real, hash y número de cuadros efectivamente decodificados. No asumir cinco rounds ni duración exacta.
2. Extraer todos los cuadros en hojas 4 x 4 de **16 cuadros consecutivos**, identificados por frame absoluto y timestamp. Generar desde el original, sin saltos ni dependencia de Gradio.
3. Para N cuadros, el total de hojas es `ceil(N/16)`. Hoja h (numeración de usuario desde 1): inicio `16*(h-1)`, final `min(16*h-1, N-1)`, ambos inclusivos. La última hoja puede tener menos de 16 cuadros.
4. Comprobar que no falten hojas o cuadros y guardar manifiesto con fuentes, FPS, resolución, hashes y correspondencias de hoja/rango.
5. Informar el total antes de anotar y seguir la tanda ordenada por el usuario. Ejemplo real R1: 8.991 cuadros → 562 hojas; última hoja 8976-8990, con 15 cuadros.

La detección, tracking y pose se reutilizan cuando sus fuentes están verificadas. Su ausencia o mala calidad no impide etiquetar una acción visible en RGB. La vinculación con poses se valida en la fase posterior.

### 3.2 Dividir y ejecutar tandas

- Tamaño habitual: **140 a 160 hojas por tanda**, aproximadamente 2.240 a 2.560 cuadros. No confundir cantidad de hojas con cantidad de acciones.
- La instrucción del usuario puede establecer otra cantidad. La tanda final cubre exactamente las restantes aunque sean menos de 140. No rellenar el round para alcanzar un tamaño de lote.
- Calcular cada rango desde la primera hoja pendiente. Ejemplo de 562 hojas: 1-150, 151-290, 291-430 y 431-562 (132 finales). Es un ejemplo de partición, no una reconstrucción del historial exacto de R1.
- Inspeccionar todos los cuadros de cada hoja, ambos atletas y la secuencia completa; no anotar solo el principio ni una muestra representativa.
- Ampliar a hojas vecinas cuando preparación, ejecución o recuperación cruce el borde. Una acción puede tener evidencia en dos hojas; guardarla una sola vez.
- Completar toda la tanda solicitada antes de la entrega. Los guardados parciales sirven para recuperación y no se presentan como tanda terminada.

### 3.3 Reglas de anotación

Clases: `punch_head`, `punch_body`, `kick_leg`, `kick_body`, `kick_head`, `other_strike` y `ambiguous`. Codos y rodillas van en `other_strike`, con subtipo en la nota cuando sea visible.

Registrar cada intento claro aunque falle o encuentre guardia. Buscar con especial cuidado patadas y puños al cuerpo, conservando también todos los puños a cabeza visibles. No forzar cantidades para equilibrar clases. No contar fintas, manos de referencia, parries, agarres o derribos como golpes.

Guardar las acciones simultáneas como eventos independientes por atacante, con rangos que pueden solaparse. Identificar a cada atleta físicamente; la ranura del tracker no es una identidad permanente. Si el atacante o destino no se distingue, conservar la incertidumbre; no inventar etiquetas ni utilizar v4 como fuente de verdad.

Si la pose sigue al árbitro o al público, descartar esa referencia y continuar con la anotación visual. Un error de pose no borra una acción visible ni se convierte en negativo. Hojas RGB sin overlay no prueban que el tracking esté correcto; dejar `track_slot=null` cuando no esté vinculado y verificado.

### 3.4 Guardado, reanudación y entrega de tanda

Usar `DirectActionStore` para el JSON `direct_strike_actions_v1`, con revisión, `action_id`, frames originales inclusivos, peleador, clase, notas y evidencia realmente inspeccionada. Guardar decisiones de revisión por separado de las acciones. Una decisión `reviewed` no significa `no_strike`.

Persistir periódicamente las acciones y hojas revisadas. Antes de retomar, leer el JSON canónico, manifiesto y progreso; comprobar la última hoja continua revisada, no confiar solo en lo dicho en el chat. Evitar duplicados al reejecutar y no marcar como revisadas hojas aún no vistas.

Cada tanda debe dejar un progreso con: pelea, round, total de hojas, rango solicitado, rango realmente revisado, siguiente hoja/frame, restantes de tanda/round, revisión y hashes, conteos nuevos/acumulados, incidencias y estado de poses. Si hay un hueco de revisión, declararlo sin saltarlo al calcular el avance.

Entregar avance exacto y conteos persistidos. No generar PDF por cada tanda salvo solicitud. Al completar el round, seguir las reglas de entrega de la sección 8 y la preferencia más reciente del usuario sobre PDF.

## 4. Validación de cobertura y cierre RGB

La ruta vigente inspecciona **todas** las hojas del round. No se omiten zonas mediante filtro binario o muestreo del tiempo rechazado. Las propuestas automáticas pueden orientar la atención, pero no reemplazan esta cobertura ni asignan etiquetas.

Antes de declarar N/N:

1. Verificar cobertura continua de cuadros 0 a N-1 y existencia de la evidencia de cada acción.
2. Comprobar límites, FPS, duplicados exactos y posibles golpes repetidos entre hojas. Conservar simultáneas intencionales.
3. Recalcular conteos por clase y peleador desde el JSON persistido; separar positivos y ambiguos. No deducir acciones a partir del número de hojas.
4. Mantener regiones sin etiqueta y negativos explícitos diferenciados. No convertir huecos entre golpes en `no_strike`, ni al defensor en negativo automático.
5. Guardar reporte final, progreso de round completo y actualización del contexto. Identificar qué revisión corresponde al reporte.

El cierre RGB solo confirma la revisión visual y sus etiquetas. Después aplicar [Plan de refinamiento post Autoanotacion](<Plan de refinamiento post Autoanotacion.md>): auditoría de tracking/identidad, vínculos de pose por cuadro, geometría, máscaras, revisión visual de articulaciones y adaptación temporal a v4. Mantener entrenamiento pendiente hasta superar esos controles.

## 5. Contratos originales y estado de implementación

Antecedente del diseño inicial: `annotations/actions.py::ActionStore` construye eventos solo
desde ventanas con `label == strike`. El constructor del corpus también valida
contra esa fuente binaria. Por tanto, la anotación directa requiere un contrato
explícito nuevo; marcar propuestas como `strike` para engañar esos validadores
contaminaría la procedencia de las etiquetas.

Estado vigente: la campaña usa `annotations/direct_actions.py::DirectActionStore` y el esquema `direct_strike_actions_v1`. La escritura de acciones y revisión por hojas está implementada. El adaptador de dataset directo a v4 y la aprobación de sus máscaras siguen pendientes; no confundir esta lista original de desarrollo con una certificación de implementación.

Diseño original de referencia:

1. Manifiesto `action_candidates_v1`: video, pelea, round, FPS, cuadros originales,
   hashes, modelos/configuración, propuesta e intervalo, señales que la originaron,
   calidad y contexto. Sus registros son sugerencias sin etiqueta humana.
2. Extender el almacenamiento con fuente discriminada y validada para candidatos,
   preservando compatibilidad del flujo binario histórico. El nuevo payload de
   acciones debe declarar tipo de fuente y hash; no reutilizar silenciosamente
   `strike_actions_v1` con una semántica distinta.
3. Guardar eventos, atacante físico y ranura técnica, rangos inclusivos, técnica,
   calidad, notas, evidencia visual, revisión y progreso. Negativos confirmados
   son decisiones de revisión, no «correcciones de error binario» en este flujo.
4. Versionar propuestas congeladas por revisión. Acciones descubiertas en huecos
   crean nuevos eventos con historial; una actualización no debe invalidar o
   reasignar silenciosamente los eventos ya anotados.
5. Extender el constructor de dataset para consumir ambos tipos de fuente,
   mantener máscaras de supervisión, no etiquetado y negativos explícitos,
   y respetar el mapa de identidad por tramo.
6. Validar round vacío: cero ataques puede ser un resultado válido tras revisión;
   no debe fallar por no tener eventos positivos.
7. Pruebas de solapamientos, identidad, límites, duplicados, revisiones obsoletas,
   reinicio, cambio de fuente, colas finales, regiones no inspeccionadas y mezcla
   de corpus histórico/directo. Particiones por pelea, sin mezclar revanchas como
   si fueran el mismo video ni filtrar cuadros de una pelea entre train y test.

Rutas propuestas, aisladas del histórico:

```text
data/processed/poses_direct_actions/<video_id>/
data/annotations/action_candidates_v1/<video_id>.json
data/annotations/action_strikes_direct_v1/<video_id>.json
output/annotation_audits/<fight_id>/<round_id>/
output/reports/direct_actions/<fight_id>/<round_id>.md
output/pdf/direct_actions/<fight_id>/<round_id>_clasificado.pdf
```

El progreso tendrá `pelea_autorizada`, `round_actual`, estado de preparación,
eventos/hojas revisados, auditoría pendiente, acciones guardadas, reporte y PDF.
Un conteo N/N será el de eventos del manifiesto, nunca una afirmación de revisión
de todo el metraje si hubo zonas no inspeccionadas.

## 6. Mejoras de tiempo de preparación (registro técnico inicial)

Cambios implementados junto con este documento:

- El detector deja de forzar CPU: `tracking_device=auto` selecciona GPU CUDA si
  PyTorch puede usarla. CPU permanece disponible. Pedir explícitamente una GPU
  ausente produce error; `auto` advierte cuando recurre a CPU.
- El dispositivo se transmite por extracción, merge y pipeline; queda registrado
  junto con los tiempos por etapa en `run_metadata.json`.
- El comando de preparación acepta `--device`, `--no-preview`, `--merge-tracks` y
  `--reuse`. El perfil `configs/app/pose_pipeline_direct_actions.yaml` une tracks
  y omite render completo por defecto.
- `--reuse` verifica video, opciones, código, pesos, versiones de dependencias y
  hashes de artefactos. Un resultado completo válido evita repetir detección y
  pose. Un caché diferente, incompleto o alterado exige una carpeta nueva.
- El preparador UI también acepta la selección de dispositivo, aunque la
  operación de anotación de esta campaña se hará fuera de Gradio.

Comando listo para la futura preparación de un round autorizado:

```powershell
& 'C:\Users\Cristian\miniconda3\envs\strikevision_ML\python.exe' scripts/ml/run_pose_pipeline.py `
  --config configs/app/pose_pipeline_direct_actions.yaml `
  --video 'data/splits/fights_remaining_mixed/alexander_volkanovski_vs_max_holloway_i_ufc_245__t_5G7bGTPxs__round1.mp4' `
  --reuse
```

Usar una carpeta nueva para una prueba parcial con `--max-frames`; nunca tratarla
como preparación completa. Los artefactos antiguos sin manifiesto se inspeccionan
y se migran mediante una herramienta específica pendiente, no se dan por válidos
solo por existir. Reanudación por etapa tras fallo parcial también queda pendiente;
la reutilización implementada cubre preparaciones completas verificadas.

**Situación observada al crear el plan; volver a verificar antes de ejecutar:** existe una RTX 3060 de 12 GB, pero
`strikevision_ML` tiene `torch 2.5.1+cpu` y `torch.cuda.is_available() == False`.
Por ello la aceleración GPU todavía no está activa. Preparar un entorno separado
con dependencias CUDA compatibles, verificar detector y pose, y comparar contra
CPU antes de migrar el flujo. Esta entrega no instala paquetes ni sustituye el
entorno actual. La pose MediaPipe no migra automáticamente a GPU por cambiar YOLO.

Medir un tramo representativo con idénticos frames, pesos y resolución antes de
comparar tiempos. Separar inicialización de inferencia estable; comparar también
cobertura de dos peleadores, fragmentación, cambios de identidad y calidad visual.
No prometer un factor de aceleración ni un tiempo final sin benchmark de round.
Evitar procesos GPU simultáneos hasta medir memoria y estabilidad. Mantener el
orden de cuadros del tracker; cualquier batch futuro debe respetarlo.

Prueba técnica realizada en esta entrega: 180 cuadros iniciales de
Volkanovski–Holloway I R1, CPU, tracking unido y sin preview. El pipeline registró
16,97 s de tracking, 5,60 s de pose y 22,65 s totales (sin incluir arranque del
proceso ni cálculo de hashes de caché). Una segunda ejecución reutilizó los
artefactos verificados sin ejecutar el pipeline; el comando completo tardó
aproximadamente 4,8 s. No es una comparación CPU/GPU ni una predicción del tiempo
de un round completo. La prueba no produjo anotaciones de acciones.

Validación de esta entrega: **20 pruebas pasadas**, incluyendo selección de
dispositivo, pipeline de pose, unión de tracks, caché con rechazo de entradas
cambiadas y preparador UI. La advertencia sobre CUDA no disponible es esperada
en el entorno CPU actual. No se ha validado todavía ejecución GPU.

## 7. Hitos originales y seguimiento

| Hito | Trabajo | Criterio para avanzar |
|---|---|---|
| A — esta entrega | Plan, inventario y mejoras de preparación | Usuario revisa plan; pruebas técnicas pasan |
| B | Contratos directos, guardado, candidatos, extracción y dataset | Pruebas anteriores y nuevas pasan sin alterar fuentes históricas |
| C | Piloto en round 1 de la primera pelea autorizada | Auditoría completa, cobertura medida, reporte y PDF revisados |
| D | Rounds restantes de esa pelea, en orden | Entrega y control de calidad por round |
| E | Consolidado de pelea | Usuario recibe resultados y decide la siguiente pelea |

El piloto se ejecuta sobre una pelea, no sobre cinco peleas diferentes. Los
rounds posteriores de esa misma pelea permiten medir variación de costo y ajustar
el filtro. No ampliar a las 32 peleas por una sola autorización inicial.

## 8. Entrega obligatoria por round y por pelea

Por cada round (PDF salvo indicación expresa del usuario de omitirlo):

1. Total de hojas, hojas revisadas, cuadros originales y duración, lotes completados y cualquier zona pendiente. En la ruta actual la cobertura exigida es todo el round; los candidatos no definen el denominador.
2. Tabla por clase, tabla por peleador y clase, total de acciones, simultaneidades,
   negativos confirmados y ambigüedades. Separar acciones de fragmentos.
3. Tabla de cada fragmento: ID, frames, segundos, acciones, atacante, límites,
   estado, evidencia inspeccionada y observaciones de tracking/identidad.
4. JSON persistido de acciones y manifiestos con procedencia, hashes y revisiones;
   tiempos reales de preparación/extracción/revisión/render.
5. PDF clasificado con una imagen representativa de **cada acción**, clase,
   atacante, frames y fragmento. Secciones propias para `no_strike` confirmado
   y casos ambiguos; tabla o mapa de cobertura que identifique lo no inspeccionado.
   Incluir muestras antes/durante/después en casos de identidad o técnica difíciles.
6. Renderizar e inspeccionar el PDF; comprobar correspondencia
   con el JSON, número de imágenes, legibilidad y ausencia de recortes de texto.

Por pelea: consolidado de clases y peleadores, enlaces a reportes/JSON/PDF de cada
round, lista de incidencias y estado del corpus. Después esperar la orden del
usuario para la siguiente pelea.

Validación de cierre: cuadros dentro del video, FPS original, sin duplicados
exactos, sin un mismo golpe repetido entre candidatos, solapamientos intencionales
conservados, ningún `no_strike` confirmado en conflicto con acción/ambigüedad,
identidad reconciliada o marcada incierta, fuente verificable y todas las hojas
realmente inspeccionadas. Si falta revisión o evidencia, informar el pendiente;
un contador completo o un PDF bonito no sustituyen la inspección visual.

## 9. Referencias del proyecto

- `docs/CONTEXTO-STRIKEVISION-ML-2026-10-26.md`: revisión por cuadros, simultáneos,
  tracking, escritura validada y regla de reporte más PDF por round.
- `docs/STRIKE-ANNOTATION-CONTRACT.md`: definición de intento, negativo y oclusión.
- `docs/PLAN-ACTION-RECOGNITION.md`: máscaras, identidad, ambiguos y particiones.

Este plan sustituye para esta campaña la obligación operativa de confirmar el
contador en Gradio. La fuente de verdad será el estado persistido, validado y
contrastado con las hojas realmente revisadas. El resto de criterios de calidad
visual del contexto se conserva.
