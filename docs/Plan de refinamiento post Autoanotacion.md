# Plan de refinamiento post Autoanotacion

Actualizado: 6 de noviembre de 2026. Alcance: un round visualmente anotado de la pelea autorizada. Este documento define el trabajo posterior; no declara que todas sus fases estén implementadas o aprobadas.

## 1. Entrada y resultado esperado

Entrada: video original, manifiesto de hojas, JSON canónico de acciones, evidencia inspeccionada y reporte final de revisión RGB. La revisión visual completa y la preparación para entrenamiento son estados separados.

Flujo: cierre RGB → inventario de fuentes → auditoría de identidad/tracking → vinculación por cuadro → auditoría geométrica → máscaras → validación visual y correcciones → adaptador temporal v4 → validación de corpus → entrega.

Trabajar round por round, dentro de una sola pelea autorizada. Una orden para refinar un round no autoriza entrenar, promover un modelo ni iniciar otra pelea. Conservar originales y etiquetas; escribir resultados derivados versionados y reproducibles.

## 2. Cerrar y congelar la anotación visual

1. Comprobar cobertura continua desde el cuadro 0 hasta el último cuadro decodificado, incluidas las hojas finales incompletas.
2. Releer el JSON y calcular acciones por clase y peleador, ambiguos, simultáneas y duplicados. Revisar posibles duplicados de un golpe a través de bordes; no borrar solapamientos de atacantes distintos.
3. Registrar revisión y SHA-256 de anotaciones, video y manifiesto. Identificar las acciones por `action_id`, no por su posición en una lista.
4. Mantener `ambiguous` fuera de positivos. Una región revisada sin acciones no es un negativo confirmado salvo decisión explícita; los huecos entre ataques tampoco lo son.

Salida: instantánea de fuentes y reporte RGB. Estado: `visual_annotation_complete=true`; entrenamiento pendiente.

## 3. Inventario y verificación de poses

- Localizar `pose.jsonl`, `tracking.jsonl`, metadatos de extracción, resolución y FPS. Verificar hashes y compatibilidad con el video y el dominio temporal original.
- Reutilizar cachés verificadas; no repetir detección y pose solo para generar imágenes. Si falta procedencia, auditarla antes de reutilizar.
- Inventariar las acciones con auditorías anteriores, referencias parciales y ausencia de referencias. Auditoría existente no equivale a pose aprobada.
- Validar límites, filas duplicadas para `(frame_index, fighter_id)` y correspondencia entre referencia, cuadro, ranura e ID de track.

Salida: inventario por acción. Si cambian fuentes o anotaciones, invalidar los derivados afectados y regenerarlos; no mezclar revisiones silenciosamente.

## 4. Auditoría visual de tracking e identidad

Generar hojas de cuadros consecutivos con cajas, esqueleto, ranura y `track_id`, usando la geometría y resolución originales. Inspeccionar todos los cuadros de cada acción y sus bordes; ampliar ante corte, oclusión o cambio de identidad. Auditar también el contexto que posteriormente usará v4.

La identidad física del atleta gobierna el mapa. La ranura 1/2 y el `track_id` pueden cambiar o seguir a otra persona. No aplicar una lista global de IDs para toda la pelea o entre planos.

Registrar por cuadro o tramo explícito: atleta, ranura, track, fuente, evidencia y decisión. Si el tracker sigue al árbitro o público, descartar esa referencia y continuar con el atleta que sí pueda verificarse. Si ambos tracks siguen al mismo atleta, no fabricar una pareja de poses. Conservar la acción visual aunque su pose no sirva.

Estados mínimos: identidad verificada, ausente, persona incorrecta, intercambio de ranuras, duplicación de persona y no resoluble. Los tramos dudosos quedan sin supervisión por pose. Corregir un mapa de identidad requiere evidencia, no continuidad asumida por ID.

## 5. Vinculación exacta por acción y cuadro

Guardar en un archivo de vínculos: `action_id`, atacante físico, rango inclusivo, FPS, hashes y referencias de ambos atletas por cuadro. Cada referencia incluye ruta, fila de JSONL empezando en 1, `frame_index`, `fighter_id`, `track_id` y estado del extractor.

Reabrir cada referencia y comprobar sus campos. Validar que la fila 1-based cuente las líneas físicas del archivo, incluidas líneas vacías si las hubiera. La ranura del atacante solo es estable si se verificó durante todo el evento; los cambios deben conservarse cuadro por cuadro.

Separar identidad completa del atacante, identidad completa de la pareja, disponibilidad de pose y calidad de articulaciones. `pose_valid=true` es salida del extractor, no aprobación visual. Evitar mantener dos archivos de vínculos contradictorios como fuentes igualmente válidas: publicar versión vigente y procedencia de las correcciones.

## 6. Auditoría geométrica automática

Aplicar los controles a las articulaciones cuya identidad esté verificada. Registrar motivos de rechazo y denominadores; no sumar incidencias de articulaciones como si fueran acciones distintas.

| Control | Configuración inicial aplicada en R1 | Interpretación |
|---|---|---|
| Confianza | >= 0,50 y <= 1 | Filtra puntos débiles del extractor |
| Datos | Coordenadas/confianza finitas; 17 articulaciones declaradas | Rechaza valores ausentes o inválidos |
| Imagen | Dentro de ancho/alto reales | Usar resolución del manifiesto, no una constante de otro video |
| Caja | Caja válida; margen de 15 % de su altura | Control aproximado de pertenencia corporal |
| Segmentos | Longitud <= 65 % de altura de caja | Señala geometría extrema; no garantiza anatomía correcta |
| Tiempo | Salto <= 30 % de altura entre cuadros consecutivos del mismo track | No conectar cortes, pérdidas o identidades diferentes |

Estos umbrales son heurísticas conservadoras, sin calibración demostrada. Revisar sus falsos rechazos sobre ataques rápidos y cambios de perspectiva antes de ajustarlos. Geometría plausible puede pertenecer a la persona incorrecta: la auditoría de identidad es obligatoria.

Salida: reporte geométrico por acción/cuadro/articulación y artefacto derivado con procedencia. Mantener pendiente la aprobación visual de articulaciones.

## 7. Máscaras y coordenadas

| Campo | Significado |
|---|---|
| `point_mask` | Articulación retenida tras identidad y geometría |
| `frame_mask` | Disponibilidad de pose del atleta en ese cuadro; documentar la regla exacta |
| `velocity_mask` | Punto válido en ambos tiempos, continuidad temporal e identidad/track compatibles |
| `target_mask` | Supervisión de etiqueta permitida; `ambiguous` siempre excluido |
| `training_mask` | Ejemplo finalmente aprobado tras todos los controles |

En el NPZ piloto R1, `frame_mask` significa **al menos una articulación retenida**, no cuerpo suficiente para normalización. `target_mask` solo distingue etiqueta no ambigua; no autoriza entrenamiento. `training_mask` permanece falso para todas las acciones.

Guardar coordenadas inválidas y confianza a cero, con máscara falsa. No interpolar ni inventar articulaciones. Documentar unidades, orden de articulaciones, ejes, padding y orden atacante/rival. Mantener máscaras separadas para cada atleta y para objetivos simultáneos; no convertir al defensor en negativo automático.

La cobertura del atacante se calcula sobre todos los cuadros del evento por 17 articulaciones. El umbral inicial >= 65 % solo selecciona candidatos a revisión visual. Informar por separado cobertura de rival, articulaciones necesarias para la técnica y contexto temporal; un promedio alto no garantiza que la muñeca o tobillo de interés sea visible.

## 8. Validación visual y refinamiento

1. Superponer las articulaciones retenidas sobre el RGB original: atacante verde, rival azul, con leyenda. Mostrar exclusiones e identidad dudosa de forma explícita.
2. Revisar todas las acciones destinadas a entrenamiento, todos sus cuadros y el contexto temporal. El PDF de seis muestras por acción ayuda a inspeccionar, pero no sustituye esta revisión exhaustiva.
3. Revisar especialmente golpes simultáneos, oclusiones, miembros mezclados, cortes, cruces de identidad, patadas y clinch. Corregir referencias y máscaras con evidencia; si el punto no es verificable, excluirlo.
4. Regenerar derivados tras cambios y registrar qué acciones/cuadros recibieron aprobación visual. No marcar aprobación humana por ejecutar un script.

Si la pose fuente es irrecuperable, conservar el evento RGB y excluir la supervisión afectada. Una reextracción o reparación de pose debe quedar en una fuente nueva con hash y auditoría propios.

## 9. Adaptador temporal para v4 y validación de corpus

Tarea de desarrollo pendiente: conectar el contrato de anotación directa y las máscaras aprobadas al constructor real de dataset, sin simular ventanas binarias ni reutilizar esquemas con significado distinto.

- Verificar en código los requisitos reales del modelo: articulaciones, normalización, origen/escala corporal, pasos temporales, contexto, FPS, orden de atletas y objetivos.
- Auditar los cuadros de contexto antes de incorporarlos. El NPZ piloto contiene solo cuadros de las acciones y no puede alimentar v4 directamente.
- Respetar máscaras al normalizar, calcular velocidades y pérdida. Un cuadro sin origen/escala fiable no debe producir características válidas por relleno.
- Comprobar identidad antes y después de remuestreo; detener velocidades en cortes y pérdidas. Excluir ambiguos y regiones sin etiqueta.
- Probar carga de ejemplos, dimensiones, finitud, padding, referencias, objetivos simultáneos, clases raras y exclusiones. Separar train/validación/test por pelea para evitar fuga.
- Publicar cuántas acciones quedan aprobadas y cuántas se excluyen por clase, con motivos. `training_ready=true` requiere validación visual y pruebas reales del adaptador, no solo cobertura geométrica.

Este plan no incluye ejecutar entrenamiento. La autorización para entrenar se gestiona como tarea posterior.

## 10. Entregables y criterios de cierre

Por round: vínculos vigentes, mapas de identidad y exclusiones, reporte geométrico JSON/Markdown, máscaras con especificación de ejes/unidades, validaciones reproducibles y estado por fase. Cuando el usuario solicite PDF, incluir resumen y cada acción por clase, atleta, frames, cobertura y estado; señalar si las imágenes son muestras.

Verificar que el PDF corresponda a la revisión vigente, se renderice correctamente y no presente candidatos como aprobados. Conservar el JSON de etiquetas y las poses originales intactos. Actualizar el archivo de contexto y el punto de continuación.

| Estado | Condición |
|---|---|
| RGB completo | Todas las hojas realmente inspeccionadas y etiquetas validadas |
| Identidad auditada | Todas las acciones tienen decisión, incluso exclusión |
| Geometría automática | Filtros y máscaras generados y comprobados |
| Articulaciones validadas | Evidencia visual y aprobación registradas por alcance |
| Corpus v4 validado | Adaptador probado con contexto auditado y exclusiones |
| Listo para entrenamiento | Todas las condiciones anteriores para el subconjunto aprobado |

## 11. Referencia real del piloto R1

Volkanovski-Holloway I, UFC 245, round 1: 562 hojas, 8.991 cuadros, 155 registros (142 positivos y 13 ambiguos). Auditoría automática: 107 candidatos con cobertura del atacante >= 65 %. Ninguna acción tiene aprobación final de entrenamiento en el NPZ piloto.

Artefactos de referencia:

- `scripts/annotation/build_round1_geometry_masks.py`: generación de filtros/máscaras del piloto; contiene resolución y exclusiones específicas de R1. Generalizar esos valores antes de reutilizar en otros rounds.
- `output/annotation_audits/volkanovski_holloway_i/round1_geometry_masks/`: NPZ, reporte y especificación actual.
- `scripts/annotation/pdf_round1_geometry.py`: catálogo de seis muestras por acción.
- `output/pdf/volkanovski_holloway_i_round1_poses_mascaras.pdf`: 157 páginas; revisión de resultados, sin certificación de entrenamiento.

No extrapolar el refinamiento de R1 a los rounds 2-5: su revisión RGB y la existencia de poses no demuestran que ya tengan esta auditoría.

Plan de entrada: [Plan de autoanotación directa](PLAN-ANOTACION-DIRECTA-FIGHTS-REMAINING-MIXED.md).
