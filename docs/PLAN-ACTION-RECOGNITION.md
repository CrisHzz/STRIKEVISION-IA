# Plan de reconocimiento de acciones de StrikeVision

Fecha: 21 de octubre de 2026. Revisión: 22 de octubre de 2026.
Estado: torneo y sintonización de desarrollo completados. El probador visual
reveló fallos temporales, falsos eventos y problemas de presentación que se
abordan en la sección 8. Las anotaciones no se han modificado. No se ha
acreditado rendimiento de extremo a extremo para producción.

Actualización de resultados: el torneo bruto situó a X3D-S congelado + cabeza
lineal en 0,567 F1 macro principal, frente a 0,487 de TCN. X3D excluyó cerca
del 20 % de los clips por no disponer de recorte de atacante completo. Al
evaluar ambos modelos en exactamente esos clips disponibles para X3D, TCN
obtuvo 0,718 F1 frente a 0,567 de X3D. Por cobertura, coste y comparación
equivalente, TCN es el finalista que se debe sintonizar primero. Estos valores
son de desarrollo, no del test reservado.

## 1. Objetivo y alcance

Extender el sistema actual para reconocer intentos de ataque por peleador:
`punch_head`, `punch_body`, `kick_leg`, `kick_body`, `kick_head` y
`other_strike`. Los golpes fallados y bloqueados cuentan. El destino se
determina por la trayectoria, no por contacto confirmado.

Conservar el detector binario strike/no_strike, la detección de personas,
el merge tracking y las poses como componentes independientes. El nuevo
reconocedor complementará esa cadena. La primera entrega será de desarrollo:
notebook, artefactos reproducibles, resultados del torneo y Gradio.
Producción se abordará posteriormente.

## 2. Inventario verificado

Fuente: `data/annotations/strike_actions_v1/*.json`, contrastada con las
anotaciones binarias y los archivos de pose referenciados. Estos son registros
humanos de acción, no un conteo demostrado de golpes individuales.

| Etiqueta | Registros | Atacante 1/2 conocido | Peleas con ejemplos |
|---|---:|---:|---:|
| punch_head | 1155 | 1139 | 8 |
| kick_leg | 265 | 262 | 8 |
| punch_body | 158 | 157 | 8 |
| kick_body | 123 | 122 | 8 |
| kick_head | 90 | 86 | 8 |
| other_strike | 11 | 11 | 6 |
| ambiguous | 837 | 34 | 8 |
| Total | 2639 | 1811 | 8 |

- 34 rounds anotados de 8 peleas; 1950/1950 fragmentos terminados.
- 1802 registros con técnica específica; 1777 con atacante conocido.
- 49 correcciones explícitas a no_strike, almacenadas por separado.
- 0 duplicados exactos por round, intervalo, atacante y etiqueta.
- 0 intervalos invertidos; 0 acciones huérfanas respecto a los fragmentos
  actuales; 0 discrepancias del hash de las fuentes binarias.
- Los videos y archivos de pose referenciados existen.
- 25 técnicas tienen atacante desconocido: conservar, pero excluir del
  entrenamiento supervisado por peleador hasta resolver su identidad.
- Hay 84 pares solapados con distintas técnicas para el mismo atacante.
  Pueden representar combinaciones válidas o ventanas de contexto compartidas.
  No eliminar ni resolver por mayoría automáticamente.
- Las duraciones medianas rondan 0,8 segundos: la interfaz proponía ventanas
  de 24 frames. No equivalen a límites precisos de inicio/fin de cada golpe.
- FPS de las anotaciones: 17 rounds a 30, 13 a 29,970 y 4 a 24. Usar tiempo
  real y metadatos; no reinterpretar frames como si todos fueran a 30.
- 11 rounds usan IDs históricos `fighter_left/right`. Aplicar la función
  existente `display_fighter_id`: left -> 1 rojo, right -> 2 morado.
  Esta equivalencia reproduce la interfaz; no demuestra identidad correcta.

Tras normalizar IDs, en las 1777 acciones con atacante conocido hay 40004
frames marcados `pose_valid` de 42876 frames de acción, aproximadamente 93,3 %.
El denominador suma intervalos y puede contar dos veces frames solapados;
no es una proporción de frames únicos del dataset.
1566 acciones tienen al menos 80 % de frames con pose válida; 8 no tienen
ninguno. `pose_valid` no certifica precisión de articulaciones o identidad.

R3/R4 de Adesanya–Pereira ya están en `untrusted_pose_videos`: respetar esa
marca, enmascarar las poses y las cajas no fiables. Mantener las etiquetas
humanas como material de revisión; no asignarlas a una persona incorrecta.

Las 8 peleas son Adesanya–Pereira, Aldo–Font, Fiziev–Bahamondes,
Holloway–Gaethje, Romero–Adesanya, Topuria–Gaethje, Topuria–Holloway y
Yan–Sandhagen. El inventario se basa en los archivos, no en verificar la
identidad o autenticidad deportiva de sus nombres.

Hay además 20 rounds disponibles sin archivo de acciones en las categorías
low_kicks_men, low_kicks_women, passive_women y aggressive_women, repartidos
entre Fiziev–Green, O'Malley–Zahabi, Joanna–Andrade, Nunes–Shevchenko y
Zhang–Joanna. No se incluirán como acciones etiquetadas. Se puede empezar
con los 34 terminados; el alcance de generalización actual es masculino.

Conclusión: base viable para desarrollar y comparar las cinco técnicas
principales. La suficiencia final se decidirá mediante validación por pelea
y curvas de aprendizaje. `other_strike`, con 11 casos heterogéneos, será una
salida experimental sin afirmación de fiabilidad. No ampliar a jab/gancho/
volado porque estas anotaciones no contienen esa distinción.

## 3. Reglas de etiquetas, tiempo e identidad

1. `ambiguous` se preserva para revisión, pero no es una clase del modelo,
   ni un positivo de no_strike, ni un negativo de las seis técnicas.
   Representa información desconocida de causas mezcladas.
2. Construir máscaras de supervisión por tiempo, peleador y categoría.
   En solapamientos conservar positivos explícitos fiables; no fabricar
   negativos de clases/personas sin supervisión inequívoca. Si una ventana
   contiene acciones con atacante desconocido, enmascarar las asignaciones
   que puedan estar afectadas. No propagar la ambigüedad de un peleador a
   una acción clara del contrario sin motivo temporal o de identidad.
3. Usar no_strike confirmado de las anotaciones binarias y las 49
   correcciones, tras reconciliarlas en una vista derivada con prioridad de
   corrección y reporte de conflictos. No reescribir originales. Solo un
   intervalo confirmado sin ataques permite negativos globales para ambos.
4. Un fragmento terminado indica revisión humana; no garantiza límites
   temporales exhaustivos por clase. No etiquetar todo tiempo sin marca
   como reposo ni obligar a una única técnica por fragmento.
5. Predicción por peleador con seis probabilidades sigmoid y máscara de
   pérdida: admite técnicas solapadas y ataques simultáneos de ambos IDs.
   Los baselines que requieran etiqueta única usan únicamente el subconjunto
   inequívoco; su comparación debe publicar soporte y conjunto común.
6. El estado visual «sin clasificación fiable» será abstención por
   confianza/calidad, no una séptima técnica entrenada llamada ambiguous.
   Si el detector binario da no_strike, mostrar ese estado. Si da strike
   pero la técnica o el atacante no son fiables, abstenerse de atribuirlos.
7. No atribuir un strike global a ambos peleadores. Entrenar presencia de
   acción por ID usando positivos fiables y reposos confirmados; evaluar
   específicamente al oponente que solo defiende durante un strike global.
   Si esos negativos son insuficientes, declarar la limitación y producir
   abstenciones, sin inventar etiquetas del defensor.
8. Mantener IDs como ranuras de tracking según colores. No vincularlos a
   nombres de atletas ni asumir continuidad tras cambios de identidad.
9. Conservar frames y FPS originales. Muestrear secuencias por timestamps
   a una cadencia temporal común, sin exportar video a 60 FPS.
10. El objetivo inicial es reconocer momentos de técnica. Un contador
    exacto de golpes requiere límites/eventos individuales verificados;
    las ventanas actuales no garantizan separar dos puños consecutivos.

## 4. Preparación y particiones

Antes de entrenar, generar un manifiesto con action_id, pelea, round,
intervalo, segundos, ID normalizado, etiquetas, máscaras, confianza de pose,
fuentes y hashes. Exportar también los casos excluidos y sus razones.
Repetir este inventario al ejecutar para detectar cambios desde el plan.

Normalizar coordenadas usando centro y escala corporal del atacante;
añadir articulaciones del rival relativas al atacante, distancias de
muñecas/tobillos al objetivo, vectores óseos, velocidades por segundo,
confianzas y máscaras de ausencia. Esto representa objetivos corporales;
no convertirlo en reglas rígidas de clase. No interpolar a través de cortes,
cambios de identidad o ausencias largas. Ajustar toda normalización aprendida
solo con entrenamiento.

Usar ventanas temporales centradas en acciones, varias escalas de contexto
y muestreo deslizante en inferencia. Las ventanas de contexto deben poder
tener múltiples etiquetas. No centrar inferencia en marcas humanas futuras.
Los intervalos anotados son supervisión temporal aproximada.

Reservar dos peleas completas para test antes de experimentar; seleccionar
la pareja mediante una regla de cobertura de etiquetas y tamaño fijada en
el manifiesto, sin inspeccionar predicciones. En las seis restantes, usar
3 folds agrupados por pelea para selección y ajuste. Ningún round, recorte,
ventana, aumento o caché supervisada de una pelea puede cruzar particiones.
Comprobar cobertura de cada clase por fold y reportar clases ausentes.

Agrupar por pelea evita compartir rounds, pero varios atletas reaparecen
en peleas distintas. Documentar ese solapamiento y añadir una evaluación
secundaria con pares de peleas sin atletas compartidos entre train y test
cuando sea factible, sin seleccionar configuraciones con sus resultados.
No prometer rendimiento en atletas completamente nuevos con el test común.

Mantener iguales particiones, etiquetas, evaluación, recursos y semillas
para todos los candidatos. La confianza y los umbrales se calibran dentro
del desarrollo; el test queda cerrado. El pipeline binario existente pudo
haber usado estas peleas: su evaluación integrada será condicional a ese
pipeline, no una prueba independiente de toda la cadena en videos nuevos.

## 5. Torneo de modelos

No es posible declarar de antemano un modelo universalmente óptimo.
La hipótesis inicial es que un modelo temporal ligero con pose del atacante
y contexto del rival dará una buena relación rendimiento/coste. La fusión
con video puede aportar información cuando las poses o trayectorias fallen.

| Candidato | Entrada | Motivo |
|---|---|---|
| A. Regresión logística regularizada | Estadísticas de pose y movimiento | Referencia rápida y auditable |
| B. ExtraTrees | Las mismas características | Relaciones no lineales con coste bajo |
| C. TCN compacta | Secuencias de pose de atacante y rival | Hipótesis principal de eficiencia |
| D. ST-GCN++ compacto | Grafo temporal de articulaciones | Comparador especializado en esqueleto |
| E. X3D-S congelado + cabeza nueva | Clips RGB centrados en atacante y contexto | Reutiliza backbone disponible; aprende estas técnicas |
| F. Fusión pose + RGB | Características de C/D y E | Complementariedad, solo si aporta en validación |

Los dos primeros usarán salidas por clase y supervisión compatible con
máscaras; no usar un clasificador multiclase sobre ventanas contradictorias.
Para ST-GCN++ revisar compatibilidad de 17 articulaciones COCO, orden,
confianza, licencia, pesos y versiones. Un checkpoint genérico no reconoce
estas categorías MMA sin adaptar y entrenar su cabeza. Si su instalación
requiere cambiar el entorno actual, aislarlo y registrar la incidencia.

Para E usar pesos originales Kinetics de X3D-S y una cabeza nueva de
acciones; no reutilizar logits binarios como técnicas. No usar un backbone
ajustado con peleas del test. Los embeddings de cuerpo completo binarios
existentes no equivalen a embeddings de recortes por atacante: reutilizar
solo si coinciden pesos, FPS, frames, recortes y transformación.

Evaluar PoseC3D como candidato de reserva si el modelo de grafo falla por
ruido y el benchmark permite su coste. VideoMAE/V-JEPA 2 son referencias
válidas, pero no serán el primer paso en este entorno CPU de 15,9 GiB:
requerirían medir extracción/memoria antes de ampliar el torneo.

Fases del torneo:

1. Medir lectura, extracción, inferencia y memoria en una muestra pequeña
   reproducible. No entrenar durante el inventario. Registrar coste fijo de
   preparación separado del de optimización y del de renderizado.
2. Comparar A/B y pilotos C/D sobre los mismos folds. E entra si la
   extracción proyectada respeta el límite; F solo tras disponer de C/D y E.
3. Optimizar las dos familias más prometedoras. Evaluar la ganancia de F
   con predicciones fuera de fold; su calibrador tampoco ve el test.
4. Elegir en desarrollo. Abrir test una sola vez para el ganador y la
   referencia fijada de antemano; no cambiar el ganador según test.
5. Reentrenar posteriormente un artefacto de desarrollo con los datos
   permitidos y una configuración bloqueada; diferenciarlo del modelo cuya
   métrica proviene del test. No llamar test a datos usados en reentreno.

## 6. Sintonización, presupuesto y evaluación

Optuna con TPESampler, semillas fijas, estudio SQLite reanudable y poda
de ensayos poco prometedores en modelos iterativos. A/B se evalúan por
fold; no simular poda por época en modelos sin entrenamiento iterativo.
Guardado atómico de artefactos, checkpoints y manifiestos; cachés por hash
para no repetir extracción tras una interrupción.

Presupuesto inicial por defecto: pilotos de hasta 5 minutos de entrenamiento
por familia; luego hasta 20 ensayos por cada una de las dos finalistas,
con límite agregado de 45 minutos de búsqueda y comprobaciones de tiempo
entre folds/épocas. Ningún ensayo puede comenzar si excede la proyección
restante. Son límites operativos, no promesas de duración total o de poder
completar todos los ensayos. Extracción RGB con presupuesto separado de
30 minutos tras el benchmark; si no cabe, conservar resultados y marcar
ese candidato aplazado. Se podrá ampliar expresamente el presupuesto.
Máximo 50 épocas por ensayo temporal, early stopping de 6; una época es
un límite de trabajo, no una obligación de llegar a 50.

Espacios iniciales, ajustables solo dentro del desarrollo:

- A: C entre 1e-3 y 1e2; balance uniforme o por frecuencia de entrenamiento.
- B: 100/200/400 árboles, profundidad 8/16/sin límite,
  min_samples_leaf 1/2/4/8, max_features sqrt/0,5/1,0.
- C/D: learning_rate 1e-4 a 3e-3, weight_decay 1e-6 a 1e-2,
  dropout 0 a 0,5, canales 32/64/96, 2/3/4 bloques,
  batch 16/32/64, 16/24/32 pasos temporales y contexto 0,5/0,8/1,2/1,6 s.
- E: cabeza lineal o MLP pequeña sobre embeddings congelados, regularización
  y dropout; no ajuste completo del backbone en esta etapa CPU.
- Todos: pesos de clase moderados calculados solo en train, BCE enmascarada
  frente a focal enmascarada, umbrales por clase validados, suavizado temporal
  y fusión de intervalos. Aumentos solo en train; limitar desplazamientos
  temporales para no cambiar la técnica visible.
- F: pesos de fusión y calibración ajustados exclusivamente fuera de fold.

La búsqueda usará macro-F1 de las cinco técnicas principales en el conjunto
común conocido, sin abstención, y publicará también macro-F1 de seis clases.
No ocultar `other_strike`: reportar sus 11 casos, resultados e incertidumbre
por separado. No premiar un candidato por descartar ejemplos difíciles.

Publicar además precisión/recall/F1/AP por clase, micro-F1, confusiones,
resultados por pelea, cobertura y precisión al abstenerse, asignación de
atacante y curvas de aprendizaje con cantidades crecientes de peleas.
Para multilabel usar métricas por clase; una matriz multiclase simple solo
se aplica al subconjunto de técnica única. Mostrar soporte y máscaras.

Evaluación temporal: emparejamiento uno a uno por ID y técnica, AP/F1 a
tIoU 0,1/0,3/0,5 y examen visual de límites; son indicadores respecto a las
ventanas humanas aproximadas. No presentarlos como precisión de impacto ni
como conteo exacto de golpes. Registrar falsos eventos por minuto en segmentos
con no_strike confirmado; evitar medir falsos positivos en tiempo desconocido.
Comparar clasificación con fragmentos humanos contra la cadena automática
completa para separar errores del reconocedor, detector binario y tracking.

Latencia: ms por ventana, frames/s y memoria, tanto del reconocedor solo
como del pipeline completo y del render. Registrar versiones y CPU.
Repetir las dos finalistas con 3 semillas si el presupuesto lo permite y
calcular dispersión por pelea; con solo dos peleas de test, los intervalos
de confianza serán necesariamente débiles. Una mejora menor que la
variabilidad o claramente más lenta no justifica preferir el modelo complejo.

Objetivo orientativo de desarrollo: macro-F1 de las cinco clases >=0,75,
con vigilancia individual de recall y revisión visual de fallos. No es una
predicción ni requisito de producción; la aprobación de producción requerirá
videos independientes, coste aceptable y criterios específicos posteriores.

## 7. Notebook, Gradio, entregables y orden de ejecución

Preparar `notebooks/ActionRecognition/action-recognition.ipynb` con secciones
reanudables: inventario; controles de calidad; particiones; características;
benchmark; pilotos; Optuna; torneo; test bloqueado; exportación; prueba visual.
La lógica reutilizable vivirá en módulos y el notebook la orquestará.
Ejecutar una celda nunca iniciará inadvertidamente todo el entrenamiento.
Instalación de dependencias y entrenamiento serán pasos explícitos.

Reutilizar `strikevision_ML` si las versiones son compatibles. No actualizar
dependencias a ciegas ni alterar el entorno que mantiene Gradio; aislar
dependencias incompatibles de competidores si hiciera falta.

Gradio de desarrollo separado del anotador, con estilo del probador actual:
video subido o round local; cantidad de frames; atajos 30 s, 1 min, 1:30,
2 min, 2:30 y 3 min; hasta 9000 frames a 30 FPS para cinco minutos.
Al cargar archivos históricos mostrar duración derivada del FPS real.
Selector de candidato/exportación, umbral avanzado opcional y progreso.
Una sola visualización con ID1 rojo/ID2 morado, pose, strike/no_strike,
técnica(s), confianza y abstención. Línea temporal y tabla de intervalos
por ID exportables a JSON/CSV. Si se incluyen contadores, llamarlos episodios
predichos de técnica y explicar su criterio de unión. No afirmar golpes
individuales hasta disponer de supervisión y evaluación de ese conteo.

Entregables previstos:

- Informe de inventario y calidad reproducible, exclusiones y manifiesto
  de particiones con hashes; originales de anotaciones intactos.
- Módulos de dataset temporal, características, modelos, evaluación,
  optimización, caché e inferencia, con configuración y registro de versiones.
- Notebook y entrada de Gradio independientes.
- Tabla del torneo con rendimiento, soportes, cobertura, coste y razones
  de selección; historial Optuna y checkpoints reanudables.
- Artefacto ganador con clases, FPS/cadencia, normalización, umbrales,
  máscaras, política de identidad y procedencia de pesos.
- Verificaciones de tiempo/FPS, equivalencia de IDs históricos, ausencia de
  filtración entre peleas, máscaras ambiguous, acciones simultáneas,
  coherencia entrenamiento/inferencia, reanudación y límites de memoria.

Orden tras autorización: implementar inventario y contrato -> crear
particiones -> preparar características/caché -> preparar notebook/torneo ->
preparar Gradio -> verificar y entregar comandos. Se mantiene la instrucción
previa del usuario: él ejecutará el entrenamiento. Autorizar la codificación
de este plan no autoriza entrenamientos largos automáticos.

No hace falta información adicional para empezar a implementar con estas
decisiones por defecto. Consultar si una decisión altera categorías, cambia
anotaciones originales, exige presupuesto o hardware externo, o implica
una modificación sustancial de alcance. Los resultados del torneo serán
evidencia, no una garantía anticipada de rendimiento.

## 8. Plan de reparación de la inferencia de acciones (22 de octubre de 2026)

### Diagnóstico reproducible

Primer corte implementado el 22 de octubre: el probador ahora conserva las
candidaturas crudas en `results.json`, exporta `action_timeline.csv`, usa el
centro de la ventana como instante de decisión y exige confianza >=0,50 y
cobertura de pose >=0,65 para el HUD. Cada ID muestra una sola técnica
principal en una tarjeta junto a su caja; los eventos de técnicas distintas
no se solapan para el mismo ID. En 30 s de
`DuPlessisvsUsman _UFCOklahomaCityCUTTED.mp4` se generaron 18 eventos cortos
(máximo 0,83 s), frente a eventos de hasta 4,8 s de la versión previa. Es una
mejora de localización y lectura, no una validación de clase.

La auditoría del candidato reveló un riesgo de entrenamiento: las muestras de
acción marcan como conocidas solo sus técnicas positivas; las demás clases se
enmascaran salvo en no-strike confirmado. Esa política evita inventar
negativos, pero permite activaciones simultáneas no penalizadas. Antes de
reentrenar hay que decidir y documentar cuándo la ausencia de otra técnica
en una anotación completa es un negativo fiable y cuándo sigue siendo
desconocida.

Decisión V2 confirmada por el anotador: intentó registrar todas las acciones
visibles desplazándose por la línea temporal, incluso cuando había varias en
unos cinco segundos. Se creó un corpus derivado separado que trata las clases
no marcadas como negativas dentro de una acción, excepto si se cruza con
`ambiguous` o con atacante desconocido. Resultado: 1751/1777 acciones tienen
supervisión completa y 26 conservan supervisión parcial. Los originales y el
dataset V1 permanecen intactos.

La variante V2 preparada añade una puerta explícita acción/no-acción por ID y
clasifica la técnica solo después de superarla. Su encoder combina atención
temporal con el pico temporal para evitar que jabs, rectos y cross de pocos
frames se diluyan al promediar toda la ventana. Debe compararse por recall de
acciones rápidas, falsos eventos por minuto y error temporal; todavía no se ha
entrenado ni se afirma que mejore esas métricas.

Ejecución manual prevista en dos pasos. Primero un piloto de cinco ensayos y
un solo fold; solo si reduce falsos eventos y conserva recall se ejecutan los
tres folds y más ensayos. El script acepta configuración, output separado,
sufijo y selección de folds para no reutilizar ni sobrescribir V1. El usuario
mantiene el control de cualquier entrenamiento largo.

Resultado del piloto V2 fold 0: no aprobado. El mejor ensayo produjo score
inicial 0,340, pero esa selección combinaba métricas separadas. Al medir la
cadena completa, su mejor macro-F1 de técnicas fue 0,251 con puerta 0,61. El
TCN V1, evaluado sobre exactamente las mismas etiquetas completas V2, obtuvo
0,317 con umbrales guardados y 0,328 al recalibrarlos. En 30 s del video de
Du Plessis–Usman, V2 emitió 42 eventos frente a 18 de V1. No ejecutar los otros
folds de esta variante. La evaluación quedó corregida para que futuros modelos
jerárquicos se seleccionen por F1 posterior a la puerta, no por dos métricas
independientes.

Siguiente ablación: TCN V1 sin puerta ni atención, entrenada sobre el corpus
V2 con los hiperparámetros ganadores ya encontrados en fold 0. Cambia solo la
supervisión y evita repetir cinco o veinte ensayos. Si no supera al V1 bajo
las mismas etiquetas y la prueba visual, se descarta la conversión exhaustiva
como solución principal y se priorizan límites temporales/hard negatives.

Resultado de la ablación: `tcn-labels-v2-fold0` alcanzó macro-F1 0,331 frente
a 0,328 del TCN V1 recalibrado sobre las mismas etiquetas. La diferencia de
0,003 no justifica reemplazar el candidato vigente. En los primeros 900 frames
de Du Plessis–Usman produjo 22 eventos (20 `punch_head`, 2 `kick_leg`) frente a
18 del TCN V1. Recuperó algunos jabs rápidos, pero añadió al menos un falso
positivo, duplicó una secuencia prolongada y confundió un puño con `kick_leg`.
Se conserva como candidato de comparación en Gradio y no se promueve todavía.

El probador actual usa el mejor ensayo TCN de un fold de desarrollo, no un
modelo final entrenado y evaluado sobre videos nuevos. En
`action_recognition/inference.py` se predicen ventanas de 0,8 s con paso de
0,4 s; una probabilidad que supera el umbral marca toda la ventana y las
ventanas contiguas se unen. La posición temporal resultante no localiza el
instante de ejecución de la técnica. En la vista se pintan los primeros dos
eventos activos en una barra común, sin agrupar por ID. No se aplica puerta
de presencia de acción por peleador ni supresión por calidad de pose.

Los dos análisis de Gradio guardados tienen 79 eventos en 90,1 s y 174 en
180,2 s. El segundo contiene 53 eventos `punch_head`, 44 `other_strike` y
244 pares de etiquetas diferentes solapadas para el mismo ID. El umbral de
`other_strike` es 0,05. Son síntomas verificables de saturación y etiquetas
simultáneas; no demuestran que cada predicción sea falsa. El F1 de ventanas
del torneo tampoco mide el retraso visual, los falsos eventos por minuto,
la atribución al atacante ni la legibilidad de la pantalla. Se debe corregir
la forma en que hemos presentado esas métricas al juzgar el producto.

### Fase 0. Línea base y casos de referencia

- Conservar el código, los modelos y los JSON de las dos ejecuciones para
  comparación A/B. No sobrescribir ni reanotar datos originales.
- Seleccionar fragmentos cortos que incluyan golpes claros, combinaciones,
  ataques simultáneos, fintas, guardia, distancia sin acción, clinch y un
  fallo de tracking. Registrar por ID y tiempo aproximado qué ocurre y cuándo
  debería aparecer/desaparecer la etiqueta. Distinguir hechos observados de
  tiempo desconocido. Usar parte de los videos nuevos sin utilizarlos para
  escoger umbrales; separar calibración y comprobación final por pelea.
- Exportar para cada frame/ventana: probabilidad por clase e ID, cobertura y
  confianza de pose, ventana que originó el evento, umbral, inicio/fin mostrado
  y estado del tracking. Generar una línea temporal de diagnóstico alineada
  con el video.
- Medir dos latencias distintas: tiempo de procesamiento hasta obtener el
  archivo y desfase entre la acción visible y su etiqueta en el video ya
  renderizado. Este probador procesa el video por lotes; no existe una cola
  de predicción en vivo, aunque la visualización pueda parecerlo.

### Fase 1. Corregir localización y formación de eventos sin reentrenar

- Mantener las probabilidades originales y proyectarlas sobre una línea
  temporal por ID. Comparar atribuir cada ventana a su centro, a su final y
  a un tramo calibrado con ejemplos revisados. Evitar presentar la ventana
  completa como si el golpe durara 0,8 s. La opción causal solo será necesaria
  si posteriormente se pide predicción en vivo.
- Aplicar entrada/salida con umbrales distintos, duración mínima, separación
  breve entre eventos y límite de unión para que una serie de ventanas no
  convierta ataques independientes en una etiqueta de varios segundos.
  Conservar verdaderas combinaciones y acciones simultáneas en el JSON.
- Introducir un estado por peleador: reposo confirmado, acción candidata,
  técnica confiable, o abstención por identidad/pose insuficiente. Evaluar
  una puerta de acción por ID usando positivos fiables y negativos revisados;
  el detector binario global no puede asignar automáticamente el ataque a
  ambos peleadores. `ambiguous` sigue fuera de la clase no_strike.
- Recalibrar umbrales por clase en peleas de desarrollo distintas de la
  comprobación final. Revisar en particular `punch_head` y `other_strike`;
  subir un umbral sin medir recall no se considerará una solución.

### Fase 2. Interfaz de lectura inmediata

- Mostrar dos paneles persistentes, ID 1 rojo e ID 2 morado. Anclar además
  una tarjeta corta junto a la caja de cada peleador cuando el tracking sea
  fiable. Cada panel enseña una técnica principal, su confianza calibrada y
  un historial breve de eventos; las técnicas simultáneas reales se agrupan
  dentro del panel de ese ID, nunca como texto repetido en la misma barra.
- Usar una animación discreta de aparición/desaparición alrededor del momento
  detectado y mantener el rótulo visible un tiempo corto configurable. No
  desplazar el instante registrado en JSON para lograr este efecto. Si se
  pierde el ID, ocultar la tarjeta espacial y señalar que no hay atribución
  fiable. Mantener el video y los IDs legibles a varias resoluciones.
- Añadir una línea temporal navegable por peleador y filtro de técnica. El
  HUD será una vista; el JSON conservará probabilidades y eventos completos.

### Fase 3. Datos y modelo solo si persisten errores de clasificación

- Construir un conjunto dirigido de negativos difíciles: fintas, preparación,
  cambios de guardia, defensa, pasos y amagos frente a punch_head; además de
  problemas de pose e identidad. Verificar manualmente qué peleador estaba
  activo. No convertir intervalos `ambiguous` en negativos.
- Revisar confusiones por clase, peleador, pelea y sexo/categoría antes de
  pedir más anotación. Incorporar las peleas de mujeres ya disponibles con
  las mismas reglas, sin mezclarlas entre train y test por rounds.
- Si el posprocesamiento y la calibración no bastan, entrenar una salida de
  presencia de acción por ID y después comparar TCN actual, TCN con esa salida
  y una variante con contexto RGB. Abrir una arquitectura nueva solo si mejora
  métricas temporales y de falsos eventos con coste razonable. El usuario
  ejecutará manualmente cualquier entrenamiento largo.

### Evaluación y criterio para avanzar

- Comparar versión actual y cada reparación sobre los mismos videos y con
  las mismas marcas de referencia. Publicar precisión/recall/F1 por técnica,
  falsos eventos por minuto en reposo confirmado, error de inicio/fin en
  segundos, atribución al ID correcto, cobertura/abstención, episodios
  partidos o unidos indebidamente y tiempo de proceso por minuto de video.
- Meta de iteración propuesta, sujeta a la precisión de las marcas humanas:
  reducir al menos a la mitad los falsos `punch_head` en reposo/fintas del
  conjunto de comprobación sin una caída importante de recall; error mediano
  de aparición <=0,25 s; cero superposiciones ilegibles del HUD; atribución
  correcta al ID en >=95 % de eventos evaluables. Son criterios de diseño,
  no resultados actuales ni garantía de producción. Informar por separado
  escenas con tracking incierto.
- No elegir una configuración por una sola pelea ni volver a optimizar sobre
  el test reservado. Una vez congeladas reglas, umbrales y artefacto, evaluar
  en videos completos independientes. Si no mejora de forma reproducible,
  documentar el límite y priorizar datos dirigidos antes de otro torneo.

Orden de implementación tras aprobar el plan: diagnóstico y línea base ->
posprocesamiento temporal -> HUD -> evaluación A/B -> decisión sobre nuevas
anotaciones y entrenamiento. La primera ronda no requiere entrenar otra vez.

## Referencias técnicas consultadas

- ST-GCN++ y configuraciones oficiales:
  https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/stgcnpp/README.md
- PoseC3D y checkpoints oficiales:
  https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md
- Modelos de PyTorchVideo:
  https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html
- Muestreo y poda de Optuna:
  https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- Validación agrupada de scikit-learn:
  https://scikit-learn.org/stable/modules/cross_validation.html
- V-JEPA 2, referencia de modelos de video:
  https://github.com/facebookresearch/vjepa2
