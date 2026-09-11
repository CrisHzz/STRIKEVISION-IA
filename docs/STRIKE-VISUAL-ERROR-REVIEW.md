# Revisión visual de errores de validación

## Correcciones autorizadas posteriormente

Tras autorización explícita del usuario, se inspeccionaron los 24 frames de
cinco casos en láminas completas y se corrigieron las etiquetas:

- Aldo–Font R2, 684–707: no_strike a strike.
- Zhang–Joanna R1, 4284–4307: no_strike a strike.
- Zhang–Joanna R4, 1092–1115: no_strike a strike.
- Zhang–Joanna R1, 8484–8507: strike a no_strike.
- Zhang–Joanna R4, 4728–4751: strike a no_strike.

El criterio aplicado es presencia de intento visible de golpe dentro de la
ventana, incluyendo golpes bloqueados, sin exigir impacto efectivo. Los tres
casos ambiguos conservan sus etiquetas. Se modificaron ventanas de 24 frames,
no los límites de los intervalos ni etiquetas de ventanas vecinas no revisadas.

Registro y copias originales: outputs/annotation-review-v1/changes.json y
outputs/annotation-review-v1/before/. Se actualizaron el manifiesto, su auditoría
y la referencia del split, conservando las mismas peleas por partición. El cache
de train/validación se migró verificando igualdad exacta de embeddings y
actualizando únicamente las etiquetas y los contratos de procedencia.

Los informes y modelos previos permanecen como resultados históricos con las
etiquetas anteriores. No se reentrenó ni se modificó el test. Una eventual subida
de métricas al reevaluar estas cinco ventanas no representa aprendizaje nuevo:
se cambió la referencia y la selección de ejemplos estaba guiada por errores.

El texto y la tabla siguientes documentan la inspección inicial, previa a las
correcciones; los GIF conservan los identificadores originales FP/FN.

Se inspeccionaron ocho secuencias: dos falsos positivos y dos falsos negativos
con puntuaciones extremas por pelea. Es una selección intencionada de casos
difíciles; no estima la frecuencia de problemas en todo el dataset.

Se exportaron los 24 frames originales de cada ventana mediante decodificación
secuencial desde el inicio del video, sin búsquedas aproximadas. Las láminas
muestran seis frames distribuidos dentro de cada ventana; los GIF incluyen los
24. La inspección del asistente se realizó sobre las ocho láminas. La etiqueta y
los límites de cada caso coinciden con el archivo de anotación original. No se
modificaron etiquetas, modelos ni métricas. No se accedió al test.

## Observaciones

| Caso | Etiqueta original | Puntuación strike | Observación visual |
|---|---|---:|---|
| Aldo–Font R5, frames 4716–4739 | no_strike | 0,923 | Transición de intercambio a agarre de cabeza/clinch; distinguir golpe de control requiere revisar la secuencia completa. |
| Aldo–Font R2, frames 684–707 | no_strike | 0,910 | Se observan extensiones de puños hacia la guardia/cabeza y esquiva. Candidato a revisar el criterio de strike o alineación. |
| Aldo–Font R3, frames 3564–3587 | strike | 0,041 | Posición de control en el suelo y movimientos cortos parcialmente ocultos. No se confirma un impacto con estas imágenes. |
| Aldo–Font R5, frames 1752–1775 | strike | 0,058 | Intercambio muy cerrado contra la reja, con brazos/cuerpo ocultos por el árbitro y el rival. |
| Zhang–Joanna R1, frames 4284–4307 | no_strike | 0,963 | Se observan extensiones de brazos y una patada baja. Alta prioridad para revisar la definición de la etiqueta. |
| Zhang–Joanna R4, frames 1092–1115 | no_strike | 0,956 | Intercambio de puños con oclusión parcial del árbitro. Alta prioridad para revisión humana. |
| Zhang–Joanna R1, frames 8484–8507 | strike | 0,003 | Predominan guardia, aproximación y giro de cámara. No se aprecia un golpe inequívoco en los seis frames. |
| Zhang–Joanna R4, frames 4728–4751 | strike | 0,026 | Predominan guardia y desplazamiento. No se aprecia un golpe inequívoco en los seis frames. |

## Interpretación y siguiente paso

La muestra sugiere posibles discrepancias entre la definición humana de strike
y el movimiento visible, además de escenas realmente difíciles por oclusión o
control en el suelo. No permite atribuir todos los errores al modelo ni afirmar
que las anotaciones sean incorrectas. Tampoco descarta desalineación en la
reproducción usada al anotar: confirmar la etiqueta almacenada no verifica qué
vio el anotador en ese momento.

Antes de otro entrenamiento, confirmar si strike significa cualquier intento
de golpe (incluidos bloqueados/fallados) o solo contacto efectivo, y si incluye
golpes en clinch y suelo. Revisar las secuencias completas con ese criterio.
Una corrección de validación cambiaría la referencia de evaluación; por sí sola
no demostraría que el modelo mejoró. Las futuras revisiones deberían incluir
también muestras aleatorias y aciertos, sin tomar la predicción como verdad.

## Secuencias exportadas

Los GIF contienen exclusivamente la ventana anotada, sin añadir segundos ni
extender sus etiquetas. FP/FN significan discrepancia respecto a la etiqueta
almacenada. La puntuación no se interpreta como confianza calibrada.

![Aldo Font FP 1](../outputs/strike-detection/head-v1/visual_review/aldo_font_FP_1.gif)
![Aldo Font FP 2](../outputs/strike-detection/head-v1/visual_review/aldo_font_FP_2.gif)
![Aldo Font FN 1](../outputs/strike-detection/head-v1/visual_review/aldo_font_FN_1.gif)
![Aldo Font FN 2](../outputs/strike-detection/head-v1/visual_review/aldo_font_FN_2.gif)
![Zhang Joanna FP 1](../outputs/strike-detection/head-v1/visual_review/zhang_joanna_1_FP_1.gif)
![Zhang Joanna FP 2](../outputs/strike-detection/head-v1/visual_review/zhang_joanna_1_FP_2.gif)
![Zhang Joanna FN 1](../outputs/strike-detection/head-v1/visual_review/zhang_joanna_1_FN_1.gif)
![Zhang Joanna FN 2](../outputs/strike-detection/head-v1/visual_review/zhang_joanna_1_FN_2.gif)

La selección y verificación de origen están en selection.json y source_check.json
dentro de outputs/strike-detection/head-v1/visual_review.
