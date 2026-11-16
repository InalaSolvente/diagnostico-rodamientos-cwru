# Revisión de la meta 100/105

    La meta sigue pendiente. La referencia más estricta obtiene 79/105 sin retroceso (75,2 %).
Tiene 13 mejoras y 26 resultados desfavorables: 4 descensos de grado y 22 tipos incorrectos.
El objetivo exige también muchas más mejoras que resultados desfavorables; tampoco se cumple.

El nuevo banco de 57 bandas desplazadas obtiene 80/105 con los mismos tipos predichos:
14 mejoras, 66 iguales, 3 descensos y 22 tipos incorrectos (25 desfavorables).
Es un ensayo exploratorio; no se han reemplazado los modelos históricos.

El ensayo aditivo de coherencia cíclica obtiene 82/105: 13 mejoras, 69 iguales,
3 descensos y 20 tipos incorrectos (23 desfavorables). Se verifica cada grupo fuera
del entrenamiento. La coherencia sola dio 64/105 y no se adopta. La meta sigue pendiente:
13 mejoras todavía son menos que 23 resultados desfavorables.

La ampliación temporal a tres vistas de selección obtiene 84/105: 14 mejoras, 70 iguales,
4 descensos y 17 tipos incorrectos (21 desfavorables). Son las mismas 52 señales de desarrollo;
las doce ventanas por archivo son vistas correlacionadas, no registros independientes.
El experimento que aprende la evidencia D/PD como 84 rasgos adicionales y elimina los
controles forzados da 74/105; se conserva como resultado negativo.

Se evalúan los mismos 133 archivos, de los cuales 105 tienen clasificación del paper.
El entrenamiento conserva los 52 registros con falla del desarrollo histórico. Cada grupo
de tipo y tamaño queda completamente fuera del ajuste al evaluarlo, incluyendo todas las
posiciones OR, cargas, sensores y muestreos. Las vistas de un mismo archivo normal permanecen juntas.
Los modelos reciben señal y catálogo completo del fabricante; la etiqueta y la frecuencia
verdadera de falla no son entradas de inferencia. Los ensayos siguientes son exploratorios:
la reutilización reiterada del dataset impide presentarlos como validación externa.

| Ensayo | Sin retroceso | Mejoras | Iguales | Descensos | Tipo incorrecto | Desfavorables |
| --- | --- | --- | --- | --- | --- | --- |
| Referencia, grupos completos | 79 | 13 | 66 | 4 | 22 | 26 |
| Preblanqueo AR32 | 73 | 9 | 64 | 2 | 30 | 32 |
| Plantillas coherentes | 73 | 8 | 65 | 5 | 27 | 32 |
| Selector anidado con ranking | 67 | 4 | 63 | 4 | 34 | 38 |
| Plantillas y modulación | 70 | 9 | 61 | 5 | 30 | 35 |
| Ranking físico monótono | 73 | 9 | 64 | 3 | 29 | 32 |
| Retirar control PD | 77 | 9 | 68 | 3 | 25 | 28 |
| Retirar ambos controles | 68 | 8 | 60 | 3 | 34 | 37 |
| Invariancia de portadora | 77 | 11 | 66 | 4 | 24 | 28 |
| Clasificador y control anidados | 74 | 8 | 66 | 4 | 27 | 31 |
| Banco desplazado de 57 bandas | 80 | 14 | 66 | 3 | 22 | 25 |
| Coherencia cíclica sola | 64 | 4 | 60 | 3 | 38 | 41 |
| Crudo y coherencia cíclica | 82 | 13 | 69 | 3 | 20 | 23 |
| Tres vistas temporales crudo+ACP | 84 | 14 | 70 | 4 | 17 | 21 |
| Evidencia84 aprendida, sin forzar | 74 | 8 | 66 | 2 | 29 | 31 |

Desfavorables incluye todos los tipos incorrectos, además de los descensos con tipo correcto.
No se selecciona retrospectivamente la mejor salida para cada archivo. Los modelos históricos
no se han reemplazado. Retirar el control PD o ambos controles es una regla uniforme para todos los registros.
Sus tipos se congelaron antes de consultar la verificación de las bandas en el segmento futuro.

## Hallazgo estructural

Al incorporar la verificación futura a la auditoría, los selectores de los bancos31 y63
solo permiten conservar 97/105 aun concediéndoles un clasificador y un detector perfectos.
Esto prueba que mejorar exclusivamente el clasificador no basta con esos selectores.
El banco57 aumenta esa cota a 99/105: tampoco permite todavía 100 con su selector fijo.
La auditoría usa etiquetas exclusivamente fuera de la inferencia; no representa el
desempeño de un modelo ciego. Véase [la cota verificada](../auditoria_cota_verificada/INFORME.md).

El control PD puede forzar una familia incorrecta incluso cuando el clasificador ya había
identificado la correcta. Su cota optimista, combinada con el banco31 actual, es 98/105.
Sin ese control, la cota de selección es 100/105. Son límites superiores calculados en una
auditoría con etiquetas, no aciertos de un modelo ni predicciones utilizables en inferencia.
La verificación futura puede reducir aún más esos límites. Véase
[la auditoría](../auditoria_control_tipo/INFORME.md).

Retirarlo resta rendimiento observado (77/105 frente a 79/105), por lo que una solución para
100 necesita resolver tanto la clasificación como la evidencia de las bandas. No se aplican
excepciones a los dos archivos perjudicados ni se altera el criterio de diagnóstico.

El banco57 añade todas las bandas desplazadas medio ancho de forma uniforme. Conserva
dos ramas disjuntas, los armónicos del catálogo, el juez y la separación entre selección
y verificación. Ajusta la probabilidad nominal a .01/(57*5*6*2) para incluir las búsquedas
adicionales. Esa corrección nominal no sustituye una calibración empírica de falsos positivos.
La selección anidada de clasificadores y controles dio 74/105; completar el ensayo de
invariancia de portadora dio 77/105. Se conservan ambos resultados negativos.

## Criterio ND y siguiente etapa

Se publican dos conteos mientras se resuelve la consulta sobre ND. La tabla mantiene la
convención estricta de las matrices recientes. La función original cuenta ND→ND como acierto
sin exigir tipo interno: solo cambia AR32, de 73 a 74; la referencia permanece en 79.
Ambos conteos están en comparacion.csv y no modifican diagnósticos.

La ampliación del entrenamiento está propuesta, pero no ejecutada: continúa vigente la
restricción del usuario a la mitad de los registros con falla. El plan excluye completamente
el grupo evaluado y está en [plan_entrenamiento_ampliado](../plan_entrenamiento_ampliado/).
Requiere resolver la consulta pendiente antes de utilizar más etiquetas de entrenamiento.

La comprobación reproducible de este balance es `python 05_Diagnostico_Ciego/resumir_revision_100.py`.

## Comparación pareada y cobertura completa

Frente al banco57 previo, el ensayo aditivo recupera 6 casos y pierde 4; la ganancia neta es 2. No se mezclan las mejores salidas de ambos modelos por archivo.

Los 133 archivos permanecen en la evaluación; el archivo corto de 48 kHz sigue como sin diagnóstico:

```
                      total  tipo_correcto  sin_diagnostico
cohorte
fallas_12k_con_paper    105             88                0
fallas_48k_sin_paper     24             18                1
normales                  4              4                0
```

Frente al ensayo de 82/105, las tres vistas recuperan 4 casos y pierden 2. Se conserva el mismo criterio y el mismo banco57. Las vistas de 2.5 s comienzan al inicio, al centro y al final de la primera mitad; ninguna termina después de esa frontera.
