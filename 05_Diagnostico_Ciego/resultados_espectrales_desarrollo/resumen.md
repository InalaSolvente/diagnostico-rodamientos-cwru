# Modelo espectral sin frecuencia de falla esperada

Ganador entre 24 ensayos: banco arbol, clasificador lda.
Primera opción correcta: 45/52 fallas (86.54%). Total: 45/54 (83.33%).
Normales correctamente identificados: 0/2.
Se repitió la predicción dejando fuera cada uno de los 16 grupos; normalización y ajuste solo en entrenamiento.
Las cargas y los sensores de una condición permanecen en el mismo grupo.
Entrada de inferencia: señal, muestreo y catálogo completo BPFI/BPFO/BSF/FTF; nunca defecto real ni fd esperada.
La etiqueta se utiliza para entrenar el clasificador, no como entrada al predecir.
L=4 fijo. Seis armónicas para todas las familias y los 31 nodos, más energía, curtosis y entropía de cada nodo.
Los 55 archivos de reserva no se cargaron. La reserva ya se había observado históricamente.
Los parámetros y el ganador se eligieron explorando desarrollo: el porcentaje no es estimación independiente ni prueba externa.
El clasificador aprendió IR/OR/B/N; no hay ejemplos de falla de jaula, por lo que no está validado para FTF como clase.
Los dos normales no se reconocieron fuera de grupo; no sirve aún como detector validado sano/falla.
El modelo exportado se ajustó con todo desarrollo; sus aciertos de entrenamiento no sustentan los porcentajes anteriores.

## Ensayos

| banco | modelo | aciertos | fallas | porcentaje | aciertos_total | total | porcentaje_total | alertas_sanos | sanos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arbol | lda | 45 | 52 | 86.54 | 45 | 54 | 83.33 | 2 | 2 |
| arbol | svm_c1 | 44 | 52 | 84.62 | 44 | 54 | 81.48 | 2 | 2 |
| arbol | svm_c10 | 44 | 52 | 84.62 | 44 | 54 | 81.48 | 2 | 2 |
| combinado | svm_c1 | 44 | 52 | 84.62 | 44 | 54 | 81.48 | 2 | 2 |
| combinado | svm_c10 | 44 | 52 | 84.62 | 44 | 54 | 81.48 | 2 | 2 |
| multibanda | svm_c10 | 43 | 52 | 82.69 | 43 | 54 | 79.63 | 2 | 2 |
| multibanda | lda | 43 | 52 | 82.69 | 43 | 54 | 79.63 | 2 | 2 |
| multibanda | svm_c1 | 41 | 52 | 78.85 | 41 | 54 | 75.93 | 2 | 2 |
| combinado | arboles | 41 | 52 | 78.85 | 41 | 54 | 75.93 | 2 | 2 |
| cruda | svm_c10 | 40 | 52 | 76.92 | 40 | 54 | 74.07 | 2 | 2 |
| cruda | lda | 40 | 52 | 76.92 | 40 | 54 | 74.07 | 2 | 2 |
| doble | svm_c1 | 40 | 52 | 76.92 | 40 | 54 | 74.07 | 2 | 2 |
| multibanda | arboles | 40 | 52 | 76.92 | 40 | 54 | 74.07 | 2 | 2 |
| combinado | lda | 40 | 52 | 76.92 | 40 | 54 | 74.07 | 2 | 2 |
| cruda | arboles | 39 | 52 | 75.0 | 39 | 54 | 72.22 | 2 | 2 |
| doble | lda | 39 | 52 | 75.0 | 39 | 54 | 72.22 | 2 | 2 |
| doble | arboles | 39 | 52 | 75.0 | 39 | 54 | 72.22 | 2 | 2 |
| cruda | svm_c1 | 38 | 52 | 73.08 | 38 | 54 | 70.37 | 2 | 2 |
| doble | svm_c10 | 37 | 52 | 71.15 | 37 | 54 | 68.52 | 2 | 2 |
| arbol | arboles | 36 | 52 | 69.23 | 36 | 54 | 66.67 | 2 | 2 |
| globales | arboles | 25 | 52 | 48.08 | 25 | 54 | 46.3 | 2 | 2 |
| globales | svm_c1 | 24 | 52 | 46.15 | 24 | 54 | 44.44 | 2 | 2 |
| globales | lda | 19 | 52 | 36.54 | 19 | 54 | 35.19 | 2 | 2 |
| globales | svm_c10 | 18 | 52 | 34.62 | 18 | 54 | 33.33 | 2 | 2 |
