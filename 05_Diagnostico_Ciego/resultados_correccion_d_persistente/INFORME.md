# Corrección por una única familia D persistente

Se comparan IR/OR/B usando exclusivamente la primera mitad. Se cambia el tipo solo si una única familia cumple D en los tres subsegmentos de selección. La banda y la familia se fijan antes de verificar la segunda mitad.
BSF y 2BSF son dos representaciones de B; no se elige una familia conociendo el defecto verdadero.

Base: {'total': 105, 'mejoran': 8, 'iguales': 70, 'empeoran': 7, 'tipo_incorrecto': 20, 'sin_retroceso': 78}
Corrección: {'total': 105, 'mejoran': 9, 'iguales': 72, 'empeoran': 7, 'tipo_incorrecto': 17, 'sin_retroceso': 81}
Recupera 6 casos que antes retrocedían; pierde 3 casos que antes no retrocedían. Neto: +3.
Se recalcularon desde MAT todas las alternativas y la verificación de los 11 registros corregidos, sin usar caché para esas comprobaciones.

El predictor público solo recibe señal, fs y catálogo completo. Se ejecutó dos veces sobre una señal real, reemplazando por ruido toda su segunda mitad: tipo, selección y alternativas se conservaron exactamente, mientras la verificación cambió.
Ese predictor usa los pesos congelados para señales nuevas; no sustituye las predicciones fuera de condición usadas para puntuar los 133 registros.

## Límites

El objetivo de al menos 90/105 sin retroceso y muchas más mejoras que empeoramientos no está alcanzado.
Esta regla fue propuesta después de inspeccionar errores del dataset: la comparación es exploratoria y no equivale a una validación externa o a una prueba ciega de selección de modelos.
Persistencia temporal y grado D no garantizan acertar el tipo. Las nuevas equivocaciones muestran que otra familia puede presentar picos compatibles y persistentes.
Se preservan las cargas, sensores y muestreos de cada condición fuera del entrenamiento. La posición OR sigue formando parte del grupo; la independencia por rodamiento físico no está demostrada.
El modelo histórico no fue reemplazado. El ranker compartido de armónicas y el clasificador basado solo en evidencia de selección empeoraron y no se adoptan.

Reproducir: `python 05_Diagnostico_Ciego/verificar_correccion_d_persistente.py`.
