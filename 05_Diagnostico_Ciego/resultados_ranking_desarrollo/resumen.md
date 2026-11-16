# Ordenación de hipótesis: desarrollo

54 señales; 16 grupos por condición, uniendo cargas y sensores de la misma condición.
Ridge alpha=1, sin búsqueda de hiperparámetros. Estandarización y ajuste solo en grupos de entrenamiento.
Características compartidas entre tipos: fuerza, armónicas, dominancia, respaldo, exclusividad y contraste.
La exclusividad aproxima solapamientos con frecuencias teóricas significativas: la caché no contiene posiciones exactas de todos los picos.
Sin frecuencia esperada ni identidad del tipo como entrada al puntaje. Las etiquetas se usan solo para aprender pesos y evaluar.
Abstención fija: margen menor que 0,1 unidades del score; el score NO es una probabilidad.
Sin evidencia y ambiguo no significan sano. Se informa cobertura y aciertos condicionados a emitir diagnóstico.
Resultados internos sobre desarrollo ya explorado, no validación externa. Reserva excluida.
Los pesos ajustados en todo desarrollo no son los usados para puntuar el grupo excluido.

| metodo | aciertos | fallas | top1_pct | top3 | emitidos_fallas | cobertura_pct | acierto_emitidos_pct | ambiguos | alertas_sanos | sanos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| referencia | 31 | 52 | 59.62 | 45 | 52 | 100.0 | 59.62 | 0 | 2 | 2 |
| coherencia | 30 | 52 | 57.69 | 44 | 52 | 100.0 | 57.69 | 0 | 2 | 2 |
| aprendido_oof | 31 | 52 | 59.62 | 45 | 52 | 100.0 | 59.62 | 0 | 2 | 2 |
| aprendido_ambiguo | 23 | 52 | 44.23 | 45 | 35 | 67.31 | 65.71 | 17 | 2 | 2 |
