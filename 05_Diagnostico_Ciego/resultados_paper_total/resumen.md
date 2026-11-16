# No retroceso frente al paper: evaluación total

Definición: ND→ND/PD/D, PD→PD/D, D→D. Tipo incorrecto se evalúa como ND.
Las filas ND de referencia cuentan automáticamente; esto no es precisión diagnóstica.
Desarrollo: predicciones fuera de grupo. Reserva: modelo congelado entrenado en desarrollo.
El total mezcla ambos protocolos y no sustituye el resultado separado de reserva.
La reserva fue observada históricamente y no es validación externa nueva.
No se usó la frecuencia de falla esperada en inferencia ni al calificar evidencia; sí se comparó después el tipo real.
Referencia S&R transcrita en el proyecto; el grado es operativo, no una reproducción de su algoritmo.

| conjunto | total | sin_retroceso | porcentaje | tipo_correcto |
| --- | --- | --- | --- | --- |
| desarrollo_oof | 52 | 47 | 90.38 | 45 |
| reserva_congelado | 53 | 52 | 98.11 | 53 |
| total_mixto | 105 | 99 | 94.29 | 98 |
