# Ejecución completa: modelo por frecuencia de muestreo

48 kHz: las 28 señales se ejecutaron con el nuevo modelo congelado y se calificaron con la regla del usuario.
Reserva: 14 archivos ejecutados; dos excluidos por contener señales idénticas a entrenamiento. Quedan 12 evaluables.
Desarrollo con falla: se usan sus 12 predicciones fuera de condición, en lugar de aciertos de entrenamiento.
Normales de desarrollo: se ejecuta el modelo final, pero se excluyen de validación por haber entrenado con ellos.
La reserva comparte condiciones con entrenamiento y ya se había visto históricamente; no es validación externa.
No hay ejemplos de pista externa en el modelo de 48k. No se calcula no-retroceso del paper para estos archivos sin referencia transcrita.
12 kHz: se conservan los 105 resultados ya calculados del modelo de 12k; los pesos y el criterio no cambiaron.
El consolidado cubre 133 archivos: 129 evaluables, dos normales de entrenamiento y dos copias de señales de entrenamiento sin puntuar. Mezcla protocolos; no es validación independiente.

| protocolo | total | evaluados | correctos | porcentaje | D | PD | ND |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reserva_modelo_congelado | 12 | 12 | 11 | 91.67 | 3 | 7 | 2 |
| desarrollo_fuera_de_condicion | 12 | 12 | 10 | 83.33 | 6 | 6 | 0 |
| reserva_copia_entrenamiento_no_validacion | 2 | 0 | None | None | 0 | 0 | 0 |
| entrenamiento_normal_sin_validacion | 2 | 0 | None | None | 0 | 0 | 0 |

Reserva con falla: 10/11 tipos correctos.
Normales de reserva: 1/1 clasificaciones correctas.
