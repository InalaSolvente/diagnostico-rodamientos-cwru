# Estrategias sin frecuencia esperada: desarrollo

Mismas 54 señales y mismas bandas L=4. Reserva excluida.
Conservador: dos armónicas, prioridad a su cantidad, corrección por 5×bandas×3.
Sensible: una armónica, orden por suma de márgenes sobre umbral, misma corrección.
Dominante: una armónica, prioriza categoría D y luego margen, misma corrección.
Sin control: regla dominante con p=0,01 por prueba, sin corrección de hipótesis.
La presencia en top3 no es diagnóstico único correcto. Dos sanos no bastan para validar especificidad.
Se eligen variantes con datos de desarrollo; no son resultados de prueba independiente.

| estrategia | fallas | top1 | top3 | top1_pct | top3_pct | alertas_sanos | sanos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conservador | 52 | 24 | 37 | 46.15 | 71.15 | 0 | 2 |
| sensible | 52 | 25 | 44 | 48.08 | 84.62 | 2 | 2 |
| dominante | 52 | 31 | 44 | 59.62 | 84.62 | 2 | 2 |
| sin_control | 52 | 31 | 45 | 59.62 | 86.54 | 2 | 2 |
