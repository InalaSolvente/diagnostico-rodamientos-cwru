# Doble banda: kurtogram y filtro de entropía 25%

105 registros. Dos bandas distintas: 76.

D+P cuenta detecciones compatibles con la falla esperada, no precisión validada.
La corrección divide p=0,01 entre el número de bandas distintas; no corrige las tres armónicas ni la selección adaptativa de bandas.
Los señuelos son controles fuera de frecuencias objetivo, no falsos positivos confirmados.

| Método | D (%) | P (%) | N (%) | D en señuelos (%) | J_D = TPR_D - FPR_D | D+P (%) | D+P en señuelos (%) | J_DP | D en cat. S&R Y (%) | D en cat. S&R P (%) | D en cat. S&R N (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kurtogram (Tarea 1) | 41.9 | 31.43 | 26.67 | 0.0 | 0.42 | 73.33 | 46.19 | 0.27 | 61.43 | 5.88 | 0.0 |
| Entropygram: conservar 25% | 45.71 | 43.81 | 10.48 | 0.95 | 0.45 | 89.52 | 64.76 | 0.25 | 68.57 | 0.0 | 0.0 |
| Doble banda sin control | 51.43 | 39.05 | 9.52 | 0.95 | 0.5 | 90.48 | 70.95 | 0.2 | 75.71 | 5.88 | 0.0 |
| Doble banda con control | 51.43 | 39.05 | 9.52 | 0.95 | 0.5 | 90.48 | 69.52 | 0.21 | 75.71 | 5.88 | 0.0 |
