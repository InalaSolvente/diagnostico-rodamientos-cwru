# Kurtogram vs. kurtogram + entropía

105 registros; dos frecuencias señuelo por registro.

Temporal: mediana de K y H en 5 ventanas; conservar 75% por H y maximizar K.
D: diagnosticable; P: parcial; N: no diagnosticable.
J_D = proporción D en fd − proporción D en señuelos. Mayor es mejor.

| Método | D (%) | P (%) | N (%) | D en señuelos (%) | J_D = TPR_D - FPR_D | D+P (%) | D+P en señuelos (%) | J_DP | D en cat. S&R Y (%) | D en cat. S&R P (%) | D en cat. S&R N (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kurtogram (Tarea 1) | 41.9 | 31.43 | 26.67 | 0.0 | 0.42 | 73.33 | 46.19 | 0.27 | 61.43 | 5.88 | 0.0 |
| Entropygram: conservar 75% | 52.38 | 23.81 | 23.81 | 0.0 | 0.52 | 76.19 | 54.76 | 0.21 | 74.29 | 11.76 | 5.56 |
| Filtro 75% + mediana temporal | 47.62 | 27.62 | 24.76 | 0.0 | 0.48 | 75.24 | 56.67 | 0.19 | 70.0 | 5.88 | 0.0 |
