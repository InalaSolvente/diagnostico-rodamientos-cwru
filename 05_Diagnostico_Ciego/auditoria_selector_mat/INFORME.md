# Causa raíz de las copias: selección incorrecta de variables MAT

## Confirmación

Algunos archivos oficiales de CWRU contienen variables de más de un registro.
El código tomaba la primera variable cuyo nombre terminaba en `_DE_time`.
En `Normal_2.mat` eso seleccionaba X098, aunque también está X099. En
`IR021_3.mat` se seleccionaba X215 en lugar de X217. La RPM también se buscaba
por el primer sufijo RPM, sin exigir que perteneciera al registro seleccionado.

Las descargas de los registros oficiales 99 y 217 confirmaron que las variables
correctas ya estaban presentes en los archivos históricos. No hacía falta
inventar señales ni escoger otras por su resultado diagnóstico.

Fuentes primarias:

- https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data

## Corrección aplicada

- `F.clave_canal` exige el ID explícito cuando el MAT es ambiguo.
- `F.rpm_registro` toma RPM del mismo ID; si falta, usa únicamente la nominal
  declarada, nunca la de otro registro incluido en el archivo.
- Los IDs internos también pueden repetirse entre archivos con señales distintas.
  El cargador central exige la ruta para desambiguarlos; no elige la primera fila.
- Se corrigieron los IDs de esas dos filas en `indice_canal_verificado.csv`.
  Las rutas lógicas y las señales históricas se conservaron. `F.leer_indice`
  utiliza ese índice verificado cuando está disponible.
- Los cargadores de diagnóstico ciego se migraron a estos selectores.
- Se verificaron 133 señales distintas y ausencia de copias entre ambas
  particiones existentes. Es una corrección respecto del resultado de 131
  señales únicas obtenido con el selector anterior, no evidencia de que los
  archivos históricos hubieran sido reemplazados.
- Se versionó el extractor para invalidar pesos y cachés anteriores. El
  exportador rechaza rasgos sin la versión de datos corregida.

Los modelos guardados no se borraron. No deben presentarse los 119/129 ni los
78/105 anteriores como resultados de esta base corregida: requieren nuevas
extracciones, entrenamiento y evaluación con procedencia consistente.

## Integridad de experimentos anteriores

En `codigo_anterior/` se conservó el código previo a la migración. Sus datos y
sus métricas son resultados históricos sobre la selección antigua. Los pilotos
de mejoras realizados antes de descubrir este fallo también pertenecen a esa
etapa y no acreditan el objetivo del modelo corregido.

La preparación nueva usa únicamente los 52 registros con falla de desarrollo
y sus dos normales. La reserva no se usa para elegir el próximo cambio. Siguen
pendientes la validación de falsas alarmas, la incertidumbre de los pocos normales
y las limitaciones de independencia entre condiciones nominales del CWRU.

La meta de al menos 90/105 sin retroceso y muchas más mejoras que empeoramientos
sigue sin alcanzarse. No se cambió la regla ND/PD/D ni se entregó la frecuencia
de falla real al predictor para resolver este defecto.

La extracción inicial v2 se realizó con la versión del lector conservada en
`codigo_anterior/Fuentes__extraccion_v2.py`. Después solo se añadió la desambiguación
de la API histórica por archivo; los selectores usados en esa extracción no cambiaron.

Reproducción: `python 05_Diagnostico_Ciego/test_selector_mat.py`.
