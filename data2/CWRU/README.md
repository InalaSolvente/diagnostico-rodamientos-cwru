# Datos CWRU

Fuente original: Case Western Reserve University Bearing Data Center
(https://engineering.case.edu/bearingdatacenter). Esta copia se descargó del repositorio público
`XiongMeijing/CWRU-1` (GitHub), que conserva los `.mat` originales con otros nombres de archivo.

- `12k_DE/`, `12k_FE/`: registros a 12 kHz con falla en el rodamiento Drive End (SKF 6205) o Fan End (SKF 6203).
  Variables: `XNNN_DE_time`, `XNNN_FE_time`, `XNNN_BA_time`, `XNNNRPM`.
- `48k_DE/`: falla Drive End a 48 kHz.
- `Normal/`: línea base sin falla (48 kHz, según Smith & Randall 2015, Tabla B1).

Nombre de archivo: `<zona><tamaño>[@posición]_<carga>.mat`
 - zona: `IR` pista interna, `OR` pista externa (posición @3, @6 o @12 h), `B` bola
 - tamaño: 007 = 0.007 in, 014, 021, 028
 - carga: 0–3 hp

`indice_cwru.csv` asocia cada archivo con su **número de registro CWRU** (el que usa el profesor, por ejemplo 282).
Notas del índice:
 - Los archivos 0.028 in (`IR028_*`, `B028_*`) corresponden a los registros oficiales 3001–3008. No traen RPM, así que se usa la nominal por carga.
 - En el repositorio de origen, `48k_DE/IR021_3` y `Normal/Normal_2` repiten el número de variable de otro archivo. Úsalos con cuidado.
