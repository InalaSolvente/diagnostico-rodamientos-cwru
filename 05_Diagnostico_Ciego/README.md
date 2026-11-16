# Inferencia ciega desde la señal

## Corrección posterior: MAT con múltiples registros

La causa raíz de las copias se encontró en el cargador: tomaba X098 en lugar de
X099 y X215 en lugar de X217 dentro de archivos que contienen ambos registros.
La selección explícita y las RPM del mismo registro recuperan **133 señales únicas**
sin reemplazar los originales. Consulte `auditoria_selector_mat/INFORME.md`.
Los pesos/cachés previos quedaron invalidados; sus métricas son históricas.
La base nueva de desarrollo se genera con `preparar_desarrollo_verificado.py`
en `resultados_desarrollo_verificado_v2/`. No se reutilizan las predicciones antiguas
para acreditar el objetivo mínimo de 90/105.

## Advertencia de auditoría — 15 de noviembre de 2026

Se encontraron copias exactas de señales a ambos lados de entrenamiento/reserva:
`Normal_1`/`Normal_2` y `48k_DE/IR021_2`/`IR021_3`. Los 133 archivos contienen
131 señales únicas del canal seleccionado. Las dos copias de reserva y los dos
normales de entrenamiento no se puntúan: quedan **119/129 tipos correctos**, no
121/131. El total sigue siendo descriptivo de protocolos mixtos, sin validación
externa. Los entrenadores rechazan ahora las particiones con señales copiadas;
hay que establecer otra partición antes de volver a entrenar/exportar.

Informe y evidencias en `auditoria_modelos_actuales/`. Los resultados y figuras
anteriores son históricos. El antiguo 99/105 contenía tres aciertos artificiales
y queda retirado; la comparación actual de grados da 90/105 sin retroceso,
sin homologación de criterios con el paper ni verificación temporal independiente.

```powershell
python 05_Diagnostico_Ciego/auditar_modelos_actuales.py
python 05_Diagnostico_Ciego/recontar_auditoria.py
python 05_Diagnostico_Ciego/integridad_senales.py
```

## Modelo separado entrenado en 48 kHz

```powershell
python 05_Diagnostico_Ciego/entrenar_48k.py
python 05_Diagnostico_Ciego/test_modelo_48k.py
python 05_Diagnostico_Ciego/evaluar_modelos_todos.py
python 05_Diagnostico_Ciego/test_ejecucion_modelos.py
```

```python
from predictor_48k import predecir
resultado = predecir(x, 48000, catalogo_completo)
```

Mismo LDA regularizado y rasgos alineados a L6/375 Hz (31 nodos entre 0–6 kHz),
con pesos nuevos exclusivos de 48k. Mitad fija de 28 señales: dos cargas por
condición de falla y los dos normales de desarrollo previos, 14 en total.
Reserva las otras 14. Reutilizar sus observaciones históricas impide llamarla
reserva externa intacta. No hay selección de hiperparámetros con sus resultados.
Valida las 12 fallas de desarrollo excluyendo una condición completa cada vez;
los dos normales se mantienen solo en entrenamiento para no eliminar la clase N.
Su desempeño sano/falla no queda validado. No hay datos de pista externa ni jaula:
solo cubre B, IR y N. Resultados y pesos separados en `resultados_modelo_48k/`.
Los rasgos de entrenamiento e inferencia usan ahora la misma geometría física.
La ejecución completa del nuevo modelo guarda las 28 salidas finales y distingue
14 reservas, 12 fallas de desarrollo fuera de condición y dos normales usados en
entrenamiento que no se puntúan como validación. `resultados_modelos_todos/`
consolida también los 105 resultados de 12k con su propio modelo, sin recalcular
ni mezclar pesos. Las 133 señales están cubiertas, pero no representan una prueba
independiente única ni se suman aciertos de entrenamiento al porcentaje validado.

## Prueba L adaptado al ancho físico

```powershell
python 05_Diagnostico_Ciego/test_alinear_bandas.py
python 05_Diagnostico_Ciego/ensayar_L_alineado.py
```

12 kHz/L4 y 48 kHz/L6 dan bandas finales de 375 Hz. Para conservar los
pesos del modelo de 31 nodos, la prueba a 48k toma solo el subárbol 0–6 kHz,
niveles globales 2–6: exactamente los intervalos físicos del árbol de 12k.
No se rellenan ni interpolan las 1.953 características ni se remuestrea.
No aprovecha las bandas adicionales de 6–24 kHz; usar los 127 nodos exigiría
una nueva representación y entrenamiento. Se conservan RMS global y
definiciones de entropía, por lo que no iguala todos los efectos de fs.
Prueba los 24 archivos con falla a 48k y cuatro normales; 12k es idéntico
por construcción y verificación numérica. Pesos y umbrales quedan congelados.
Salidas separadas: `resultados_L_alineado/`.

## Regla explícita ND/PD/D del usuario

```powershell
python 05_Diagnostico_Ciego/test_criterio_usuario.py
python 05_Diagnostico_Ciego/evaluar_criterio_usuario.py
python 05_Diagnostico_Ciego/evaluar_criterio_usuario.py --todos
```

D exige la primera armónica identificada, al menos dos en total y ningún
pico más fuerte a su izquierda. Un pico más alto a la derecha no impide D.
PD exige una o más armónicas identificadas y un pico previo mayor; puede
faltar la primera. En ese caso, el pico previo debe superar la primera
armónica realmente identificada, no el ruido en la fundamental. ND es el resto. Se excluye DC y se mantiene tolerancia
±2%; «identificada» significa pico local significativo contra fondo local.
Se exploran hasta seis armónicas en los 31 nodos y las familias del catálogo,
con p nominal corregida por 31×5×6. La izquierda excluye la ventana de la
primera; los empates no son «mayor». BSF/2BSF se documentan explícitamente.

La implementación nueva está en `criterio_usuario.py`; los procedimientos
históricos conservan su regla mediante `criterio="historico"`. El evaluador
nuevo llama `calificar(..., criterio="usuario")` sobre las mismas 54
predicciones de desarrollo ya fijadas, sin cambiar el modelo ni cargar otros
archivos. Guarda banda, familia, picos y evidencia en `evidencia.json`, junto
con parámetros exactos, CSV y resumen en `resultados_criterio_usuario_desarrollo/`.
Una salida PD/D de tipo equivocado se muestra como error, nunca como ND
fabricado usando la etiqueta real. Una salida ND genuina sí puede conservar
ND del paper. La rúbrica nueva se declara propia, no idéntica a la del paper.
`--todos` ejecuta los 133 archivos del índice: 105 con falla a 12k, 24 con
falla a 48k y cuatro normales. Las referencias del paper transcritas cubren
solo los primeros 105; normales y 48k se informan aparte y no se suman como
ND automáticos. Resultados en `resultados_criterio_usuario_total/`.
Se reutilizan los tipos ya fijados cuando corresponden al mismo modelo;
los tipos restantes se calculan con el modelo congelado. Los grados se
recalculan desde las señales bajo la nueva regla, sin reentrenamiento.

## No retroceso frente al paper

Evaluación total congelada: `python 05_Diagnostico_Ciego/evaluar_paper_total.py`.
Resultados en `resultados_paper_total/`: 99/105 sin retroceso (94,29%),
98/105 tipos correctos. Desarrollo fuera de grupo: 47/52; reserva con modelo
congelado: 52/53. El total mezcla protocolos y no es una prueba independiente.
Los 53 registros con falla de reserva ya se usaron para esta evaluación final:
no deben describirse como reserva sin observar en experimentos posteriores.
La meta interpretada es 95 casos de 105, no 95%.

```powershell
python 05_Diagnostico_Ciego/test_no_retroceso.py
python 05_Diagnostico_Ciego/evaluar_no_retroceso.py
```

Nueva métrica ordinal: ND→ND/PD/D, PD→PD/D y D→D cuentan como éxito.
Se usa la transcripción S&R del proyecto (Y=D/P=PD/N=ND). La predicción
de tipo es la verificada fuera de grupo; se califica su evidencia espectral
en los 31 nodos, sin consultar el tipo real ni la categoría del paper.
En evaluación, un tipo incorrecto se degrada a ND. Las filas originalmente
ND cuentan siempre como no retroceso: por eso el porcentaje no representa
precisión de diagnóstico. Se informa el tipo correcto en paralelo.
El grado es operativo según el criterio existente, no reproducción exacta
del método original de S&R. El evaluador de desarrollo guarda salidas
en `resultados_no_retroceso_desarrollo/`.

## Clasificador espectral supervisado

```python
from predictor_espectral import predecir
resultado = predecir(x, fs, frecuencias_fabricante=catalogo_completo)
```

Nueva alternativa: evidencia de seis armónicas de todas las familias en cada
uno de los 31 nodos (L=4 fijo), más energía, K y H por nodo. El árbol completo
evita perder evidencia al conservar solo dos bandas. El extractor no recibe
el tipo de defecto, fd real, ruta de archivo, RPM ni etiquetas. El catálogo
completo en Hz se prepara igual que en el resto del proyecto.

Se ensayan seis bancos × cuatro clasificadores instalados. Cada predicción
fuera de grupo excluye todas las cargas y sensores de esa condición; escalado
y ajuste usan solo entrenamiento. Se elige el ganador sobre desarrollo:
esto introduce selección sobre desarrollo y no equivale a prueba externa.
Los 55 archivos de reserva no se ejecutan. Se conserva el modelo heurístico
anterior para comparación; esta API nueva es la alternativa supervisada.

```powershell
python 05_Diagnostico_Ciego/test_espectral.py
python 05_Diagnostico_Ciego/ensayar_espectral.py
python 05_Diagnostico_Ciego/verificar_y_exportar_espectral.py
```

Resultados, caché de rasgos, predicciones fuera de grupo, auditoría y coeficientes
numéricos sin pickle: `resultados_espectrales_desarrollo/`. La exportación final
se ajusta en todo desarrollo; las métricas corresponden a modelos entrenados
sin el grupo evaluado. La inferencia verifica la firma del extractor.
Se requiere scikit-learn para entrenar/verificar, ya instalado; la inferencia
del modelo numérico usa NumPy. Scores LDA no son probabilidades calibradas.
No hay entrenamiento de fallas de jaula y solo hay dos normales en desarrollo;
la capacidad para declarar salud sigue sin validarse.

## Ensayo de ordenación y abstención

```powershell
python 05_Diagnostico_Ciego/test_ranking.py
python 05_Diagnostico_Ciego/ranking_catalogo.py
```

Reutiliza exclusivamente las 54 evidencias de desarrollo, con p=0,01 sin
corrección, y compara la referencia, un puntaje manual de coherencia y un
puntaje ridge compartido entre tipos. No recalcula FFT ni consulta reserva.
El puntaje aprendido usa fuerza, armónicas, dominancia, apoyo de las bandas,
solapamientos teóricos con otras familias y contraste con la competencia.
No utiliza identidad del tipo ni frecuencia esperada como característica.
La caché solo permite aproximar solapamientos mediante armónicas teóricas.

Validación: deja fuera una condición completa cada vez, uniendo sus cargas y
sensores; normaliza y ajusta solo en los demás grupos. Alpha=1 fijo, sin buscar
pesos contra los resultados del grupo excluido. Las etiquetas sí se usan en
el entrenamiento, como corresponde a un puntaje supervisado.
La abstención usa margen fijo 0,1 entre los dos primeros scores (no probabilidades).
Se informan aciertos, cobertura, precisión entre diagnósticos emitidos y alertas
en los dos normales. Ninguna abstención confirma salud. Solo desarrollo ya
explorado: aún no representa generalización externa. Las salidas y la auditoría
quedan en `resultados_ranking_desarrollo/`. Los pesos finales son experimentales;
este ensayo no cambia automáticamente el modelo predeterminado.

## Regla experimental actual: dominante con control

El valor predeterminado es `estrategia="dominante"`: admite una armónica
significativa, prioriza los casos D (pico dominante en el rango definido por
el criterio) y luego la suma de márgenes sobre el umbral. Sigue comparando
las cinco hipótesis del catálogo y corrigiendo p/(5×bandas×3). No utiliza
el tipo real ni fd esperada para seleccionar la hipótesis.

Comparación fija en desarrollo: conservador 24/52 top1 y 37/52 top3;
dominante 31/52 y 44/52. Alertas en normales: pasan de 0/2 a 2/2.
Esto mejora identificación de tipo en fallas, pero no demuestra una mejora
global ni especificidad suficiente. Son resultados de desarrollo, no de
validación externa. No equivalen al antiguo D+P con objetivo conocido.

```powershell
python 05_Diagnostico_Ciego/comparar_estrategias.py
python 05_Diagnostico_Ciego/evaluar_catalogo.py --estrategia dominante
python 05_Diagnostico_Ciego/evaluar_catalogo.py --estrategia conservador
```

El primer comando calcula una vez las bandas y compara cuatro reglas. Guarda
evidencia para todas las familias en `resultados_estrategias_desarrollo/` y
verifica la reproducción de la referencia previa. El evaluador separa las
salidas por estrategia para no sobrescribir resultados históricos.

## Flujo actual: catálogo completo y media muestra

```python
catalogo = {"BPFI": 160., "BPFO": 106., "BSF": 70., "FTF": 12.}
resultado = inferir(x, fs, frecuencias_fabricante=catalogo)
```

Exige las cuatro frecuencias; compara cinco hipótesis incluyendo 2BSF, sin
indicar el defecto real. Agrupa BSF/2BSF en elemento rodante. La regla dominante
admite una armónica; la conservadora exige dos. Corrige p/(5×bandas×3), excepto
en la variante explícita sin_control. Devuelve hasta tres tipos
distintos (IR, OR, B, FTF); `sin_evidencia` no equivale a rodamiento sano.

`evaluar_catalogo.py` y el comando `evaluar_ciego.py` usan solo la mitad por
carpeta, semilla 2026, fijada en `particion_desarrollo.csv`. Rechazan registros
de reserva. No usan el tipo de defecto para dividir ni construir el catálogo.
La clave es la ruta de archivo: el índice contiene dos normales con número 98.
El catálogo se calcula con todos los factores del rodamiento y RPM. Estos
factores proceden de la referencia del proyecto, no de ficha de fabricante
verificada. El modelo recibe frecuencias en Hz, no RPM ni etiquetas.

Resultados separados: `resultados_catalogo_desarrollo/`. Se guardan las
predicciones antes de evaluar el tipo real; se informa tipo correcto top1,
top3 y alertas en normales. Top3 no es precisión de diagnóstico único.
La reserva ya se examinó en experimentos anteriores: limita futuras pruebas,
pero no constituye validación independiente retrospectiva.

## Búsqueda libre anterior (sin catálogo)

```python
from modelo_ciego import inferir
resultado = inferir(x, fs)
```

Entrada: x, fs y parámetros generales (L=4, fmin=10 Hz, fmax=1000 Hz,
p=0,01). No acepta fd, RPM, tipo de defecto ni sensor. La profundidad
es fija: no usa `nivel_maximo(fs, fd)` ni las bandas de los CSV antiguos.
Selecciona dos nodos: máxima K y máxima K entre el 25% con menor H.
Busca máximos de las envolventes y propone fundamentales f_pico/h,
h=1,2,3. Exige dos armónicas significativas y ordena por cantidad de
armónicas y margen sobre el umbral. Deduplica las dos bandas y candidatos
cercanos (2%). Devuelve hasta tres candidatos, o `sin_evidencia`.
`sin_evidencia` no significa rodamiento sano; periodicidad no significa falla.
No identifica tipo de defecto sin información adicional del rodamiento.

El umbral nominal divide p por candidatos × bandas × tres armónicas.
No es una garantía de falsas alarmas: candidatos/bandas se eligen con los
mismos datos y el fondo no tiene por qué seguir el modelo exponencial.
Es una heurística nueva sin ajuste con etiquetas; hay ambigüedad entre
fundamentales, subarmónicas y modulaciones ajenas a la falla. El rango y L
deben fijarse antes de evaluar y no elegirse según el defecto del registro.

```powershell
python 05_Diagnostico_Ciego/test_ciego.py
python 05_Diagnostico_Ciego/evaluar_ciego.py
```

El evaluador histórico cargaba señales 12k y normales y ejecutaba inferencia solo con x y fs,
y guarda predicciones antes de consultar las frecuencias de referencia.
Después mide coincidencia top1 y top3 con fd, 2fd o 3fd, tolerancia ±2%.
Sanos: informa periodicidades detectadas, no falsos positivos confirmados.
Los canales DE/FE del benchmark son los usados en los experimentos previos;
los registros sanos tienen otra frecuencia de muestreo. No hay todavía
validación externa ni estimación calibrada de precisión de diagnóstico.
Admite números de registro y `--salida` para ensayos separados.

Las carpetas 01–04 conservan experimentos históricos con frecuencia objetivo;
sus tasas D+P no son exactitud de inferencia ciega ni comparables con top1/top3.
