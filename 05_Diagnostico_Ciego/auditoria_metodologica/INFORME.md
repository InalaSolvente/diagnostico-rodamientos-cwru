# Auditoría metodológica del predictor y de los 99/105 casos

Fecha: 15 de noviembre de 2026. Alcance: código, señales originales de desarrollo,
rasgos, pesos exportados, partición, predicciones y evaluación ordinal.

## Dictamen

El predictor supervisado es un experimento reproducible y no encontré una entrada
directa de la frecuencia real ni del tipo real de falla al predecir. Sin embargo,
**99/105 no demuestra 99 diagnósticos válidos ni superioridad sobre el paper**.
Hay tres éxitos artificiales en la métrica, categorías no homologadas y una
validación que no permite interpretar el total como generalización independiente.
La afirmación anterior de haber satisfecho la meta debe limitarse al contador
operativo implementado; fue demasiado fuerte como conclusión metodológica.

No se modificaron los pesos ni las predicciones históricas durante esta auditoría.
Los recuentos alternativos están separados de los resultados originales.

## 1. Crítico: el evaluador fabrica ND después de conocer la verdad

En `evaluar_no_retroceso.py:22`, `grado_evaluable` se convierte a ND si el tipo
predicho es incorrecto. La comparación posterior declara éxito para cualquier
referencia ND. Esto no significa que el modelo haya acertado una salida ND.

La auditoría verificó que el modelo/calificador emitió **87 D, 18 PD y ningún ND**
en los 105 registros. Los ND de la figura se asignaron después de comparar con
el tipo real. Tres casos incorrectos acabaron contando como aciertos:

| Registro | Tipo real | Tipo predicho | Paper | Por qué contó |
|---|---|---|---|---|
| 57 | IR | B | ND | El evaluador convirtió el diagnóstico incorrecto a ND |
| 58 | IR | B | ND | Igual |
| 200 | OR | B | ND | Igual |

Eso contradice la interpretación «atinarle a que era ND»: una salida de falla
equivocada no es una abstención correcta. Una alternativa coherente con la salida
emitida cuenta ND→ND si realmente se emitió ND, o conservación/mejora si el tipo
es correcto. Aplicada a las mismas predicciones produce **96/105**, no 99/105.
El tipo de falla por sí solo fue correcto en 98/105, un tercer número distinto.

No se ha corregido silenciosamente la métrica original. El recuento alternativo
está en `recuento_estricto.csv`; su definición también necesita fijarse antes de
los siguientes ensayos.

## 2. Crítico: D/PD/ND no son el mismo criterio que Y/P/N del paper

La [publicación original, sección 6.2 y tabla 4](https://www.researchgate.net/publication/276248554_Rolling_Element_Bearing_Diagnostics_Using_the_Case_Western_Reserve_University_Data_A_Benchmark_Study)
considera claridad y características diagnósticas en tiempo y frecuencia,
incluyendo situaciones clásicas, no clásicas, parciales y no diagnosticables.
La implementación usa significancia de picos y dominancia dentro de un rango
espectral. No comprueba esa misma definición de claridad ni las estructuras
temporales y de modulación necesarias para una homologación directa.

Además, el paper usa el espectro de la envolvente cuadrada; el código calcula
potencia espectral de la envolvente de amplitud centrada. No son la misma
transformación. La diferencia no invalida el método nuevo, pero sí impide
presentar las letras como equivalentes sin validación adicional.

En `calificar`, se busca el mejor grado en 31 nodos y se retorna al encontrar D.
Es una regla de optimización interna, no un juez externo de diagnósticos. No se
puede demostrar una mejora sobre la clasificación del paper solo cambiando el
reglamento de lo que cuenta como D. La referencia local también colapsa los
subtipos y toma el mejor M1–M3 por sensor; debe declararse así, no como una única
predicción original. No se certificó una transcripción celda por celda de los
105 registros; se contrastaron las definiciones y casos de referencia, incluido
el mapeo por tipo de los registros 0,028 pulgadas.

## 3. Alto: selección del ganador sobre los mismos resultados reportados

`ensayar_espectral.py` evalúa 6 bancos de rasgos × 4 clasificadores y los ordena
por su rendimiento. `verificar_y_exportar_espectral.py:16` elige `res.iloc[0]` y
repite sus mismos grupos. Cada ajuste de ese ganador excluye el grupo evaluado,
lo cual es correcto; **la elección del ganador no lo excluye**.

La repetición demuestra reproducibilidad, no corrige el sesgo de selección entre
24 alternativas. El 86,54% de desarrollo no es una estimación imparcial de un
modelo elegido sin observar esos resultados. Buscar variantes hasta superar
un umbral refuerza el riesgo de selección sobre desarrollo.

Corrección: validación anidada por grupos. Cada grupo exterior debe permanecer
fuera de la selección del banco, clasificador e hiperparámetros. Todo el proceso
de elegir el modelo se realiza dentro de sus grupos de entrenamiento.

## 4. Alto: la reserva comparte condiciones con entrenamiento

La partición inicial se hizo por archivos, dentro de las carpetas de adquisición.
Los archivos son distintos, pero **49 de los 53 registros con falla de reserva**
pertenecen a condiciones nominales presentes en desarrollo. Solo cuatro registros
de `12k_FE/OR014@3_*` pertenecen a una condición ausente de desarrollo.

Esto no demuestra duplicación de muestras ni que se entrenara directamente con
los archivos de reserva. Sí significa que el 53/53 de tipo correcto en reserva
evalúa principalmente otras cargas de condiciones ya conocidas, mientras que el
45/52 de desarrollo deja fuera condiciones completas. No son dificultades ni
protocolos equivalentes. Las condiciones nominales son una aproximación al grupo
físico de adquisición; no sustituyen IDs verificados de cada rodamiento/instalación.

La discusión de la sección 6.2 del paper también relaciona los resultados con
la instalación del rodamiento. Por tanto, una división aleatoria por archivo
puede favorecer reconocimiento de la condición de montaje en este conjunto.

Además, los archivos de reserva se habían observado en experimentos anteriores
y ahora ya se abrieron para la evaluación final. No existe una reserva intacta
para seguir afinando contra ella.

## 5. Alto: el total mezcla predicciones de modelos y protocolos diferentes

| Parte | Señales | Tipo correcto | Sin retroceso publicado |
|---|---:|---:|---:|
| Desarrollo, ganador evaluado fuera de grupo | 52 | 45 | 47 |
| Reserva, modelo ajustado en todo desarrollo | 53 | 53 | 52 |
| Total mixto | 105 | 98 | 99 |

El total es una descripción de lo ejecutado, no una prueba única independiente
de un único modelo con un único criterio de separación. Tampoco cubre todas las
señales disponibles: son los 105 archivos con falla de las dos carpetas 12 kHz.
Los cuatro normales y los datos 48 kHz con falla no forman parte de esos 105.

## 6. Alto: el resultado no valida sano frente a falla

Los dos normales de desarrollo fueron errores en validación por grupos. Hay un
problema estructural: ambos pertenecen al único grupo `Normal`; al excluirlo,
el entrenamiento contiene solo B, IR y OR. El clasificador no puede predecir una
clase N que no recibió durante ese ajuste. Ese diseño no permite una evaluación
cerrada y fiable de reconocimiento de normales.

Hay también un factor de confusión: los normales están muestreados a 48 kHz y los
archivos con falla usados a 12 kHz. El árbol usa fracciones de Nyquist, por lo que
sus nodos representan regiones físicas diferentes. Esto puede permitir que un
modelo aprenda diferencias de adquisición; no se demostró que esa sea la causa
de sus decisiones actuales. Se necesitan varias adquisiciones normales
independientes y una representación físicamente comparable entre muestreos.

ND del paper significa que esa falla no fue diagnosticable con los métodos
considerados, no que el rodamiento estuviera sano. Usar ND como sustituto de
salud sería otro error.

## 7. Medio-alto: el criterio ordinal tiene éxitos automáticos

Hay 18 referencias ND y todas cuentan como «sin retroceso» bajo la regla publicada.
Tres corresponden a tipos predichos incorrectos. Si se excluyen las filas ND,
quedan **81/87 sin retroceso**, todavía usando grados no homologados.

La métrica es legítima como indicador operativo si se define y etiqueta así;
no es exactitud diagnóstica ni evidencia de que el modelo reconoció esos 18 ND.
Además, si se promovieran arbitrariamente todos los grados emitidos a D sin
cambiar los tipos actuales, el contador subiría a 101/105. No se hizo eso: es
un contraejemplo de por qué maximizar este contador sin un juez independiente
de la calidad puede incentivar promociones artificiales.

## 8. Medio: se asumió que ya sabemos qué componente medir

El evaluador elige el canal con `r.sensor_falla` y construye el catálogo del
rodamiento de ese extremo. Es razonable si el componente y el sensor objetivo
están configurados de antemano. No demuestra un diagnóstico completamente ciego
de una máquina donde todavía no sabemos qué rodamiento está afectado.

Las frecuencias se obtienen de factores cinemáticos almacenados en el proyecto
y de RPM disponibles en la adquisición. No se entrega solo la frecuencia de la
falla real; se entrega el catálogo completo. Pero tampoco se verificó una ficha
de fabricante para cada rodamiento. Es una entrada física adicional que debe
figurar explícitamente en la descripción del método.

## 9. Medio: Bonferroni nominal no prueba una tasa real de falsas alarmas

La corrección divide p por bandas, familias y armónicas. Bonferroni no requiere
independencia entre bandas; el problema es que cada prueba necesita un valor p
calibrado. Aquí se supone un fondo aproximadamente exponencial, se estima la
media local del mismo espectro y se buscan máximos. Su calibración bajo ruido
real, interferencias y selección adaptativa no está demostrada.

Que p sea 0,01 no significa que el sistema tenga 1% de falsas alarmas. Faltan
controles con varias señales sanas, ruido coloreado, impulsos aislados y
modulaciones sin falla. Los señuelos tampoco son negativos confirmados.

## 10. Medio: pasó de regla de bandas a clasificación supervisada

El ganador es LDA con 1.953 características y 54 señales de desarrollo. La
regularización hace posible el ajuste; una dimensionalidad alta no prueba por
sí misma sobreajuste, pero refuerza la necesidad de selección y evaluación
independientes. Se usan etiquetas reales al entrenar, de forma legítima para
un clasificador supervisado. No es el mismo problema que una regla no supervisada
de kurtogram/entropía. No debe presentarse la mejora como atribuible solo al
filtro de entropía.

Las clases entrenadas son B, IR, OR y N. FTF participa como característica, pero
no existen ejemplos de falla de jaula como clase. La salida es forzada entre las
cuatro clases; no hay un detector validado de condiciones desconocidas.

## 11. Medio: trazabilidad y pruebas con alcance limitado

Los hashes actuales y la reproducción de pesos son controles útiles, pero
`ajuste_con_reserva=False` es un valor escrito por el script, no una prueba
independiente de toda la historia de selección. Los artefactos numéricos y sus
CSV tampoco sustituyen un registro de experimentos congelado antes de evaluar.

La caché de extracción usa ruta, tamaño, fecha del archivo y hash del extractor.
No incorpora todos los parámetros del índice/catálogo ni sus dependencias y
versiones. Un cambio de factores cinemáticos sin cambio del extractor puede
reutilizar una caché anterior. En el estado actual la auditoría recalculó los
54 vectores y descartó una discrepancia presente, pero el riesgo futuro existe.

La evaluación total no guarda el nodo y la familia que justifican cada D/PD,
aunque `calificar` los calcula. Eso dificulta inspeccionar las mejoras una a una.
También hay dos normales con ID 98 en el índice: el método nuevo usa rutas
únicas y no los confunde; los flujos históricos por número necesitan cautela.

Varias pruebas fijan recuentos como 99 y umbrales como 80%. Verifican que el código
reproduzca resultados esperados, no que las definiciones sean científicamente
válidas. No deberían usarse como certificación externa de rendimiento.

## Lo que sí pasó la auditoría

- Recalculados los 54 vectores del árbol desde los archivos `.mat`: coinciden
  con la caché. No fue necesario confiar solo en sus CSV.
- Reentrenado LDA exclusivamente con esos 54 vectores: medias, escalas,
  coeficientes, interceptos y clases coinciden con el modelo exportado.
- Las rutas de desarrollo y reserva son disjuntas: no se encontró duplicación
  directa de archivo entre esas dos particiones.
- El escalado se ajusta dentro de cada grupo de entrenamiento en validación.
- La API de inferencia recibe señal, fs y catálogo completo; no recibe etiqueta,
  número de registro, ruta ni frecuencia marcada como correcta.
- El índice físico y las etiquetas no aparecen como características del modelo.
  La ruta sí interviene en la organización de grupos y en el evaluador.
- El hash del modelo quedó intacto durante esta auditoría.

Estos controles no demuestran ausencia de cualquier sesgo imaginable, pero sí
descartan varios atajos directos de implementación en el estado inspeccionado.

## Qué afirmación puede mantenerse

«En este conjunto explorado, el modelo reproduce 45/52 tipos correctos fuera de
grupo en desarrollo. En la reserva por archivos obtuvo 53/53, con fuerte
solapamiento de condiciones. El indicador ordinal implementado dio 99/105;
una variante que no acredita salidas equivocadas como ND da 96/105. No hay
una demostración independiente de 95 diagnósticos clínicamente/mecánicamente
válidos ni de superioridad bajo la definición original del paper».

## Protocolo recomendado antes de otra afirmación de éxito

1. Fijar por separado tipo correcto, sano/falla, abstención y grado ND/PD/D.
   Una salida equivocada no debe convertirse en una abstención correcta a
   posteriori. Conservar siempre la salida original y el fallo, sin ocultarlos
   con un grado derivado.
2. Homologar la clasificación del paper mediante criterios publicados antes
   de evaluar, con inspección ciega por expertos o una rúbrica común validada.
   Guardar banda, familia, armónicas, umbral y espectro de cada diagnóstico.
3. Seleccionar modelos dentro de validación anidada por grupos de adquisición,
   no sobre el mismo resultado exterior que se reporta.
4. Obtener datos independientes o establecer una evaluación final que no se
   reutilice para nuevas decisiones. No volver a llamar «intacta» a esta reserva.
5. Reunir varios grupos normales, conservar presencia de todas las clases
   entrenables por pliegue y controlar la frecuencia de muestreo.
6. Verificar catálogo físico y sensor como configuración disponible en uso
   real. Evaluar ambos extremos si no se conoce el componente afectado.
7. Calibrar umbrales y abstención con controles reales/sintéticos, informar
   errores por clase y cobertura. Los intervalos de incertidumbre deben tener
   en cuenta los grupos, no asumir 105 experimentos independientes.

## Reproducción

```powershell
python 05_Diagnostico_Ciego/auditar_metodo.py
```

Resultados auditables: `hechos_verificados.json` y `recuento_estricto.csv`.
El programa no cambia el modelo ni recalifica los resultados históricos.
