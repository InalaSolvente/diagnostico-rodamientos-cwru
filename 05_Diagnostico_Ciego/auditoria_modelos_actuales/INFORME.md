# Auditoría de los modelos actuales: identificación y calidad diagnóstica

Fecha: 15 de noviembre de 2026. Alcance: los 133 archivos locales, ambos modelos
congelados, sus particiones, el flujo de inferencia y los grados ND/PD/D.

## Dictamen

**Hay filtración demostrada entre entrenamiento y reserva por señales idénticas
con nombres distintos. Además, el grado D no está validado como prueba de falla:
el calificador también encuentra D en las tres señales normales únicas al
examinar hipótesis de falla por separado.** No se puede certificar un modelo
«sin atajos» ni presentar los resultados actuales como validación independiente.

No se encontró que las APIs de inferencia reciban la frecuencia real de falla,
el tipo real, el nombre de archivo ni el grado del paper. Eso descarta ese acceso
directo en el flujo inspeccionado; no descarta aprendizaje de diferencias de
adquisición ni sesgo por la historia de experimentación.

Los pesos no se cambiaron para mejorar los números. Se corrigieron controles
de integridad y evaluación, y se conservaron las predicciones originales como
evidencia histórica. La prueba temporal adicional se informa separadamente.

## 1. Crítico y confirmado: copias de señales atraviesan la partición

Se cargó el canal realmente usado en cada uno de los 133 archivos y se calculó
SHA-256 sobre sus muestras completas, normalizadas únicamente a representación
binaria float64. No se compararon solo rutas, IDs o hashes de los contenedores MAT.

| Señal usada para entrenar | Copia contada como reserva | Consecuencia |
| --- | --- | --- |
| `Normal/Normal_2.mat` | `Normal/Normal_1.mat` | Contamina la reserva de ambos modelos; el consolidado utilizaba la predicción del modelo de 48 kHz |
| `48k_DE/IR021_3.mat` | `48k_DE/IR021_2.mat` | Contamina la reserva de 48 kHz |

Los archivos MAT completos tienen hashes distintos, pero los canales seleccionados
son idénticos muestra por muestra. Hay **131 señales únicas entre 133 archivos**.
Ambas copias de reserva se habían contado como aciertos. El registro IR021_3
con predicción fuera de condición puede mantenerse: en ese pliegue se excluye
el grupo IR021 completo y la copia de reserva no forma parte del ajuste.

Además, las dos copias normales no tienen RPM en el MAT y reciben distintas
RPM nominales del índice: 1772 y 1750. Por tanto, la misma señal recibió dos
catálogos cinemáticos distintos. Es una inconsistencia de metadatos que debe
resolverse desde la fuente; aquí no se inventó una RPM corregida.

Correcciones implementadas:

- `integridad_senales.py` detecta igualdad de muestras aunque cambie el nombre.
- Los entrenadores/exportadores de 12 y 48 kHz rechazan las particiones actuales
  antes de ajustar/exportar pesos. No se reorganizó la reserva buscando un número mejor.
- `evaluar_modelos_todos.py` conserva la salida final de las copias para trazabilidad,
  pero no las puntúa como validación. Excluye también los dos normales de entrenamiento.
- `recontar_auditoria.py` deja una versión explícita de los 133 archivos con
  motivos de exclusión, sin borrar el registro histórico.

| Recuento descriptivo de tipo correcto | Antes | Sin copias de entrenamiento |
| --- | --- | --- |
| 12 kHz | 98/105 (93,33 %) | 98/105 (93,33 %) |
| 48 kHz | 23/26 (88,46 %) | 21/24 (87,50 %) |
| Total mixto | 121/131 (92,37 %) | **119/129 (92,25 %)** |

La reserva de 48 kHz queda en 11/12: 10/11 fallas y 1/1 normal. Un único normal
de reserva no permite afirmar un rendimiento fiable sano/falla.

## 2. Crítico para interpretar mejoras: el calificador encuentra D en normales

Se evaluaron B, IR y OR por separado en las señales etiquetadas normales,
sin dejar que la predicción N impidiera inspeccionar sus bandas. Las tres
señales normales únicas generan al menos un **D** bajo alguna hipótesis.
Las cuatro rutas generan PD/D en las doce combinaciones ruta/hipótesis.

Esto **no es una matriz de falsas alarmas del modelo completo**: si el clasificador
emite N, el sistema no abre esas hipótesis. Es un contraejemplo a usar el grado
como juez independiente: encontrar armónicos y dominancia tras buscar bandas
no demuestra por sí solo que exista la falla atribuida. La confianza del diagnóstico
está depositada en el clasificador, cuyo reconocimiento de normales apenas se validó.

También se construyeron 48 señales sintéticas: seis realizaciones por combinación
de cuatro controles y dos frecuencias de muestreo. Se probaron tres hipótesis
de falla por señal, es decir, 144 llamadas al calificador. Hubo un PD en una
señal con modulación sintética sin rodamiento y ningún D. El resto fue ND.
Esto es un ensayo pequeño de estrés, **no una estimación calibrada de falsas
alarmas**, ni justifica interpretar el p nominal de 0,01 como una tasa real del 1 %.
La corrección de Bonferroni no necesita independencia entre bandas; lo que falta
es validar la calibración de las pruebas individuales con el fondo local estimado.

Evidencia íntegra: `controles_calificador.csv`. Las hipótesis se evaluaron
deliberadamente por separado; no se modificaron las etiquetas reales para
convertirlas en falsos negativos o positivos del sistema completo.

## 3. Alto y confirmado: el flujo actual no exige estabilidad temporal

`calificar` busca el mejor grado entre 31 bandas y retorna al encontrar D.
La inferencia supervisada y la asignación actual del grado no comprueban que
esa evidencia se repita en otro segmento. Los experimentos temporales históricos
de otros métodos no implican que esta comprobación esté en el modelo actual.

Se hizo una prueba adicional, fijada antes de ver su resultado:

1. Conservar el tipo ya predicho, sin consultar el tipo real para seleccionar bandas.
2. Elegir banda y familia usando únicamente la primera mitad.
3. Comprobar esa misma banda y familia en la segunda mitad, sin volver a buscar otra.
4. Tomar como grado conservador el mínimo entre el histórico, la primera mitad
   y la verificación de la segunda mitad.

En los 105 registros de 12 kHz con referencia:

| Resultado respecto del paper, separando tipos incorrectos | Regla anterior | Prueba temporal adicional |
| --- | --- | --- |
| Mejoran con tipo correcto | 24 | 22 |
| Iguales con tipo correcto | 66 | 56 |
| Empeoran con tipo correcto | 8 | 20 |
| Tipo incorrecto | 7 | 7 |
| Sin retroceso y con tipo correcto | 90/105 (85,71 %) | **78/105 (74,29 %)** |

En total, 19 grados emitidos de 12 kHz se degradan bajo este requisito adicional.
No se cambió retroactivamente la definición de la métrica anterior: se muestra
cuánto se sostiene bajo una exigencia adicional de persistencia.

**Límite decisivo:** el tipo conservado fue predicho usando el registro completo.
Por eso este ensayo es condicional a la predicción existente; no es una evaluación
independiente del clasificador ni una validación temporal completamente ciega.
El cambio de duración también cambia la resolución y la potencia estadística:
una degradación no demuestra por sí sola que la falla desaparezca en el tiempo.

`persistencia_temporal.csv` conserva el ensayo inicial completo, incluidas las
rutas repetidas para trazabilidad. `persistencia_temporal_sin_copias.csv` excluye
la copia contaminada de reserva; contiene 128 señales únicas con falla.

## 4. Alto: todavía hay varias fuentes de optimismo sin cuantificar

- **Selección sobre el resultado informado.** Para 12 kHz se ensayaron seis
  bancos por cuatro clasificadores y se exportó el ganador de esos mismos
  resultados (`ensayar_espectral.py`, `verificar_y_exportar_espectral.py`).
  Repetir la validación del ganador no equivale a selección anidada.
- **Reserva por archivos, con condiciones conocidas.** De las 53 fallas de reserva
  de 12 kHz, 49 comparten condición nominal con desarrollo. Todas las fallas
  de reserva de 48 kHz comparten condición con desarrollo. Excluir las copias
  exactas no elimina esta dependencia entre cargas/condiciones de adquisición.
- **Reserva ya observada.** Los archivos fueron usados en experimentos anteriores.
  Volver a llamarlos «reserva intacta» o crear otra partición aleatoria no borra esa historia.
- **Confusión sano/falla con adquisición.** El modelo denominado de 12 kHz se
  entrenó con 52 fallas de 12 kHz y dos normales de 48 kHz. Sus rasgos pueden
  codificar diferencias de adquisición. No se demostró cuánto utiliza ese atajo.
- **Pocos datos frente a muchos rasgos.** Se usan 1953 características; 54 señales
  en el ajuste antiguo y 14 en el de 48 kHz. La regularización permite el ajuste,
  pero no demuestra generalización ni elimina las fuentes anteriores.

Los porcentajes mixtos actuales se mantienen únicamente como descripción de
las predicciones ejecutadas. **No se conoce todavía el porcentaje de acierto
en adquisiciones genuinamente nuevas.** No se calculó un intervalo que trate los
archivos correlacionados como 129 experimentos independientes.

## 5. Alto: cobertura y significado de las salidas

El modelo de 48 kHz solo entrenó B, IR y N. No puede emitir OR; frente a una
falla externa devolverá alguna de las clases conocidas. Ninguno de los dos
modelos valida una clase de falla de jaula ni una salida de condición desconocida.
Los scores son discriminantes, no probabilidades calibradas.

El método actual es un clasificador supervisado sobre un árbol de características
espectrales que incluye curtosis y entropía. No debe atribuirse su rendimiento
exclusivamente a «kurtogram + entropygram» o a un filtro de doble banda.
En 48 kHz usa solo 0–6 kHz; no está aprovechando todas las bandas hasta 24 kHz.

El evaluador selecciona sensor y catálogo mediante `sensor_falla`. Eso es válido
si el componente/sensor se configura antes de diagnosticar, pero no demuestra
localización ciega del rodamiento afectado entre todos los componentes de una máquina.
Los factores del catálogo proceden de una tabla de referencia del proyecto;
no se verificó aquí una ficha original del fabricante. Sí se entregan todas las
familias, no únicamente la frecuencia de la falla real.

ND significa «evidencia no diagnosticable» bajo una regla; no equivale a normal.
Además, D/PD/ND son reglas propias, distintas de la rúbrica del paper. Las 24
o 22 mejoras son **transiciones operativas de grado**, no una demostración de
superioridad bajo un criterio común validado. Las referencias transcritas cubren
105 archivos de 12 kHz y no se certificaron de nuevo celda por celda.

## 6. Corrección de una trampa histórica que seguía disponible en código

El antiguo `comparar_grado` convertía cualquier tipo equivocado en ND. Si el
paper también decía ND, la evaluación sumaba un éxito aunque se hubiera emitido
PD o D para otra falla. Se eliminó esa implementación duplicada: ahora reutiliza
la comparación que conserva el error de tipo. Los evaluadores históricos también
recalculan las filas importadas y conservan la columna `tipo_incorrecto`.

El antiguo 99/105 queda retirado como resultado válido. Los archivos históricos
se conservan para reproducir la auditoría anterior; no deben confundirse con la
regla actual (90/105) ni con la prueba temporal adicional (78/105).
Las pruebas ya no exigen que aquel contador inválido sea 99 para pasar.
También se retiró el requisito de superar el 80 % para exportar un ajuste:
un resultado bajo debe conservarse e informarse, no invalidar una ejecución correcta.

## 7. Comprobaciones que pasaron y límites de esta auditoría

- Inspección del flujo de inferencia: señal, fs y catálogo completo; sin etiqueta
  o frecuencia marcada como correcta en los argumentos del predictor.
- Lectura y huella de los canales usados en los 133 archivos; las dos parejas
  idénticas fueron detectadas sin depender de los nombres.
- Recalculados desde señal cruda los 14 vectores de entrenamiento de 48 kHz:
  error máximo frente al archivo guardado **0,0**.
- Reajustado el LDA de 48 kHz desde esos vectores: medias, escalas, coeficientes,
  interceptos y clases coinciden con el exportado.
- La reproducción integral de los 54 vectores de 12 kHz pertenece a la auditoría
  anterior. Aquí se revisaron su flujo y su artefacto congelado, sin afirmar que
  se haya repetido ese recálculo integral en esta ejecución.
- Los pesos de ambos modelos permanecieron intactos. La validación del escalado
  y el ajuste dentro de cada pliegue no corrige la selección del ganador fuera de esos pliegues.

SHA-256 de los modelos auditados:

```text
12k  75d7ef5443b7861740499592c22ce09a06c51e47d31cf1bd7da4768c2537df56
48k  0b049c9e838046ce8fdb5e9703227851779379e6179c96ff1698c3cce49dbb5a
```

## 8. Condiciones mínimas para la siguiente evaluación

1. Resolver duplicados y metadatos desde la fuente; agrupar copias, adquisiciones
   del mismo rodamiento y sus cargas antes de dividir. Una ruta distinta no es otra adquisición.
2. Fijar el protocolo antes de puntuar: selección interna de variantes, prueba
   exterior por grupos físicos, preprocesado comparable y varias adquisiciones normales.
3. Seleccionar tipo, banda y umbrales sin usar el segmento donde se juzga su
   evidencia; calibrar también la selección entre bandas y las falsas alarmas.
4. Separar identificación del tipo, detección sano/falla, abstención y calidad
   de evidencia. La rúbrica debe penalizar diagnósticos fuertes en controles normales.
5. Mantener la incertidumbre para clases sin datos. No presentar OR/FTF o condiciones
   desconocidas como cobertura validada de un clasificador que no las entrenó.
6. Obtener adquisiciones realmente nuevas para una afirmación final de generalización.
   La base actual puede servir para desarrollar y hacer pruebas internas, con sus límites.

No se ajustaron pesos, umbrales, particiones ni criterios buscando recuperar la meta de 95.

## Reproducción local

```powershell
python 05_Diagnostico_Ciego/auditar_modelos_actuales.py
python 05_Diagnostico_Ciego/recontar_auditoria.py
python 05_Diagnostico_Ciego/integridad_senales.py
python 05_Diagnostico_Ciego/evaluar_modelos_todos.py
python 05_Diagnostico_Ciego/test_ejecucion_modelos.py
python 05_Diagnostico_Ciego/test_no_retroceso.py
python 05_Diagnostico_Ciego/test_criterio_usuario.py
```

El auditor informa el estado que exista al comenzar. `hechos.json` de la primera
ejecución conserva los 131 evaluables anteriores a la corrección; los recuentos
corregidos están en `consolidado_auditado_133.csv` y `resumen_sin_copias.csv`.
Se pueden verificar inventario, filtraciones, grados y controles sin aceptar una
conclusión basada solamente en un porcentaje agregado.
