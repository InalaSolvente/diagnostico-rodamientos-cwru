# Revisión de condiciones y experimentos de mejora

El resultado histórico 88/105 mezcla OOF en desarrollo con un ajuste congelado en registros que comparten condición con entrenamiento.
No se elimina: se conserva como resultado del protocolo mixto, pero no prueba generalización a condiciones nuevas.

La reevaluación excluye del entrenamiento todas las cargas, sensores y muestreos de la condición evaluada.
Conserva los 133 archivos, incluidos el corto sin diagnóstico y los cuatro normales. Reproduce los 52 OOF históricos caso por caso.
Las condiciones OR conservan la posición angular en su identificador. Esto no garantiza independencia entre rodamientos físicos: los identificadores disponibles no la demuestran.

Resultado corregido: 78/105 sin retroceso; 8 mejoras, 7 empeoramientos y 20 errores de tipo.
48 kHz: 18/24 tipos correctos, incluido un archivo insuficiente. Normales: 4/4.

## Experimentos fijos sobre los 52 casos de desarrollo

Clasificador histórico: 42/52 tipos correctos y 37/52 sin retroceso.
Las variantes conservan detector, juez y separación temporal; no se ha reemplazado el modelo histórico.
Los grados de las predicciones correctas se recuperan de la evidencia ya calculada para el mismo archivo y tipo, después de congelar la predicción. La etiqueta no participa en la selección del tipo ni del filtro.

 total  mejoran  iguales  empeoran  tipo_incorrecto  sin_retroceso                            variante  cambios_tipo_vs_base
    52        4       32         5               11             36          resultados_tipo_por_sensor                     3
    52        3       34         5               10             37            resultados_tipo_promedio                     0
    52        3       34         5               10             37 resultados_tipo_envolvente_cuadrada                     0

## Alcance

No se alcanzó el objetivo de al menos 90/105 sin retroceso y muchas más mejoras que empeoramientos.
El dataset y el desarrollo han sido inspeccionados repetidamente. Ninguno de estos ensayos constituye validación externa ni demuestra ausencia de sobreajuste por selección reiterada.
No se usa la frecuencia verdadera en inferencia, no se fabrican ND para errores y no se modifica la clasificación de referencia.

Reproducir la verificación: `python 05_Diagnostico_Ciego/verificar_revision_condiciones.py`.
