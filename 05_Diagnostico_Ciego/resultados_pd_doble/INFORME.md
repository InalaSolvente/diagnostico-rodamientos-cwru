# Globales, control de giro, doble banda y PD persistente

Resultado: {'total': 105, 'mejoran': 16, 'iguales': 72, 'empeoran': 5, 'tipo_incorrecto': 12, 'sin_retroceso': 88}. Cuatro normales correctos. Cobertura: 133 archivos; referencia del paper: 105.
Se elige el banco de 31 por simplicidad: el de 63 arroja los mismos recuentos y exige más cómputo. No se elige un banco distinto por registro.
Los 52 casos de desarrollo siguen separados por condición en evaluación; se reprodujeron todos los tipos y se recalcularon desde MAT las correcciones PD.
Cambiar completamente la segunda mitad conservó tipo, alternativas y ambas bandas del predictor público; solo cambió la verificación. No acepta etiquetas, nombres de archivos ni frecuencia de falla esperada.
Los grados D fueron comprobados por rama: primera armónica significativa, al menos dos armónicas y ningún pico previo mayor. Nunca se suman armónicas de bandas distintas. El umbral conserva la corrección por el banco completo.

## Límites metodológicos

La meta de al menos 90/105 sin retroceso todavía no se alcanza.
Las reglas fueron desarrolladas después de inspeccionar resultados del dataset. Estos recuentos son exploratorios: no equivalen a una selección de modelos ciega ni a validación externa.
La exclusión por condición todavía distingue posiciones OR de una misma dimensión de defecto. La independencia entre rodamientos físicos no está demostrada. Debe auditarse con una exclusión más conservadora antes de afirmar generalización.
Los p son nominales y no constituyen una garantía empírica de falsas alarmas; cuatro normales y dos archivos normales de desarrollo son pocos.
El modelo original se conserva. Los pesos nuevos se ajustan solo con los 52 registros de desarrollo y no se usan para fabricar las cifras OOF.

Inferencia: `from predictor_global_doble import predecir`; `predecir(senal, fs, catalogo_completo)`.
Exportar pesos: `python 05_Diagnostico_Ciego/predictor_global_doble.py --exportar`.
Verificar: `python 05_Diagnostico_Ciego/verificar_global_doble.py`.
