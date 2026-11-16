# Rasgos crudos y coherencia cíclica

El ensayo obtiene **82/105 sin retroceso**: 13 mejoras, 69 iguales, 3 descensos y 20 tipos incorrectos. Los 23 resultados desfavorables todavía superan las mejoras. La meta de 100/105 sigue pendiente.

| Configuración | Sin retroceso | Mejoras | Desfavorables |
|---|---:|---:|---:|
| Rasgos previos y banco57 | 80 | 14 | 25 |
| Coherencia sola y banco57 | 64 | 4 | 41 |
| Rasgos previos + coherencia y banco57 | 82 | 13 | 23 |

Frente a la primera fila, el ensayo aditivo recupera **6 casos y pierde 4**. No se elige la mejor respuesta de ambos modelos para cada archivo. Los cambios se detallan en `cambios_vs_banco57.csv`.

Se añaden 600 rasgos ACP a los 2021 anteriores. La coherencia normaliza la correlación entre componentes separados por una frecuencia cíclica. Se usa ventana semiseno de 256 muestras, solape del 50 % y corrección de fase absoluta; esta combinación mantiene constante la suma de cuadrados de ventanas en el interior. El estimador se comprobó contra la ecuación directa. La referencia es [Antoni (2007), ecuaciones 13, 17 y 18](https://doi.org/10.1016/j.jsv.2007.02.029).

El extractor examina todas las familias del catálogo y sus armónicos y bandas laterales. Solo usa el prefijo fijo de 2,5 segundos. La clasificación aditiva conserva el LDA y los controles D/PD anteriores; la variante de coherencia sola selecciona internamente entre LDA, SVM y un peine físico fijo. El espectro y el criterio D/PD/ND del juez permanecen iguales; las dos bandas se fijan en la primera mitad y se verifican en la segunda.

Se parte del conjunto autorizado de 52 archivos con falla. En cada ajuste se excluye el tipo y tamaño completos del grupo evaluado, con todas sus posiciones OR, cargas, sensores y muestreos. Ninguna etiqueta exterior ni frecuencia verdadera decide el tipo en inferencia. Las etiquetas se usan para entrenar y, después de congelar los tipos, para puntuar.

La verificación reprodujo los 22 ajustes exteriores de ambos ensayos, los rasgos ACP desde los 132 archivos con longitud suficiente y los grados de los 12 archivos cuyo tipo cambió en el modelo aditivo. También comprobó independencia del futuro a 12 y 48 kHz. El archivo corto sigue dentro de los 133 como sin diagnóstico. Véase `verificacion.json`.

Todo es exploratorio: el dataset ya se ha inspeccionado repetidamente. No hay validación externa ni sustitución de los modelos históricos. Se mantiene el catálogo disponible y sus supuestos previos de geometría, incluida la equivalencia usada para las variantes NTN; no se corrigen sus frecuencias mediante etiquetas verdaderas.

Reproducción desde la raíz:

```powershell
python .\05_Diagnostico_Ciego\coherencia_ciclica.py --preparar
python .\05_Diagnostico_Ciego\evaluar_coherencia_ciclica.py
python .\05_Diagnostico_Ciego\evaluar_crudo_coherencia.py
python .\05_Diagnostico_Ciego\verificar_coherencia_ciclica.py
```

La tabla con los 133 registros y el balance global están en [la revisión de la meta](../revision_meta_100/INFORME.md).
