# Comparación curtosis / entropía

Ejecutar desde la raíz del proyecto:

```powershell
python 01_MFD_Kurtogram_Tarea1/test_entropia.py
python 01_MFD_Kurtogram_Tarea1/comparacion_entropia.py 287 282
python 01_MFD_Kurtogram_Tarea1/comparacion_entropia.py
```

Sin números procesa los registros de 12 kHz. `--datos` permite cambiar la carpeta
CWRU y `--salida` la carpeta de resultados. Se escriben la comparación por registro
y el conteo D/P/N por peso, en `resultados/entropia/`.

Los pesos están en orden **curtosis / entropía**: 70/30, 50/50, 30/70,
con 100/0 como referencia. Se maximiza
`S = wK * (K-min(K))/(max(K)-min(K)) + (1-wK)*(1-H)`.
La normalización de K se calcula entre todos los nodos de cada registro;
si K es constante aporta cero. H es Shannon del espectro de potencia de la
envolvente centrada, sin DC, dividida por log(número de bins positivos).
Se utiliza el mismo soporte frecuencial para todos los nodos. Una envolvente
constante o sin energía recibe H=1. Los empates eligen el primer nodo.
La selección no usa fd; el nivel máximo y el diagnóstico siguen el criterio
existente, que sí utiliza fd. El score 100/0 conserva K sin normalizar y no
debe compararse numéricamente con los scores mixtos.

Menor H favorece concentración espectral, pero también puede favorecer una
modulación ajena a la falla. Esto es una hipótesis experimental: comparar bandas,
diagnóstico y falsas alarmas antes de concluir que corrige la selección.
Min-max también depende de valores extremos de K. Las ponderaciones no tienen
interpretación probabilística. El flujo original permanece como referencia.
