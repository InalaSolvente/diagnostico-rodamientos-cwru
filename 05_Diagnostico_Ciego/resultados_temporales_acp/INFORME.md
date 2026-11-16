# Tres vistas temporales de selección

El ensayo obtiene **84/105 sin retroceso**: 14 mejoras, 70 iguales, 4 descensos y
17 tipos incorrectos. Los 21 resultados desfavorables superan las 14 mejoras:
la meta de 100/105 y el criterio de muchas más mejoras siguen pendientes.

Frente a 82/105, recupera 4 casos y pierde
2. No se elige la mejor respuesta por archivo.
Los cambios se detallan en `cambios_vs_82.csv`.

Se extraen tres vistas de 2.5 segundos al inicio, centro y final de la primera
mitad. Cada una aporta cuatro ventanas de rasgos crudos más ACP; el LDA agrega
sus puntuaciones. Son doce vistas correlacionadas de cada archivo, no doce
casos independientes. El corpus de desarrollo sigue teniendo 52 archivos con
falla; cada ajuste excluye todo el tipo y tamaño del grupo evaluado, incluidas
todas las posiciones OR, cargas, sensores y muestreos.

El catálogo completo disponible, fs y la señal son las entradas físicas.
Las etiquetas solo se emplean en entrenamiento y evaluación. No se utiliza
la frecuencia verdadera de falla. Se mantienen el detector, los controles D/PD,
el banco57 y el criterio de diagnóstico. La segunda mitad solo verifica las
bandas fijadas en selección; no aporta ventanas al clasificador.

La verificación conjunta reprodujo los 22 ajustes exteriores de este ensayo y
del clasificador con evidencia84, y recalculó las doce vistas desde los 132 MAT
de longitud suficiente. También reprodujo los grados de los diez archivos cuyo
tipo cambió, comprobó las fronteras temporales y modificó el futuro a 12/48 kHz
sin alterar los rasgos. El archivo corto sigue dentro de los 133 como sin
diagnóstico. Véase `verificacion.json`.

El clasificador alternativo que añade los 84 indicadores D/PD como rasgos y
elimina los forzados obtuvo 74/105; no se adopta. Sus indicadores se reprodujeron
desde la primera mitad de los 127 archivos donde se usan, para las tres hipótesis,
con diferencia máxima cero. Esa alternativa aprende el tipo, no cambia el juez.

Todo es exploratorio, sin validación externa. Los modelos históricos no se han
reemplazado. Los supuestos previos del catálogo y la geometría siguen vigentes.

Reproducción desde la raíz:

```powershell
python .\05_Diagnostico_Ciego\rasgos_temporales_acp.py --preparar
python .\05_Diagnostico_Ciego\evaluar_temporales_acp.py
python .\05_Diagnostico_Ciego\evaluar_juez_aprendido_acp.py
python .\05_Diagnostico_Ciego\verificar_juez84_fuente.py
python .\05_Diagnostico_Ciego\verificar_temporales_y_juez_acp.py
```

La comparación completa y la cobertura de los 133 registros están en
[la revisión de la meta](../revision_meta_100/INFORME.md).
