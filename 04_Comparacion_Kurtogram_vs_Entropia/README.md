# Kurtogram vs. kurtogram + entropía

## Doble banda con control

```powershell
python 04_Comparacion_Kurtogram_vs_Entropia/test_doble_banda.py
python 04_Comparacion_Kurtogram_vs_Entropia/doble_banda.py
python 04_Comparacion_Kurtogram_vs_Entropia/doble_banda.py --solo-resumen
```

Reutiliza los nodos del kurtogram y del filtro 25% ya calculados en
`resultados/filtros/comparacion_12k.csv` y vuelve a evaluar sus envolventes.
Verifica que se reproduzcan las categorías originales en fd y los señuelos.
La unión acepta la mejor categoría (D > P > N), sin sumar ni mezclar señales.
Si los nodos coinciden, evalúa una sola banda. Con dos bandas distintas usa
p=0,005 por banda en la variante controlada (Bonferroni entre bandas), frente
a p=0,01 en la unión sin corrección. Mantiene las tres armónicas y ±2%.
Esto controla el incremento nominal por buscar en dos bandas: **no** garantiza
1% de falsas alarmas total, pues no corrige las tres armónicas, la selección
adaptativa ni posibles desviaciones del modelo estadístico del fondo.
No se exige una segunda armónica: se prueba únicamente la corrección de umbral.
Los resultados van a `resultados/doble_banda/` con resumen, figura, transiciones
respecto al kurtogram, conteos y configuración con hash del CSV de bandas.
Los señuelos siguen siendo detecciones fuera del objetivo, no fallas descartadas.
Se pueden especificar registros, `--datos`, `--bandas` y `--salida`.

## Estabilidad temporal

```powershell
python 04_Comparacion_Kurtogram_vs_Entropia/test_temporal.py
python 04_Comparacion_Kurtogram_vs_Entropia/compara_entropia.py --temporal
python 04_Comparacion_Kurtogram_vs_Entropia/compara_entropia.py --temporal --solo-resumen
```

Compara kurtogram, filtro 75% global y filtro 75% con medianas temporales.
Divide la señal cruda en cinco ventanas iguales no solapadas (ajustable con
`--ventanas`), calcula el mismo árbol y L en cada una y agrega por nodo las
medianas de K y H. Conserva el 75% con menor H mediana y maximiza K mediana.
La mediana reduce la influencia de ventanas aisladas; no impone un umbral
de variabilidad ni garantiza persistencia en todas las ventanas. La MAD de
K y H se guarda como dato descriptivo y no interviene en la selección.
Se descartan menos de cinco muestras finales con el valor predeterminado;
se registran el descarte y la duración de ventana en muestras en el CSV.
El diagnóstico y los dos señuelos se evalúan sobre la banda de la señal
completa, igual para los tres métodos. Las ventanas cortas cambian resolución
y pueden ocultar fallas intermitentes; esta variante es una hipótesis experimental.
Resultados separados en `resultados/temporal/`, con figura y transiciones
respecto al kurtogram original. `--solo-resumen` conserva el número de ventanas
del CSV; no vuelve a calcular la selección.

## Filtros entropygram → kurtogram

```powershell
python 04_Comparacion_Kurtogram_vs_Entropia/compara_entropia.py --filtros
python 04_Comparacion_Kurtogram_vs_Entropia/compara_entropia.py --filtros --solo-resumen
python 04_Comparacion_Kurtogram_vs_Entropia/test_filtros.py
```

Conserva el 25%, 50% o 75% de los nodos con menor entropía H y elige
el de mayor K entre los conservados. La referencia conserva el 100%.
Se filtran todos los niveles juntos; se redondea hacia arriba el número de
candidatos. Empates de H o K respetan el orden original del árbol.
La definición de H es la misma del experimento ponderado; aquí no se suman
K y H. Cada banda seleccionada se evalúa sin cambiarla en fd y dos señuelos.
Las salidas se guardan aparte en `resultados/filtros/`.
`transiciones_clasificacion.png` compara el kurtogram original (filas) con
cada filtro (columnas): incluye las nueve transiciones ND/P/D, sus conteos
y los totales que mejoran, empeoran o no cambian. Se evalúa en fd, no en
señuelos. El CSV homónimo conserva N como código de ND. Se regenera con
`--filtros --solo-resumen`, sin recalcular señales.

Desde la raíz:

```powershell
python 04_Comparacion_Kurtogram_vs_Entropia/compara_entropia.py
python 04_Comparacion_Kurtogram_vs_Entropia/compara_entropia.py --solo-resumen
```

Compara curtosis pura y pesos curtosis/entropía 70/30, 50/50 y 30/70.
Admite números de registro para ejecutar un subconjunto y `--salida` para
guardarlo aparte. `--datos` cambia la carpeta CWRU.

Reutiliza el criterio, los dos señuelos por registro, la referencia Smith &
Randall y las funciones de resumen y gráfica de 03. Cada método selecciona
su banda una sola vez por registro y esa misma banda se evalúa en fd y los
señuelos. El árbol y L son iguales para todos los pesos.
La definición de la métrica está en `../01_MFD_Kurtogram_Tarea1/ENTROPIA.md`.

En `resultados/` genera `comparacion_12k.csv` con bandas y diagnósticos,
`resumen.csv`, `resumen.md` y `comparacion_resumen.png`.
Los porcentajes D/P/N usan el número de registros; las falsas alarmas usan
el doble, por los dos señuelos. J_D y J_DP son diferencias de proporciones,
igual que en 03. Las detecciones en señuelos son un control experimental,
no una medición de especificidad en rodamientos sanos. Los porcentajes por
categoría S&R son descriptivos; las categorías sin registros dan NaN.
