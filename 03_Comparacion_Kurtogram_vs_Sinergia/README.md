# 03 — Kurtogram (Tarea 1) vs. Sinergia fraterna

Esta comparación usa **el mismo árbol, el mismo nivel máximo, el mismo criterio de diagnóstico y los mismos registros** para todos los métodos:

- 105 registros CWRU de 12 kHz (60 DE y 45 FE).
- Árbol MFD/MDFD con L según la regla de la Tarea 1 (Δf_L ≥ 2·fd, en la práctica L = 4).
- Diagnóstico con `Fuentes.evaluar_diagnostico` de la Tarea 1 (D = diagnosticable, P = parcial, N = no).
- Falsa alarma: el mismo criterio en 2 **frecuencias señuelo** por registro (c·fr con c, 2c y 3c lejos de toda línea cinemática conocida). Si un método "diagnostica" ahí, está viendo ruido.

`python compara.py` genera `resultados/comparacion_12k.csv` (registro por registro), `resultados/resumen.csv` y `resultados/comparacion_resumen.png`.

## Resultados

| Método | D en fd | D en señuelos | J_D | D+P en fd | D+P en señuelos |
|---|---|---|---|---|---|
| Kurtogram (Tarea 1) | 41.9 % | 0.0 % | 0.42 | 73.3 % | 46.2 % |
| Sinergia Z | 45.7 % | 0.5 % | 0.45 | 81.9 % | 63.8 % |
| Descenso por sinergia | 58.1 % | 0.5 % | 0.58 | 95.2 % | 72.4 % |
| Nodo raíz (envolvente cruda) | 59.0 % | 0.5 % | 0.59 | 97.1 % | 65.7 % |
| Sinergia dirigida S_fd | **60.0 %** | 1.4 % | **0.59** | 95.2 % | 76.2 % |

Comparación pareada de la categoría D contra el kurtogram (cuántos registros gana / pierde cada método):
- Sinergia Z: 12 / 8.
- Descenso por sinergia: 18 / 1.
- Sinergia dirigida: 21 / 2.

## Lectura
1. **"Diagnosticable" (D) es una categoría específica:** casi nunca aparece en frecuencias señuelo (≤ 1.4 %). **"Parcial" (P) no es específica:** entre 46 y 76 % de los señuelos muestran "alguna línea significativa", porque en CWRU la envolvente está llena de líneas discretas (armónicos de fr, de FTF y sus combinaciones). Por eso la comparación se hace con la categoría D.
2. **El kurtogram de la Tarea 1 es el método más débil (42 %).** Pierde contra la envolvente sin filtrar (59 %) porque en muchos registros elige bandas con ráfagas impulsivas ajenas a la falla, a menudo junto a Nyquist.
3. **Las reglas que usan el árbol recuperan esa pérdida.** El descenso por sinergia llega a 58 % y la sinergia dirigida a 60 %, con falsa alarma ≤ 1.4 %.
4. **La sinergia Z (argmax ciego) mejora poco al kurtogram (46 %).** Igual que el kurtogram, puede quedar atrapada por impulsos no periódicos (limitación documentada en `02_Sinergia_Fraterna`).
5. Los números no coinciden con la sección de CWRU del informe de la sinergia porque aquel usó otro criterio (relación armónica/fondo ≥ 6 dB sobre un SES de Welch, L = 6). La conclusión cualitativa se mantiene: el kurtogram es el más débil, el descenso por sinergia rinde como la envolvente cruda y la sinergia dirigida tiene el mejor balance entre los métodos dirigidos.
