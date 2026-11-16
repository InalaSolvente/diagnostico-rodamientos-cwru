# Prueba de alineación física de bandas

A 48 kHz se usa L=6 para alcanzar 375 Hz, igual que 12 kHz/L4.
El modelo congelado recibe los 31 nodos físicamente equivalentes de 0–6 kHz (niveles globales 2–6).
No es usar todo el árbol de 127 nodos: su dimensionalidad sería incompatible con los pesos existentes.
Sin remuestreo, ajuste de umbrales ni entrenamiento. RMS global, soporte de entropía y densidad espectral siguen dependiendo de fs.
La calidad se evalúa sobre esos mismos 31 nodos y con la regla del usuario.
Las señales a 12 kHz permanecen iguales por construcción y por prueba de equivalencia numérica.
Los normales de desarrollo habían entrenado con la geometría antigua; esta prueba cambia sus rasgos sin recalibrar pesos.

| carpeta | total | tipos correctos antes | tipos correctos ahora | D | PD | ND |
| --- | --- | --- | --- | --- | --- | --- |
| 48k_DE | 24 | 0 | 12 | 10 | 13 | 1 |
| Normal | 4 | 2 | 0 | 2 | 2 | 0 |

## Interpretación verificada

El modelo asignó IR a las 28 señales: 12 aciertos en 24 fallas y ninguno en los cuatro normales. El 50% en fallas coincide con elegir siempre la clase IR; no demuestra discriminación entre tipos. Igualar anchos cambió las predicciones, pero no resolvió la transferencia a 48 kHz. Los pesos se entrenaron con la geometría anterior para los normales a 48k: una futura evaluación consistente requerirá preparar entrenamiento e inferencia con la misma representación.
