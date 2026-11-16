# 01 — Tarea 1: MFD + Kurtogram + Envolvente Espectral

Implementación del método del profesor (Multiscale Fourier Decomposition vía curtosis espectral).

| Archivo | Contenido |
|---|---|
| `Fuentes.py` | MFD (máscaras, sub-señales, señal analítica, curtosis compleja, kurtogram), envolvente espectral, criterio de diagnóstico (D/P/N) y gráficas |
| `comparacion_descenso.py` | `python comparacion_descenso.py` → figuras kurtogram vs. descenso por sinergia en `figuras/comparacion_descenso/`: envolvente relativa al fondo local (umbral horizontal, picos coloreados por familia: armónica de fd, bandas laterales, n·fr, giro–jaula), zoom a 1×, 2× y 3×fd con ventana ±2 %, fondo ±0.3 fd, umbral y pico; registros 107, 108, 147, 170, 280, 297, 287, 282, 222, más `galeria_1.png` y `galeria_2.png` con los 18 registros donde el descenso gana (≈2 min) |
| `tarea1_main.py` | `python tarea1_main.py 287 282` → figuras + fichas; `python tarea1_main.py` → los 105 registros |
| `figuras/287/`, `figuras/282/` | las 6 figuras de la tarea + kurtogram + `diagnostico.json` |
| `resultados/fichas.md` | fichas de entregable (Parcialmente / No diagnosticable) |
| `resultados/tabla_registros_12k.csv` | diagnóstico de los 105 registros (señal cruda y filtrada) |
| `informe/Tarea1_informe.pdf` | informe de la tarea (y su fuente LaTeX) |

Casos elegidos:
- **Parcialmente diagnosticable:** archivo 287 (FE, bola 0.014 in, 1 hp).
- **No diagnosticable:** archivo 282 (FE, bola 0.007 in, 0 hp; el ejemplo de clase).
