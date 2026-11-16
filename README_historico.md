# Proyecto — Detección de Anomalías (PUCV, 2° sem. 2026)

Diagnóstico de rodamientos con el dataset CWRU. Cada implementación de un método está en su propia carpeta.

```
Proyecto deteccion de anomalias/
├── TAREA1.pdf                          enunciado de la Tarea 1
├── data/
│   └── CWRU/                           datos del Bearing Data Center (Case Western Reserve University)
│       ├── 12k_DE/  (60 .mat)          falla en rodamiento Drive End, Fs = 12 kHz
│       ├── 12k_FE/  (45 .mat)          falla en rodamiento Fan End,   Fs = 12 kHz
│       ├── 48k_DE/  (24 .mat)          falla Drive End,               Fs = 48 kHz
│       ├── Normal/  (4 .mat)           línea base sin falla,          Fs = 48 kHz
│       ├── indice_cwru.csv             n° de registro CWRU, zona, tamaño, carga, rpm, notas
│       └── README.md
├── 01_MFD_Kurtogram_Tarea1/            método del profesor: MFD + kurtogram + envolvente espectral
├── 02_Sinergia_Fraterna/               métrica nueva basada en la estructura padre-hijos del árbol
└── 03_Comparacion_Kurtogram_vs_Sinergia/  mismo árbol, mismo criterio, frecuencias señuelo
```

## Modelo sin frecuencia de falla conocida

La inferencia nueva está en `05_Diagnostico_Ciego/`: recibe señal y catálogo
completo del rodamiento, sin indicar la falla real, con profundidad fija.
Las próximas pruebas usan solo la mitad de desarrollo fijada por registro.
Los resultados históricos de 01–04 usaban frecuencia objetivo, incluso para
definir L; no representan exactitud de diagnóstico ciego.

## Requisitos
Python 3.10+ con `numpy`, `pandas`, `scipy` (sólo para leer `.mat`) y `matplotlib`.
Para el notebook: `jupyter`.

```
pip install numpy pandas scipy matplotlib jupyter
```

## Cómo correr
| Carpeta | Comando | Qué produce |
|---|---|---|
| 01 | `python tarea1_main.py 287 282` | 6 figuras + kurtogram + ficha de cada registro |
| 01 | `python tarea1_main.py` | diagnóstico de los 105 registros de 12 kHz (≈5 min) |
| 02 | abrir `codigo/sinergia_fraterna.ipynb` | teoría, simulaciones y ejemplo CWRU de la sinergia |
| 03 | `python compara.py` | tabla y figura comparativa (≈20 min) |

Todos los scripts encuentran los datos solos en `data/CWRU` (rutas relativas al proyecto).
