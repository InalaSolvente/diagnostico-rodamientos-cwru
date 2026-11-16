# Búsqueda bibliográfica — novedad de la "sinergia fraterna"

Fecha: 23-09-2026. Acceso institucional PUCV (Scopus vía Elsevier ID; Web of Science Core Collection vía proxy biblioteca PUCV).

## Tamaño del campo
| Base | Consulta | Resultados |
|---|---|---|
| Scopus | `TITLE-ABS-KEY((kurtogram OR "demodulation band" OR "informative frequency band" OR "optimal frequency band" OR "optimal demodulation") AND bearing*)` | 601 (≈50–65/año desde 2018; MSSP 75, Measurement 45, MST 35) |
| WoS | `TS=((kurtogram OR "demodulation band" OR "informative frequency band" OR "optimal frequency band" OR "optimal demodulation") AND bearing*)` | 646 |

## Búsquedas de novedad
| Idea buscada | Scopus | WoS | ¿Algo igual? |
|---|---|---|---|
| Selección de banda + hermanos / nodos padre-hijo / bandas adyacentes | 5 | 0 | No (antenas, metamateriales) |
| Descomposición / herencia de la curtosis en sub-bandas (`kurtosis W/3 decompos*…`) | 110 | 27 | No (métodos de descomposición + curtosis) |
| Co-curtosis / curtosis cruzada entre bandas | 57 | 0 | No (finanzas, filtros RF) |
| Espectro cruzado de envolventes cuadráticas entre bandas | 0 | 0 | — |
| Correlación de envolventes entre bandas / acoplamiento entre frecuencias (máquinas) | 37 | 3 | No; sólo correlación cruzada de envolventes entre **dos sensores** (MST 2025) |
| Kurtogram / selección de banda + surrogados, prueba de hipótesis o falsa alarma | 7 | 4 | No |
| Mapas de correlación / dependencia entre bandas | 10 | 9 | **Sí, lo más cercano** (grupo de Wrocław) |
| Espectro de envolvente de producto / "dual band" | 9 | 7 | **Relacionado** con la versión cíclica |

## Trabajos que hay que citar y diferenciar
1. Nowicki, Hebda-Sobkowicz, Zimroz et al. (2021) *Dependency measures for the diagnosis of local faults…*, Applied Acoustics, 10.1016/j.apacoust.2021.107974
2. Hebda-Sobkowicz, Zimroz, Kumar et al. (2025) *Robust correlation measures for IFB selection in heavy-tailed signals*, Adv. Eng. Informatics, 10.1016/j.aei.2025.103174
3. Michalak, Hebda-Sobkowicz, Kumar et al. (2025) *NTF-based dependence map analysis…*, Measurement, 10.1016/j.measurement.2025.116840
4. Chen et al. (2023) *Product envelope spectrum optimization-gram (PESOgram)*, MSSP, 10.1016/j.ymssp.2023.110270
5. Wang et al. (2026) *EGOgram: … dual band synergistic enhancement …*, MST, 10.1088/1361-6501/ae8c72 — multiplica los espectros de envolvente de dos bandas "fuertemente correlacionadas".
6. Peng et al. (2026) *Adaptive Correlated Kurtogram and Product Envelope Spectrum*, IEEE TIM, 10.1109/TIM.2026.3731718
7. (2025) *Identification of compound faults … envelope-cross-correlation and improved 1D-LBP*, MST, 10.1088/1361-6501/adbd5d — correlación cruzada de envolventes entre dos sensores.

## Conclusión
Ninguna base mostró la identidad de herencia de la curtosis padre → hijos, la sinergia estandarizada con surrogados fraternos ni la descomposición cíclica de la sinergia (espectro cruzado entre hermanos, insesgado ante ruido).
La idea general de "similitud entre bandas" existe (mapas de dependencia) y la combinación de espectros de envolvente de varias bandas también (PES/EGOgram). El aporte hay que plantearlo como: identidad exacta y sus consecuencias, nula de surrogados y base teórica del espectro cruzado entre hermanos.
