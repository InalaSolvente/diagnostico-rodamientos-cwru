"""Inferencia experimental desde x y fs; sin etiquetas, RPM ni frecuencia de falla."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_MFD_Kurtogram_Tarea1"))
import Fuentes as F


def buscar_periodicidad(espectros, fmin=10., fmax=1000., p=.01):
    """Busca familias de hasta tres armónicas; requiere al menos dos significativas.

    Heurística: hasta 32 máximos por banda generan candidatos f_pico/h.
    La corrección nominal cubre candidatos, bandas y tres armónicas; no
    calibra la selección adaptativa ni demuestra una tasa real de falsas alarmas.
    """
    if not np.isfinite([fmin, fmax, p]).all() or not 0 < fmin < fmax or not 0 < p < 1:
        raise ValueError("Rango frecuencial o p inválido")
    candidatos = []
    for f, s in espectros.values():
        indices = np.flatnonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])) + 1
        indices = indices[(f[indices] >= fmin) & (f[indices] <= 3*fmax)]
        indices = indices[np.argsort(-s[indices], kind="stable")[:32]]
        candidatos.extend(float(f[i]/h) for i in indices for h in (1, 2, 3)
                          if fmin <= f[i]/h <= fmax)
    candidatos = sorted(set(candidatos))
    if not candidatos or not espectros:
        return []
    # Bonferroni nominal: todos los candidatos realmente examinados.
    p_prueba = p / (len(candidatos) * len(espectros) * 3)
    resultados = []
    for frecuencia in candidatos:
        for nodo, (f, s) in espectros.items():
            d = F.evaluar_diagnostico(f, s, frecuencia, H=3, p=p_prueba)
            armonicas = [a for a in d["armonicas"] if a["significativa"]]
            if len(armonicas) < 2:
                continue
            score = sum(max(0., a["ratio_db"]-a["umbral_db"]) for a in armonicas)
            resultados.append(dict(frecuencia_hz=frecuencia, nodo=list(nodo),
                                   armonicas=[a["h"] for a in armonicas],
                                   score=float(score), p_prueba=p_prueba))
    resultados.sort(key=lambda r: (-len(r["armonicas"]), -r["score"], r["frecuencia_hz"]))
    distintos = []
    for r in resultados:
        if all(abs(r["frecuencia_hz"]-a["frecuencia_hz"]) > .02*a["frecuencia_hz"] for a in distintos):
            distintos.append(r)
        if len(distintos) == 3:
            break
    return distintos


def validar_catalogo(frecuencias, fs):
    if not isinstance(frecuencias, dict) or set(frecuencias) != {"BPFI", "BPFO", "BSF", "FTF"}:
        raise ValueError("Se requiere el catálogo completo BPFI/BPFO/BSF/FTF")
    valores = np.array(list(frecuencias.values()), dtype=float)
    if not np.isfinite(valores).all() or np.any(valores <= 0) or 6*max(valores) >= fs/2:
        raise ValueError("Frecuencias inválidas o sin margen para las armónicas")


ESTRATEGIAS = ("conservador", "sensible", "dominante", "sin_control")


def evidencia_catalogo(espectros, frecuencias, p=.01):
    """Evidencia de todas las hipótesis, antes de elegir tipo de falla."""
    hipotesis = [("BPFI", "IR", frecuencias["BPFI"]), ("BPFO", "OR", frecuencias["BPFO"]),
                 ("BSF", "B", frecuencias["BSF"]), ("2BSF", "B", 2*frecuencias["BSF"]),
                 ("FTF", "FTF", frecuencias["FTF"])]
    if not espectros:
        return []
    resultados = []
    for familia, tipo, frecuencia in hipotesis:
        for nodo, (f, s) in espectros.items():
            for control in (True, False):
                p_prueba = p/(len(hipotesis)*len(espectros)*3) if control else p
                d = F.evaluar_diagnostico(f, s, float(frecuencia), H=3, p=p_prueba)
                armonicas = [a for a in d["armonicas"] if a["significativa"]]
                resultados.append(dict(familia=familia, tipo=tipo, frecuencia_hz=float(frecuencia),
                    nodo=list(nodo), armonicas=[a["h"] for a in armonicas],
                    score=float(sum(a["ratio_db"]-a["umbral_db"] for a in armonicas)),
                    categoria=d["categoria"], control=control, p_prueba=p_prueba))
    return resultados


def clasificar_evidencia(evidencia, estrategia="dominante"):
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"Estrategia inválida: {estrategia}")
    minimo = 2 if estrategia == "conservador" else 1
    resultados = [r for r in evidencia if r["control"] == (estrategia != "sin_control")
                  and len(r["armonicas"]) >= minimo]
    def orden(r):
        if estrategia == "conservador":
            return (-len(r["armonicas"]), -r["score"], r["familia"])
        if estrategia in ("dominante", "sin_control"):
            return (r["categoria"] != "D", -r["score"], r["familia"])
        return (-r["score"], r["familia"])
    resultados.sort(key=orden)
    distintos = []
    for r in resultados:
        if r["tipo"] not in [a["tipo"] for a in distintos]:
            distintos.append(r)
    return distintos[:3]


def buscar_catalogo(espectros, frecuencias, p=.01, estrategia="dominante"):
    return clasificar_evidencia(evidencia_catalogo(espectros, frecuencias, p), estrategia)


def inferir(x, fs, *, frecuencias_fabricante=None, L=4, fmin=10., fmax=1000., p=.01,
            estrategia="dominante"):
    """Señal y catálogo completo opcional; nunca la frecuencia ni el defecto real."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 1024 or not np.isfinite(x).all():
        raise ValueError("Se requiere una señal finita unidimensional de al menos 1024 muestras")
    if not np.isfinite(fs) or fs <= 0 or not isinstance(L, int) or not 0 <= L <= 8:
        raise ValueError("fs o L inválidos")
    if x.size / 2**L < 32:
        raise ValueError("Demasiados niveles para la longitud de señal")
    if not np.isfinite([fmin, fmax, p]).all() or not 0 < fmin < fmax or 3*fmax >= fs/2 or not 0 < p < 1:
        raise ValueError("Se requiere 0 < fmin < fmax < fs/6 y p en (0,1)")
    if frecuencias_fabricante is not None:
        validar_catalogo(frecuencias_fabricante, fs)
    if estrategia not in ESTRATEGIAS:
        raise ValueError("Estrategia inválida")
    if not np.any(x != x[0]):
        return dict(estado="sin_evidencia", candidatos=[], evidencia_catalogo=[], bandas=[], L=L)
    tabla = F.kurtogram(x, fs, L, con_entropia=True)
    subconjunto = tabla.sort_values("H", kind="stable").head(int(np.ceil(.25*len(tabla)))).sort_index()
    nodos = list(dict.fromkeys([F.seleccionar_banda(tabla), F.seleccionar_banda(subconjunto)]))
    espectros = {}
    energia = np.mean((x-x.mean())**2)
    for nodo in nodos:
        xf = F.filtrar(x, fs, *nodo)
        energia_banda = np.mean(xf**2)
        if energia_banda <= np.finfo(float).eps * energia:
            continue
        f, s = F.envolvente_espectral(xf, fs)
        if s.sum() > np.finfo(float).eps * energia_banda:
            espectros[nodo] = (f, s)
    evidencia = evidencia_catalogo(espectros, frecuencias_fabricante, p) if frecuencias_fabricante is not None else []
    candidatos = (buscar_periodicidad(espectros, fmin, fmax, p) if frecuencias_fabricante is None
                  else clasificar_evidencia(evidencia, estrategia))
    return dict(estado="periodicidad_detectada" if candidatos else "sin_evidencia",
                candidatos=candidatos, evidencia_catalogo=evidencia, bandas=[list(n) for n in nodos], L=L)
