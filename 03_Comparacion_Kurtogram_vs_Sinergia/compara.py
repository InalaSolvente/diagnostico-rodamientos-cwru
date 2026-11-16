"""
compara.py — Kurtogram de la Tarea 1 vs. métricas de sinergia fraterna, con el MISMO criterio.

Para cada registro CWRU de 12 kHz (DE y FE, 105 registros):
  * mismo árbol MDFD y mismo nivel máximo L (regla de la Tarea 1: Δf_L >= 2·fd),
  * cada método elige una banda,
  * la banda se evalúa con el mismo criterio de la Tarea 1 (Fuentes.evaluar_diagnostico sobre la
    envolvente espectral de clase): D = diagnosticable, P = parcial, N = no diagnosticable,
  * falsa alarma: el mismo criterio aplicado a 2 frecuencias SEÑUELO por registro, lejos de toda
    línea cinemática (si el método "detecta" ahí, detecta ruido).

Métodos:
  Kurtogram (Tarea 1)      argmax K                      (ciego)
  Sinergia Z               argmax Z (surrogados)         (ciego)
  Descenso por sinergia    raíz -> hoja, para si Z >= 3  (ciego)
  Nodo raíz (SES crudo)    sin filtrar                   (referencia)
  Sinergia dirigida S_fd   argmax S_fd                   (dirigido: usa fd)

Uso:  python compara.py   (≈25 min)  ->  resultados/comparacion_12k.csv, resumen.csv, comparacion_resumen.png
      python compara.py --solo-resumen   (rehace tabla y figura desde el CSV)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(RAIZ, "01_MFD_Kurtogram_Tarea1"))
sys.path.insert(0, os.path.join(RAIZ, "02_Sinergia_Fraterna", "codigo"))

import Fuentes as F                                   # noqa: E402  (Tarea 1)
import mdfd_sinergia as SF                            # noqa: E402  (sinergia fraterna)
from experimento_cwru import BENCH                    # noqa: E402  (categorías Smith & Randall)

RES = os.path.join(AQUI, "resultados")
METODOS = ["Kurtogram (Tarea 1)", "Sinergia Z", "Descenso por sinergia", "Nodo raíz (SES crudo)",
           "Sinergia dirigida S_fd"]


def senuelos(sensor, n=2):
    """Multiplicadores c de fr tales que c, 2c y 3c quedan lejos (>2.8 %) de toda línea cinemática conocida:
    armónicos de fr (1..39), armónicas 1..6 de BPFI, BPFO, BSF, 2BSF y FTF, y bandas laterales
    h·BPFI ± k·fr, h·2BSF ± FTF. (Las armónicas densas de FTF impiden un señuelo 100 % limpio.)"""
    c = F.MULTIPLICADORES[sensor]
    L = list(range(1, 40))
    for fq in (c["BPFI"], c["BPFO"], c["BSF"], 2 * c["BSF"], c["FTF"]):
        L += [h * fq for h in range(1, 7)]
    L += [h * c["BPFI"] + s * k for h in (1, 2, 3) for k in (1, 2) for s in (-1, 1)]
    L += [h * 2 * c["BSF"] + s * c["FTF"] for h in (1, 2, 3) for s in (-1, 1)]
    L = np.array(L)
    cand = np.arange(2.3, 6.0, 0.005)
    score = np.array([min(np.min(np.abs(h * x - L)) / (h * x) for h in (1, 2, 3)) for x in cand])
    out = []
    for _ in range(n):
        i = int(np.argmax(score)); out.append(round(float(cand[i]), 3)); score[np.abs(cand - cand[i]) < 0.5] = -1
    return out


def bandas_ciegas(x, fs, L, seed):
    T = SF.MDFDTree(x, fs, L)
    K = SF.kurtogram(T)
    syn = SF.synergy_map(T, n_surr=12, rng=np.random.default_rng(seed))
    sel = {"Kurtogram (Tarea 1)": SF.argmax_node(K),
           "Sinergia Z": SF.argmax_node({k: v["Z"] for k, v in syn.items()}),
           "Descenso por sinergia": SF.synergy_descent(T, syn, 3.0)[0],
           "Nodo raíz (SES crudo)": (0, 0)}
    return T, sel


def banda_dirigida(T, fd):
    sfd = {(l, m): SF.targeted_synergy(T.z(l + 1, 2 * m), T.z(l + 1, 2 * m + 1), T.fs, fd)
           for l in range(T.L) for m in range(2 ** l)}
    return SF.argmax_node(sfd)


def evaluar(x, fs, nodo, f_obj, fr):
    xf = x - x.mean() if nodo == (0, 0) else F.filtrar(x, fs, *nodo)
    return F.evaluar_diagnostico(*F.envolvente_espectral(xf, fs), f_obj, fr)["categoria"]


def correr():
    idx = F.leer_indice()
    idx = idx[idx.carpeta.isin(["12k_DE", "12k_FE"])]
    filas = []
    for _, r in idx.iterrows():
        reg = F.cargar_registro(int(r.registro))
        x, fs, fd, fr = reg["x"], reg["fs"], reg["fd"], reg["fr"]
        L = F.nivel_maximo(fs, fd)
        T, sel = bandas_ciegas(x, fs, L, int(r.registro))
        freqs = {"fd": fd}
        for j, c in enumerate(senuelos(reg["sensor_falla"])):
            freqs[f"senuelo{j}"] = c * fr
        fila = dict(registro=int(r.registro), archivo=r.nombre, sensor=reg["sensor_falla"], zona=reg["zona"],
                    L=L, bench=BENCH.get(int(r.registro), "?"))
        for tag, f in freqs.items():
            sel_t = dict(sel)
            sel_t["Sinergia dirigida S_fd"] = banda_dirigida(T, f)
            for met, nodo in sel_t.items():
                fila[f"{met}|{tag}"] = evaluar(x, fs, tuple(nodo), f, fr)
                if tag == "fd":
                    fl, fu = F.bordes_banda(*nodo, fs)
                    fila[f"{met}|banda"] = f"{fl:.0f}-{fu:.0f}"
        filas.append(fila)
        print(r.registro, r.nombre, {m: fila[f"{m}|fd"] for m in METODOS}, flush=True)
    df = pd.DataFrame(filas)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, "comparacion_12k.csv"), index=False)
    return df


def resumir(df, METODOS=METODOS, RES=RES):
    filas = []
    for m in METODOS:
        cat = df[f"{m}|fd"]
        fa = pd.concat([df[f"{m}|senuelo0"], df[f"{m}|senuelo1"]])
        tpr_d, fpr_d = (cat == "D").mean(), (fa == "D").mean()
        tpr, fpr = cat.isin(["D", "P"]).mean(), fa.isin(["D", "P"]).mean()
        fila = {"Método": m, "D (%)": 100 * tpr_d, "P (%)": 100 * (cat == "P").mean(), "N (%)": 100 * (cat == "N").mean(),
                "D en señuelos (%)": 100 * fpr_d, "J_D = TPR_D - FPR_D": tpr_d - fpr_d,
                "D+P (%)": 100 * tpr, "D+P en señuelos (%)": 100 * fpr, "J_DP": tpr - fpr}
        for g in ("Y", "P", "N"):
            sub = df[df.bench == g]
            fila[f"D en cat. S&R {g} (%)"] = 100 * (sub[f"{m}|fd"] == "D").mean()
        filas.append(fila)
    res = pd.DataFrame(filas).round(2)
    res.to_csv(os.path.join(RES, "resumen.csv"), index=False)
    return res


def graficar(res, df, METODOS=METODOS, RES=RES):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"D": "#1d5fb0", "P": "#8fb5e6", "N": "#e4e3df"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]
    y = np.arange(len(METODOS))[::-1]
    izq = np.zeros(len(METODOS))
    for c, lab in (("D", "Diagnosticable"), ("P", "Parcialmente"), ("N", "No diagnosticable")):
        v = res[f"{c} (%)"].values
        ax.barh(y, v, left=izq, color=col[c], edgecolor="white", label=lab)
        for yi, vi, li in zip(y, v, izq):
            if vi >= 6:
                ax.text(li + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if c == "D" else "#0b0b0b")
        izq += v
    ax.set_yticks(y); ax.set_yticklabels(METODOS, fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel(f"% de los {len(df)} registros (12 kHz, DE y FE)")
    ax.set_title("Resultado en la fd verdadera", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.07))
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1]
    ax.barh(y + 0.18, res["D (%)"], height=0.36, color="#3d3d3a", label="diagnosticable (D) en fd verdadera")
    ax.barh(y - 0.18, res["D en señuelos (%)"], height=0.36, color="#c9c7bf", label="D en frecuencias señuelo")
    for yi, a, b, j in zip(y, res["D (%)"], res["D en señuelos (%)"], res["J_D = TPR_D - FPR_D"]):
        ax.text(a + 1, yi + 0.18, f"{a:.0f}", va="center", fontsize=7.5)
        ax.text(b + 1, yi - 0.18, f"{b:.0f}   J={j:.2f}", va="center", fontsize=7.5, color="#52514e")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlim(0, 100); ax.set_xlabel("%")
    ax.set_title("D: fd verdadera vs. señuelos", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(RES, "comparacion_resumen.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RES, "comparacion_12k.csv")) if "--solo-resumen" in sys.argv else correr()
    res = resumir(df)
    graficar(res, df)
    print(res.to_string(index=False))
