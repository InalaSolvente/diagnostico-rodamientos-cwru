"""
comparacion_descenso.py — Método de la Tarea (kurtogram) vs. descenso por sinergia, registro a registro,
con las armónicas de fd mostradas de forma que se vea POR QUÉ el criterio las reconoce.

Figura detallada de cada registro (figuras/comparacion_descenso/<n>.png):
  (a) kurtogram MFD con las dos bandas marcadas (kurtogram y descenso por sinergia) y el camino del descenso;
  (b) para cada banda, la envolvente espectral RELATIVA AL FONDO LOCAL, R(α) = S_env(α) / fondo(α), en dB.
      El fondo es la media de S_env en α ± 0.3 fd sin la ventana ±2 % (el mismo fondo del criterio), así que
      la línea del umbral (p = 1 %) es horizontal y todo pico que la cruza es una línea "real". Los picos que
      cruzan el umbral se colorean según a qué familia cinemática pertenecen por su POSICIÓN:
      armónica de fd, banda lateral h·fd ± fr (pista interna) o h·fd ± FTF (bola), armónico de la velocidad
      de giro n·fr, combinación giro–jaula m·fr ± n·FTF, u otra. (La clasificación es solo para leer la
      figura; el diagnóstico D/P/N usa únicamente las ventanas en h·fd.)
  (c) para cada banda, un zoom a cada armónica h·fd (h = 1, 2, 3) en escala lineal: ventana de búsqueda
      ±2 %, zona de fondo ±0.3 fd, media del fondo, umbral k·media y el pico encontrado con su altura.

Galería (figuras/comparacion_descenso/galeria_<i>.png): los registros donde el descenso logra D y el
kurtogram no, una fila por registro (kurtogram | descenso), en la misma escala relativa al fondo.

    python comparacion_descenso.py                 # figuras del informe + galería
    python comparacion_descenso.py 105 130         # figuras detalladas de otros registros (sin galería)

El descenso por sinergia está en ../02_Sinergia_Fraterna/codigo/mdfd_sinergia.py:
parte en la raíz; si el nodo es "sinérgico" (Z >= 3: sus dos hijos están co-modulados, la banda es
indivisible) se detiene; si no, baja al hijo que aporta más curtosis heredada (w^2 K).
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "02_Sinergia_Fraterna", "codigo"))
import Fuentes as F                       # noqa: E402
import mdfd_sinergia as SF                # noqa: E402

SALIDA = os.path.join(AQUI, "figuras", "comparacion_descenso")
C_K, C_D, TINTA, TINTA2, GRILLA = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
C_FR, C_LAT, C_FTF, C_OTRA = "#2e9e5b", "#9b59b6", "#d99a1e", "#b4b2aa"
TOL, ANCHO_FONDO, P = 0.02, 0.3, 0.01           # los mismos parámetros de Fuentes.evaluar_diagnostico

# Registros donde el descenso logra D y el kurtogram no (03_Comparacion, comparacion_12k.csv)
GANA_DESCENSO = [49, 50, 51, 106, 107, 108, 170, 209, 212, 156, 158, 147, 234, 282, 280, 281, 297, 315]
DETALLE = [107, 108, 147, 170, 280, 297, 287, 282, 222]


# ---------------------------------------------------------------------------
# Bandas y señales
# ---------------------------------------------------------------------------
def bandas(reg):
    x, fs, fd = reg["x"], reg["fs"], reg["fd"]
    L = F.nivel_maximo(fs, fd)
    tabla = F.kurtogram(x, fs, L)
    nk = F.seleccionar_banda(tabla)
    T = SF.MDFDTree(x, fs, L)
    syn = SF.synergy_map(T, n_surr=12, rng=np.random.default_rng(reg["registro"]))
    nd, camino = SF.synergy_descent(T, syn, 3.0)
    return L, tabla, nk, nd, camino, syn


def analizar(reg):
    x, fs, fd, fr = reg["x"], reg["fs"], reg["fd"], reg["fr"]
    L, tabla, nk, nd, camino, syn = bandas(reg)
    out = dict(L=L, tabla=tabla, camino=camino, syn=syn)
    for tag, nodo in (("k", nk), ("d", nd)):
        xf = x - x.mean() if nodo == (0, 0) else F.filtrar(x, fs, *nodo)
        a, S = F.envolvente_espectral(xf, fs)
        out[tag] = dict(nodo=nodo, banda=F.bordes_banda(*nodo, fs), a=a, S=S,
                        diag=F.evaluar_diagnostico(a, S, fd, fr), fondo=fondo_local(a, S, fd))
    return out


# ---------------------------------------------------------------------------
# Fondo local y clasificación de picos por familia cinemática
# ---------------------------------------------------------------------------
def fondo_local(a, S, fd, tol=TOL, ancho=ANCHO_FONDO):
    """Media de S en |α' - α| <= ancho·fd excluyendo |α' - α| <= tol·α (igual que el criterio, pero en cada α)."""
    cs = np.concatenate([[0.0], np.cumsum(S)])
    lo_b, hi_b = np.searchsorted(a, a - ancho * fd), np.searchsorted(a, a + ancho * fd, side="right")
    lo_s, hi_s = np.searchsorted(a, a - tol * a), np.searchsorted(a, a + tol * a, side="right")
    n = (hi_b - lo_b) - (hi_s - lo_s)
    s = (cs[hi_b] - cs[lo_b]) - (cs[hi_s] - cs[lo_s])
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def umbral(a, fd, h, tol=TOL, p=P):
    """k del criterio para la armónica h (depende del número M de bins de la ventana)."""
    M = int((np.abs(a - h * fd) <= tol * h * fd).sum())
    return -np.log(1 - (1 - p) ** (1 / M))


def familias(reg, fmax):
    """Líneas cinemáticas esperadas hasta fmax: [(frecuencia, familia, etiqueta)]."""
    fd, fr, zona = reg["fd"], reg["fr"], reg["zona"]
    ftf = F.MULTIPLICADORES[reg["sensor_falla"]]["FTF"] * fr
    L = [(h * fd, "fd", f"{h}×fd") for h in range(1, int(fmax / fd) + 1)]
    if zona == "IR":      # la carga modula el defecto al girar con el eje: bandas h·fd ± k·fr
        L += [(h * fd + s * k * fr, "lat", f"{h}fd{'+' if s > 0 else '−'}{k if k > 1 else ''}fr")
              for h in range(1, int(fmax / fd) + 1) for k in (1, 2) for s in (-1, 1)]
    elif zona == "B":     # la bola gira con la jaula: bandas h·fd ± FTF
        L += [(h * fd + s * ftf, "lat", f"{h}fd{'+' if s > 0 else '−'}FTF")
              for h in range(1, int(fmax / fd) + 1) for s in (-1, 1)]
    L += [(n * fr, "fr", f"{n}fr") for n in range(1, int(fmax / fr) + 1)]
    # combinaciones de giro y jaula m·fr ± n·FTF (aparecen con holgura/deslizamiento de la jaula)
    L += [(m * fr + n * ftf, "ftf", f"{m}fr{'+' if n > 0 else '−'}{abs(n)}FTF")
          for m in range(0, int(fmax / fr) + 2) for n in (-2, -1, 1, 2)]
    return [z for z in L if 0 < z[0] <= fmax]


def picos_sobre_umbral(a, R, thr, sep=1.0):
    """Máximos locales de R que superan thr, fusionando los que están a menos de `sep` Hz."""
    i = np.where((R[1:-1] >= R[:-2]) & (R[1:-1] >= R[2:]) & (R[1:-1] > thr))[0] + 1
    i = i[np.argsort(-R[i])]
    out = []
    for j in i:
        if all(abs(a[j] - a[q]) > sep for q in out):
            out.append(j)
    return sorted(out)


def clasificar(f, reg, lineas):
    """Familia de un pico por su posición: fd (±2 %), luego bandas laterales y n·fr (±1 %)."""
    fd = reg["fd"]
    h = round(f / fd)
    if h >= 1 and abs(f - h * fd) <= TOL * h * fd:
        return "fd"
    for fam, tol_rel, tol_abs in (("lat", 0.01, 0.5), ("fr", 0.01, 0.5), ("ftf", 0.003, 0.25)):
        if any(abs(f - fl) <= max(tol_rel * fl, tol_abs) for fl, ff, _ in lineas if ff == fam):
            return fam
    return "otra"


# ---------------------------------------------------------------------------
# Paneles
# ---------------------------------------------------------------------------
def panel_mapa(ax, reg, info):
    fs, L, tabla = reg["fs"], info["L"], info["tabla"]
    img = np.full((L + 1, 2 ** L), np.nan)
    for _, r in tabla.iterrows():
        w = 2 ** (L - int(r.l)); img[int(r.l), int(r.m) * w:(int(r.m) + 1) * w] = r.K
    im = ax.imshow(img, aspect="auto", cmap="Greys", interpolation="nearest", extent=[0, fs / 2, L + 0.5, -0.5])
    for l in range(1, L + 1):
        for m in range(1, 2 ** l):
            ax.plot([m * fs / 2 ** (l + 1)] * 2, [l - 0.5, l + 0.5], color="white", lw=0.5)
    for (l, m) in info["camino"][:-1]:
        fl, fu = F.bordes_banda(l, m, fs)
        ax.add_patch(plt.Rectangle((fl, l - 0.5), fu - fl, 1, fill=False, ec=C_D, lw=1.0, ls=(0, (2, 2))))
    for tag, c, lab in (("k", C_K, "kurtogram"), ("d", C_D, "descenso por sinergia")):
        (l, m), (fl, fu) = info[tag]["nodo"], info[tag]["banda"]
        ax.add_patch(plt.Rectangle((fl, l - 0.5), fu - fl, 1, fill=False, ec=c, lw=2.4, label=lab))
    ax.set_yticks(range(L + 1)); ax.set_ylabel("nivel ℓ"); ax.set_xlabel("frecuencia [Hz]", labelpad=1)
    ax.plot([], [], color=C_D, lw=1.0, ls=(0, (2, 2)), label="camino del descenso")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    zona = {"IR": "pista interna", "OR": "pista externa", "B": "bola"}[reg["zona"]]
    ax.set_title(f"Registro {reg['registro']} ({reg['sensor_falla']}, {zona}, {reg['tamano_in']:.3f} in, "
                 f"{reg['carga_hp']} hp; fd = {reg['fd']:.1f} Hz, fr = {reg['fr']:.2f} Hz): kurtogram MFD",
                 loc="left", fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("K")


def panel_relativo(ax, reg, b, color, titulo, fmax, leyenda=True, compacto=False):
    """R(α) = S/fondo en dB, umbral horizontal, picos sobre el umbral coloreados por familia."""
    fd = reg["fd"]
    a, S, fondo, diag = b["a"], b["S"], b["fondo"], b["diag"]
    sel = (a > 1.0) & (a <= fmax) & np.isfinite(fondo)
    aa, R = a[sel], 10 * np.log10(S[sel] / fondo[sel])
    thr = 10 * np.log10(umbral(a, fd, 1))
    ax.axvspan(0.5 * fd, 3.5 * fd, color="#f4f3ef", zorder=0)
    ax.plot(aa, R, color=TINTA2, lw=0.45, zorder=2)
    ax.axhline(thr, color=TINTA, lw=0.8, ls=(0, (4, 2)), zorder=3)
    if not compacto:
        ax.text(fmax, thr, f" umbral p=1 % ({thr:.1f} dB)", fontsize=6.5, va="bottom", ha="right", color=TINTA,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    lineas = familias(reg, fmax)
    col = {"fd": color, "lat": C_LAT, "fr": C_FR, "ftf": C_FTF, "otra": C_OTRA}
    for j in picos_sobre_umbral(aa, R, thr):
        fam = clasificar(aa[j], reg, lineas)
        if fam != "fd":       # las armónicas de fd se marcan abajo con el pico que eligió el criterio
            ax.plot(aa[j], R[j], "o", ms=3.2, color=col[fam], mec="white", mew=0.4, zorder=5)
    for z in diag["armonicas"]:
        if z["significativa"]:
            j = np.argmin(np.abs(aa - z["f_pico"]))
            ax.plot(aa[j], R[j], "o", ms=4.6, color=color, mec="white", mew=0.5, zorder=6)
    for h in range(1, int(fmax / fd) + 1):
        z = next((q for q in diag["armonicas"] if q["h"] == h), None)
        ok = bool(z and z["significativa"])
        ax.axvline(h * fd, color=color, lw=0.8 if ok else 0.5, ls=(0, (3, 2)), alpha=0.9 if ok else 0.5, zorder=1)
        ax.text(h * fd, 1.0, f"{h}×fd" + (" ✓" if ok else ""), transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7 if not compacto else 6.5, color=TINTA if ok else TINTA2,
                fontweight="bold" if ok else "normal")
    g = diag["pico_global_hz"]
    ig = np.argmin(np.abs(aa - g))
    ax.plot(g, R[ig], marker="*", ms=10 if not compacto else 8, mfc="none", mec=TINTA, mew=0.9, zorder=6)
    ymax = max(np.nanmax(R) + 3, thr + 8)
    ax.set_xlim(0, fmax); ax.set_ylim(-12, ymax)
    ax.set_ylabel("S / fondo [dB]", fontsize=7.5 if compacto else 8.5)
    ax.set_title(titulo, loc="left", fontsize=8.5 if not compacto else 7.5, fontweight="bold",
                 pad=12 if not compacto else 10, color=TINTA)
    if compacto:
        ax.set_title(f"{titulo}  →  {diag['categoria']}", loc="left", fontsize=7.5, fontweight="bold", pad=10)
    else:
        ax.text(0.995, 0.97, F.NOMBRE_CATEGORIA[diag["categoria"]], transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, fontweight="bold", color=TINTA,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.0))
    ax.yaxis.grid(True, color=GRILLA, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    if leyenda:
        hs = [Line2D([], [], marker="o", ls="", color=color, mec="white", label="armónica de fd"),
              Line2D([], [], marker="o", ls="", color=C_LAT,
                     label={"IR": "banda lateral h·fd ± k·fr", "B": "banda lateral h·fd ± FTF"}.get(reg["zona"],
                                                                                                 "banda lateral")),
              Line2D([], [], marker="o", ls="", color=C_FR, label="armónico de giro n·fr"),
              Line2D([], [], marker="o", ls="", color=C_FTF, label="giro–jaula m·fr ± n·FTF"),
              Line2D([], [], marker="o", ls="", color=C_OTRA, label="otra línea"),
              Line2D([], [], marker="*", ls="", mfc="none", mec=TINTA, ms=9, label="pico más alto de S_env en [0.5, 3.5]·fd")]
        if reg["zona"] == "OR":
            hs.pop(1)
        ax.legend(handles=hs, fontsize=6.5, ncol=6, loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False,
                  handletextpad=0.2, columnspacing=0.9)


def panel_zoom(ax, reg, b, h, color):
    """Zoom lineal a h·fd: ventana ±2 %, fondo ±0.3 fd, media del fondo, umbral y pico encontrado."""
    fd = reg["fd"]
    a, S, diag = b["a"], b["S"], b["diag"]
    z = next((q for q in diag["armonicas"] if q["h"] == h), None)
    d = np.abs(a - h * fd)
    win, loc = d <= TOL * h * fd, (d <= ANCHO_FONDO * fd) & ~(d <= TOL * h * fd)
    media = S[loc].mean()
    k = umbral(a, fd, h)
    vis = d <= 0.36 * fd
    Y = S[vis] / media
    ax.axvspan(h * fd - ANCHO_FONDO * fd, h * fd + ANCHO_FONDO * fd, color="#f1f0ec", zorder=0)
    ax.axvspan(h * fd - TOL * h * fd, h * fd + TOL * h * fd, color=color, alpha=0.18, lw=0, zorder=1)
    ax.plot(a[vis], Y, color=TINTA2, lw=0.6, zorder=2)
    ax.axhline(1.0, color=TINTA2, lw=0.7, zorder=3)
    ax.axhline(k, color=TINTA, lw=0.8, ls=(0, (4, 2)), zorder=3)
    ymax = max(np.nanmax(Y), k) * 1.55
    for fl, fam, lab in familias(reg, (h + 0.4) * fd):
        if fam in ("fr", "lat") and abs(fl - h * fd) <= 0.34 * fd:
            c = C_FR if fam == "fr" else C_LAT
            ax.plot(fl, ymax * 0.985, marker="v", ms=4, color=c, zorder=4, clip_on=False)
    if z:
        yp = 10 ** (z["ratio_db"] / 10)
        ok = z["significativa"]
        ax.plot(z["f_pico"], yp, "o", ms=5.5, color=color if ok else "white", mec=color, mew=1.2, zorder=6)
        veces = f"{yp:.0f}" if yp >= 20 else f"{yp:.1f}"
        ax.annotate(f"{veces}× fondo\n({z['ratio_db']:.1f} dB) {'✓' if ok else '✗'}",
                    xy=(z["f_pico"], yp), xytext=(0.03, 0.93), textcoords="axes fraction", fontsize=6.8,
                    va="top", ha="left", color=TINTA, fontweight="bold" if ok else "normal",
                    arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.5))
    ax.text(0.98, k / ymax, f"umbral\n{k:.1f}×", transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            color=TINTA)
    ax.set_xlim(h * fd - 0.36 * fd, h * fd + 0.36 * fd); ax.set_ylim(0, ymax)
    ax.set_xticks([h * fd - 0.3 * fd, h * fd, h * fd + 0.3 * fd])
    ax.set_xticklabels([f"−0.3fd", f"{h}×fd\n{h * fd:.0f} Hz", "+0.3fd"], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    if h == 1:
        ax.set_ylabel("S / fondo (lineal)", fontsize=7.5)


def leyenda_familias(zona=None):
    hs = [Line2D([], [], marker="o", ls="", color=C_K, mec="white", label="armónica de fd (banda del kurtogram)"),
          Line2D([], [], marker="o", ls="", color=C_D, mec="white", label="armónica de fd (banda del descenso)"),
          Line2D([], [], marker="o", ls="", color=C_LAT,
                 label={"IR": "banda lateral h·fd ± k·fr", "B": "banda lateral h·fd ± FTF"}.get(zona, "banda lateral")),
          Line2D([], [], marker="o", ls="", color=C_FR, label="armónico de giro n·fr"),
          Line2D([], [], marker="o", ls="", color=C_FTF, label="giro–jaula m·fr ± n·FTF"),
          Line2D([], [], marker="o", ls="", color=C_OTRA, label="otra línea"),
          Line2D([], [], marker="*", ls="", mfc="none", mec=TINTA, ms=9, label="pico más alto de S_env en [0.5, 3.5]·fd"),
          Line2D([], [], color=TINTA, lw=0.8, ls=(0, (4, 2)), label="umbral p = 1 %")]
    if zona == "OR":
        hs.pop(2)
    return hs


def figura(numero):
    reg = F.cargar_registro(numero)
    info = analizar(reg)
    fd = reg["fd"]
    fmax = 4.0 * fd
    fig = plt.figure(figsize=(8.3, 12.2), constrained_layout=True)
    gs = fig.add_gridspec(5, 3, height_ratios=[0.85, 1.0, 0.8, 1.0, 0.8])
    panel_mapa(fig.add_subplot(gs[0, :]), reg, info)
    for fila, (tag, c, nombre) in ((1, ("k", C_K, "Método de la Tarea (kurtogram)")),
                                   (3, ("d", C_D, "Descenso por sinergia"))):
        b = info[tag]
        (l, m), (fl, fu) = b["nodo"], b["banda"]
        panel_relativo(fig.add_subplot(gs[fila, :]), reg, b, c,
                       f"{nombre}: banda {fl:.0f}–{fu:.0f} Hz (ℓ={l}, m={m}) — envolvente relativa al fondo local",
                       fmax, leyenda=False)
        for h in (1, 2, 3):
            panel_zoom(fig.add_subplot(gs[fila + 1, h - 1]), reg, b, h, c)
    fig.legend(handles=leyenda_familias(reg["zona"]), loc="outside lower center", ncol=4, fontsize=7,
               frameon=False, handletextpad=0.2, columnspacing=1.0)
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, f"{numero}.png")
    fig.savefig(ruta, dpi=160); plt.close(fig)
    return ruta, resumen(numero, reg, info)


def resumen(numero, reg, info):
    def r(b):
        return dict(nodo=b["nodo"], banda=b["banda"], cat=b["diag"]["categoria"], db_fd=b["diag"]["ratio_db_fd"],
                    pico_global_hz=b["diag"]["pico_global_hz"],
                    armonicas=[{k: z[k] for k in ("h", "f_pico", "ratio_db", "umbral_db", "significativa")}
                               for z in b["diag"]["armonicas"]])
    out = dict(registro=numero, fd=reg["fd"], fr=reg["fr"], kurtogram=r(info["k"]), descenso=r(info["d"]))
    out["descenso"]["camino"] = info["camino"]
    out["descenso"]["Z_camino"] = [round(float(info["syn"][n]["Z"]), 1) for n in info["camino"] if n in info["syn"]]
    return out


def galeria(registros, por_pagina=9):
    rutas = []
    for p in range(0, len(registros), por_pagina):
        grupo = registros[p:p + por_pagina]
        fig, axs = plt.subplots(len(grupo), 2, figsize=(8.3, 1.25 * len(grupo) + 0.6), constrained_layout=True,
                                squeeze=False)
        for i, n in enumerate(grupo):
            reg = F.cargar_registro(n)
            info = analizar(reg)
            zona = {"IR": "IR", "OR": "OR", "B": "bola"}[reg["zona"]]
            for j, (tag, c, nom) in enumerate((("k", C_K, "kurtogram"), ("d", C_D, "descenso"))):
                b = info[tag]
                fl, fu = b["banda"]
                panel_relativo(axs[i, j], reg, b, c,
                               f"{n} ({reg['sensor_falla']} {zona} {reg['tamano_in']:.3f}\"): {nom} {fl:.0f}–{fu:.0f} Hz",
                               3.6 * reg["fd"], leyenda=False, compacto=True)
                if j == 1:
                    axs[i, j].set_ylabel("")
                if i < len(grupo) - 1:
                    axs[i, j].set_xticklabels([])
            print("galería", n, info["k"]["diag"]["categoria"], info["d"]["diag"]["categoria"], flush=True)
        for ax in axs[-1]:
            ax.set_xlabel("frecuencia cíclica α [Hz]", fontsize=7.5)
        hs = leyenda_familias()
        fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False,
                   handletextpad=0.2, columnspacing=0.8)
        ruta = os.path.join(SALIDA, f"galeria_{p // por_pagina + 1}.png")
        fig.savefig(ruta, dpi=160); plt.close(fig)
        rutas.append(ruta)
    return rutas


def _json(o):
    return o.item() if isinstance(o, np.generic) else str(o)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    regs = args or DETALLE
    os.makedirs(SALIDA, exist_ok=True)
    ruta_res = os.path.join(SALIDA, "resumen.json")
    out = {}
    if os.path.exists(ruta_res):
        with open(ruta_res, encoding="utf-8") as fh:
            out = json.load(fh)
    for n in regs:
        ruta, res = figura(n)
        out[str(n)] = res
        print(ruta, res["kurtogram"]["cat"], "->", res["descenso"]["cat"], flush=True)
    with open(ruta_res, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=_json)
    if not args:
        print(galeria(GANA_DESCENSO))
