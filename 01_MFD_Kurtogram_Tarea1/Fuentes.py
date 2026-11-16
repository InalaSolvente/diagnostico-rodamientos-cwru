"""
Fuentes.py — Tarea 1: Multiscale Fourier Decomposition (MFD) + Envolvente Espectral
Detección de Anomalías en Sistemas Críticos (PUCV) — Prof. Nibaldo Rodríguez A.
Autor: Cristóbal Martínez Arochas

Contenido (lo que pide la tarea en "Entregables: Python"):
    1. MFD           : descomposición diádica por máscaras FFT/IFFT y kurtogram (curtosis compleja).
    2. Envolvente    : envolvente espectral S_env[k] = |FFT{a~[n]}|^2  (definición de clase).
    3. Gráficas      : señal, espectro y envolvente espectral (cruda y filtrada) + mapa del kurtogram.
    4. Diagnóstico   : prueba de significancia sobre la envolvente espectral para clasificar
                       Diagnosticable / Parcialmente diagnosticable / No diagnosticable.

Núcleo numérico sólo con NumPy (Python 3.12 / NumPy / Pandas, como pide el curso);
SciPy se usa únicamente para leer los archivos .mat del CWRU.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. Datos CWRU
# ---------------------------------------------------------------------------
# Multiplicadores cinemáticos (x fr) — Smith & Randall (2015), Tabla 2
MULTIPLICADORES = {
    "DE": dict(BPFI=5.4152, BPFO=3.5848, FTF=0.39828, BSF=2.3568),   # SKF 6205-2RS JEM
    "FE": dict(BPFI=4.9469, BPFO=3.0530, FTF=0.3817, BSF=1.9940),    # SKF 6203-2RS JEM
}
ZONAS = {"IR": "Pista interna (inner race)", "OR": "Pista externa (outer race)",
         "B": "Elemento rodante (ball)", "N": "Sin falla (normal)"}

RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_DATOS = os.path.join(RAIZ_PROYECTO, "data", "CWRU")


def leer_indice(ruta_datos: str = RUTA_DATOS) -> pd.DataFrame:
    """Índice de registros (número de archivo CWRU, carpeta, falla, tamaño, carga)."""
    verificado = os.path.join(ruta_datos, "indice_canal_verificado.csv")
    return pd.read_csv(verificado if os.path.isfile(verificado) else os.path.join(ruta_datos, "indice_cwru.csv"))


def cargar_registro(numero: int, sensor: str | None = None, ruta_datos: str = RUTA_DATOS, *, archivo: str | None = None):
    """Carga un registro CWRU por su número (p. ej. 282).

    Devuelve dict con x (señal), fs, rpm, fr, sensor, zona, tamaño, carga y fd (frecuencia de falla).
    Si sensor es None se usa el rodamiento con falla (DE para carpetas *_DE, FE para 12k_FE).
    """
    from scipy.io import loadmat
    idx = leer_indice(ruta_datos)
    fila = idx[idx.registro == numero]
    if archivo is not None:
        fila = fila[fila.archivo == archivo]
    if fila.empty:
        raise ValueError(f"Registro {numero} no está en el índice")
    if len(fila) != 1:
        raise ValueError(f'Registro {numero} ambiguo entre archivos; indique archivo explícitamente')
    fila = fila.iloc[0]
    d = loadmat(os.path.join(ruta_datos, fila.archivo))
    sensor = sensor or fila.sensor_falla
    clave = clave_canal(d, sensor, int(fila.registro))
    x = np.ravel(d[clave]).astype(float)
    rpm = rpm_registro(d, int(fila.registro), nominal=float(fila.rpm_nominal))
    fr = rpm / 60.0
    zona = fila.zona
    fd = frecuencia_falla(zona, fila.sensor_falla, fr)
    return dict(x=x, fs=float(fila.fs), rpm=rpm, fr=fr, sensor=sensor, zona=zona,
                tamano_in=fila.tamano_in, carga_hp=int(fila.carga_hp), registro=int(numero),
                fd=fd, archivo=fila.archivo, sensor_falla=fila.sensor_falla)


def frecuencia_falla(zona: str, sensor_falla: str, fr: float) -> float | None:
    """BPFI (IR), BPFO (OR) o 2·BSF (bola, criterio usado en clase para el registro 282)."""
    c = MULTIPLICADORES[sensor_falla]
    return {"IR": c["BPFI"] * fr, "OR": c["BPFO"] * fr, "B": 2 * c["BSF"] * fr}.get(zona)


# ---------------------------------------------------------------------------
# 1. MFD: segmentación diádica, máscaras y sub-señales (diapositivas 4-11)
# ---------------------------------------------------------------------------

def bordes_banda(l: int, m: int, fs: float):
    """B_{l,m} = [m Δf_l, (m+1) Δf_l),  Δf_l = Fs / 2^(l+1)."""
    df = fs / 2 ** (l + 1)
    return m * df, (m + 1) * df


def frecuencia_central(l: int, m: int, fs: float) -> float:
    """f^c_{l,m} = (2m+1) Fs / 2^(l+2)."""
    return (2 * m + 1) * fs / 2 ** (l + 2)


def mascara(N: int, fs: float, l: int, m: int) -> np.ndarray:
    """H_{l,m}[k] = 1 si f^L <= |f_k| < f^U (la última banda incluye Fs/2); simétrica => sub-señal real."""
    f = np.abs(np.fft.fftfreq(N, d=1 / fs))
    fl, fu = bordes_banda(l, m, fs)
    if m == 2 ** l - 1:
        return ((f >= fl) & (f <= fu)).astype(float)
    return ((f >= fl) & (f < fu)).astype(float)


def subsenal(X: np.ndarray, fs: float, l: int, m: int) -> np.ndarray:
    """x_{l,m}[n] = IFFT{ H_{l,m}[k] X[k] } (la FFT X se calcula una sola vez)."""
    return np.real(np.fft.ifft(mascara(X.size, fs, l, m) * X))


def mfd_nivel(x: np.ndarray, fs: float, l: int) -> list[np.ndarray]:
    """D_l = {x_{l,m} : m = 0..2^l - 1}; su suma reconstruye x."""
    X = np.fft.fft(x)
    return [subsenal(X, fs, l, m) for m in range(2 ** l)]


# ---------------------------------------------------------------------------
# 2. Señal analítica, curtosis compleja y kurtogram (diapositivas 15-21)
# ---------------------------------------------------------------------------

def senal_analitica(x: np.ndarray) -> np.ndarray:
    """z = x + jH{x} vía FFT: Z[k] = X[k]·h[k], h = 1 (k=0, N/2), 2 (0<k<N/2), 0 (k>N/2)."""
    N = x.size
    X = np.fft.fft(x)
    h = np.zeros(N)
    h[0] = 1
    if N % 2 == 0:
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[1:(N + 1) // 2] = 2
    return np.fft.ifft(X * h)


def curtosis_compleja(z: np.ndarray) -> float:
    """K = mu4/mu2^2 - 2 con z centrada (referencia gaussiana compleja circular = 0)."""
    zc = z - z.mean()
    e = np.abs(zc) ** 2
    mu2, mu4 = e.mean(), (e ** 2).mean()
    return float(mu4 / mu2 ** 2 - 2) if mu2 > 0 else 0.0


def entropia_envolvente(z: np.ndarray) -> float:
    """Shannon del espectro de la envolvente sin DC, normalizada a [0, 1].

    Se usa el mismo soporte (todos los bins positivos) para todas las bandas.
    Envolvente constante/sin energía: H=1, sin recompensa por concentración.
    """
    a = np.abs(z)
    potencia = np.abs(np.fft.rfft(a - a.mean()))[1:] ** 2
    if potencia.size < 2 or potencia.sum() <= np.finfo(float).eps * np.sum(a ** 2):
        return 1.0
    p = potencia / potencia.sum()
    p = p[p > 0]
    return float(np.clip(-np.sum(p * np.log(p)) / np.log(potencia.size), 0, 1))


def kurtogram(x: np.ndarray, fs: float, L: int, con_entropia: bool = False) -> pd.DataFrame:
    """Kurtogram MFD: para l=0..L y m=0..2^l-1 calcula K_{l,m}. Devuelve tabla (l, m, fL, fU, fc, df, K)."""
    X = np.fft.fft(x - x.mean())
    filas = []
    for l in range(L + 1):
        for m in range(2 ** l):
            z = senal_analitica(subsenal(X, fs, l, m))
            fl, fu = bordes_banda(l, m, fs)
            filas.append(dict(l=l, m=m, fL=fl, fU=fu, fc=0.5 * (fl + fu), df=fu - fl,
                              K=curtosis_compleja(z)))
            if con_entropia:
                filas[-1]["H"] = entropia_envolvente(z)
    return pd.DataFrame(filas)


def puntuacion_bandas(tabla_K: pd.DataFrame, peso_k: float = 1.0) -> pd.Series:
    """S = peso_k * minmax(K) + (1-peso_k) * (1-H), por registro.

    Curtosis constante aporta cero. Los empates conservan el primer nodo.
    peso_k=1 conserva la selección original sin requerir H.
    """
    if not np.isfinite(peso_k) or not 0 <= peso_k <= 1:
        raise ValueError("peso_k debe estar entre 0 y 1")
    if tabla_K.empty or not np.isfinite(tabla_K.K).all():
        raise ValueError("Se requiere una tabla no vacía con K finita")
    if peso_k == 1:
        return tabla_K.K.copy()
    if "H" not in tabla_K or not (np.isfinite(tabla_K.H) & tabla_K.H.between(0, 1)).all():
        raise ValueError("Se requiere H finita en [0, 1]; use con_entropia=True")
    rango = tabla_K.K.max() - tabla_K.K.min()
    kn = (tabla_K.K - tabla_K.K.min()) / rango if rango > 0 else tabla_K.K * 0
    return peso_k * kn + (1 - peso_k) * (1 - tabla_K.H)


def seleccionar_banda(tabla_K: pd.DataFrame, peso_k: float = 1.0) -> tuple[int, int]:
    """Argmax de curtosis o de la métrica curtosis/entropía."""
    fila = tabla_K.loc[puntuacion_bandas(tabla_K, peso_k).idxmax()]
    return int(fila.l), int(fila.m)


def nivel_maximo(fs: float, fd_max: float, factor: float = 2.0) -> int:
    """Mayor L tal que Δf_L >= factor·fd (la banda debe contener la modulación de la falla y sus bandas laterales)."""
    L = 0
    while fs / 2 ** (L + 2) >= factor * fd_max:
        L += 1
    return L


def filtrar(x: np.ndarray, fs: float, l: int, m: int) -> np.ndarray:
    """x*[n] = x_{l*,m*}[n]."""
    return subsenal(np.fft.fft(x - x.mean()), fs, l, m)


# ---------------------------------------------------------------------------
# 3. Espectro y envolvente espectral (diapositiva 20 de "Envelope espectral")
# ---------------------------------------------------------------------------

def espectro(x: np.ndarray, fs: float):
    """Espectro de amplitud unilateral |X[k]|·2/N."""
    N = x.size
    X = np.fft.rfft(x - x.mean())
    return np.fft.rfftfreq(N, 1 / fs), 2 * np.abs(X) / N


def envolvente_espectral(x: np.ndarray, fs: float):
    """a[n] = |x[n] + jH{x[n]}|,  a~ = a - media(a),  S_env[k] = |FFT{a~}|^2 (unilateral, normalizada por N^2)."""
    a = np.abs(senal_analitica(x - x.mean()))
    a = a - a.mean()
    N = a.size
    return np.fft.rfftfreq(N, 1 / fs), np.abs(np.fft.rfft(a)) ** 2 / N ** 2


# ---------------------------------------------------------------------------
# 4. Criterio de diagnóstico (objetivo y reproducible)
# ---------------------------------------------------------------------------

def evaluar_diagnostico(alpha, S, fd, fr=None, H=3, tol=0.02, p=0.01):
    """Clasifica la envolvente espectral en D / P / N.

    Para cada armónica h·fd (h=1..H) se toma el pico en ±tol y se compara con la media local de
    S_env en h·fd ± 0.3 fd (sin la ventana del pico). Bajo ruido, los bins de un periodograma son
    ~exponenciales, así que el máximo de M bins supera k·media con probabilidad 1-(1-e^{-k})^M.
    Se elige k para que esa probabilidad sea p (1 %)  =>  pico significativo si S_pico > k·media.

      D (Diagnosticable)            : h=1 o h=2 significativa Y es el pico más alto de S_env en [0.5 fd, H·fd+0.5 fd]
      P (Parcialmente diagnosticable): alguna armónica significativa, pero no domina el espectro
      N (No diagnosticable)          : ninguna armónica significativa
    Nota: en CWRU la envolvente tiene muchas líneas discretas (fr, FTF y combinaciones), por lo que la
    categoría P es sensible pero poco específica; la categoría D (pico dominante) es muy específica
    (ver 03_Comparacion: tasa de D en frecuencias señuelo).
    """
    alpha, S = np.asarray(alpha), np.asarray(S)
    res = dict(fd=fd, armonicas=[])
    signif = []
    for h in range(1, H + 1):
        d = np.abs(alpha - h * fd)
        win = d <= tol * h * fd
        loc = (d <= 0.3 * fd) & ~win
        if not win.any() or not loc.any():
            continue
        i = np.where(win)[0][np.argmax(S[win])]
        M = int(win.sum())
        k = -np.log(1 - (1 - p) ** (1 / M))
        media = S[loc].mean()
        ratio_db = 10 * np.log10(S[i] / media)
        es_sig = S[i] > k * media
        res["armonicas"].append(dict(h=h, f_pico=float(alpha[i]), ratio_db=float(ratio_db),
                                     umbral_db=float(10 * np.log10(k)), significativa=bool(es_sig)))
        signif.append((h, es_sig, S[i], alpha[i]))
    rango = (alpha >= 0.5 * fd) & (alpha <= (H + 0.5) * fd)
    pico_global = alpha[rango][np.argmax(S[rango])] if rango.any() else np.nan
    res["pico_global_hz"] = float(pico_global)
    domina = any(s and h <= 2 and abs(pico_global - fa) < 1e-9 for h, s, _, fa in signif)
    if domina:
        cat = "D"
    elif any(s for _, s, _, _ in signif):
        cat = "P"
    else:
        cat = "N"
    res["categoria"] = cat
    h1 = [a for a in res["armonicas"] if a["h"] == 1]
    res["ratio_db_fd"] = h1[0]["ratio_db"] if h1 else np.nan
    return res


NOMBRE_CATEGORIA = {"D": "Diagnosticable", "P": "Parcialmente diagnosticable", "N": "No diagnosticable"}

# ---------------------------------------------------------------------------
# 5. Gráficas (las 6 figuras de la tarea + kurtogram)
# ---------------------------------------------------------------------------
TINTA, TINTA2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
C_CRUDA, C_FILTRADA, C_ARMONICA = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10, "savefig.dpi": 160,
                     "axes.edgecolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2})


def _fig(ancho=7.5, alto=2.8):
    fig, ax = plt.subplots(figsize=(ancho, alto), constrained_layout=True)
    ax.yaxis.grid(True, color=GRILLA, lw=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def graficar_senal(x, fs, titulo, color, ruta, segundos=0.5):
    n = int(segundos * fs)
    t = np.arange(n) / fs
    fig, ax = _fig()
    ax.plot(t, x[:n], color=color, lw=0.6)
    ax.set_xlabel("tiempo [s]"); ax.set_ylabel("aceleración [g]")
    ax.set_title(titulo, loc="left")
    fig.savefig(ruta); plt.close(fig)


def graficar_espectro(x, fs, titulo, color, ruta, banda=None):
    f, A = espectro(x, fs)
    fig, ax = _fig()
    if banda is not None:
        ax.axvspan(banda[0], banda[1], color=C_FILTRADA, alpha=0.12, lw=0,
                   label=f"banda elegida {banda[0]:.0f}–{banda[1]:.0f} Hz")
        ax.legend(frameon=False, loc="upper right")
    ax.plot(f, A, color=color, lw=0.5)
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("frecuencia [Hz]"); ax.set_ylabel("|X(f)|")
    ax.set_title(titulo, loc="left")
    fig.savefig(ruta); plt.close(fig)


def graficar_envolvente(x, fs, fd, fr, titulo, color, ruta, H=5, diag=None):
    """Envolvente espectral en dB (relativa a su máximo en el rango graficado) con cursores de armónicas."""
    alpha, S = envolvente_espectral(x, fs)
    fmax = max(H + 0.5, 3.5) * fd
    sel = (alpha > 1.0) & (alpha <= fmax)
    SdB = 10 * np.log10(S[sel] / S[sel].max() + 1e-12)
    fig, ax = _fig(alto=3.1)
    ax.plot(alpha[sel], SdB, color=color, lw=0.6)
    for h in range(1, H + 1):
        ax.axvline(h * fd, color=C_ARMONICA, lw=0.7, ls=(0, (3, 2)))
        ax.text(h * fd, 1.0, f"{h}×fd", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7, color=TINTA2)
    for k in range(1, int(fmax / fr) + 1):
        ax.plot(k * fr, 0, marker="^", ms=4, color="#1baf7a", clip_on=False, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, fmax)
    ax.set_ylim(max(SdB.min(), -60), 3)
    ax.set_xlabel(f"frecuencia [Hz]   (líneas: armónicas de fd = {fd:.1f} Hz · triángulos: armónicas de fr = {fr:.2f} Hz)")
    ax.set_ylabel("S_env [dB rel. máx]")
    ax.set_title(titulo, loc="left", pad=14)
    if diag is not None:
        txt = (f"{NOMBRE_CATEGORIA[diag['categoria']]}\n"
               f"pico en fd: {diag['ratio_db_fd']:.1f} dB sobre el fondo local")
        ax.text(0.995, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=TINTA, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRILLA))
    fig.savefig(ruta); plt.close(fig)


def graficar_kurtogram(tabla_K, fs, L, ruta, seleccion=None, titulo="Kurtogram (MFD)"):
    img = np.full((L + 1, 2 ** L), np.nan)
    for _, r in tabla_K.iterrows():
        w = 2 ** (L - int(r.l))
        img[int(r.l), int(r.m) * w:(int(r.m) + 1) * w] = r.K
    fig, ax = plt.subplots(figsize=(7.5, 3.0), constrained_layout=True)
    im = ax.imshow(img, aspect="auto", cmap="Blues", interpolation="nearest",
                   extent=[0, fs / 2, L + 0.5, -0.5])
    for l in range(1, L + 1):
        for m in range(1, 2 ** l):
            x = m * fs / 2 ** (l + 1)
            ax.plot([x, x], [l - 0.5, l + 0.5], color="white", lw=0.5)
    if seleccion is not None:
        l, m = seleccion
        fl, fu = bordes_banda(l, m, fs)
        ax.add_patch(plt.Rectangle((fl, l - 0.5), fu - fl, 1, fill=False, ec=TINTA, lw=1.8))
    ax.set_yticks(range(L + 1)); ax.set_ylabel("nivel ℓ"); ax.set_xlabel("frecuencia central [Hz]")
    ax.set_title(titulo, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("K (curtosis compleja)")
    fig.savefig(ruta); plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Pipeline completo de la tarea para un registro
# ---------------------------------------------------------------------------

def diagnosticar(reg: dict, L: int | None = None) -> dict:
    """MFD vía curtosis espectral + envolvente espectral para un registro cargado con cargar_registro()."""
    x, fs, fd, fr = reg["x"], reg["fs"], reg["fd"], reg["fr"]
    if L is None:
        L = nivel_maximo(fs, fd)
    tabla = kurtogram(x, fs, L)
    l, m = seleccionar_banda(tabla)
    xf = filtrar(x, fs, l, m)
    diag_cruda = evaluar_diagnostico(*envolvente_espectral(x, fs), fd, fr)
    diag_filt = evaluar_diagnostico(*envolvente_espectral(xf, fs), fd, fr)
    return dict(L=L, tabla=tabla, banda=(l, m), limites=bordes_banda(l, m, fs), K=float(tabla.K.max()),
                xf=xf, diag_cruda=diag_cruda, diag_filtrada=diag_filt)


def figuras_registro(reg: dict, res: dict, carpeta: str) -> list[str]:
    os.makedirs(carpeta, exist_ok=True)
    x, fs, fd, fr, xf = reg["x"], reg["fs"], reg["fd"], reg["fr"], res["xf"]
    b = res["limites"]
    rutas = [os.path.join(carpeta, n) for n in (
        "1_senal_cruda.png", "2_espectro_senal_cruda.png", "3_envolvente_espectral_cruda.png",
        "4_senal_filtrada.png", "5_espectro_senal_filtrada.png", "6_envolvente_espectral_filtrada.png",
        "0_kurtogram.png")]
    graficar_senal(x, fs, "Figura 1: Señal cruda (Raw Signal)", C_CRUDA, rutas[0])
    graficar_espectro(x, fs, "Figura 2: Espectro de la señal cruda (Raw Signal Spectrum)", C_CRUDA, rutas[1], banda=b)
    graficar_envolvente(x, fs, fd, fr, "Figura 3: Envolvente espectral de la señal cruda", C_CRUDA, rutas[2],
                        diag=res["diag_cruda"])
    graficar_senal(xf, fs, f"Figura 4: Señal filtrada, banda {b[0]:.0f}–{b[1]:.0f} Hz (Filtered Signal)", C_FILTRADA, rutas[3])
    graficar_espectro(xf, fs, "Figura 5: Espectro de la señal filtrada (Filtered Signal Spectrum)", C_FILTRADA, rutas[4], banda=b)
    graficar_envolvente(xf, fs, fd, fr, "Figura 6: Envolvente espectral de la señal filtrada", C_FILTRADA, rutas[5],
                        diag=res["diag_filtrada"])
    graficar_kurtogram(res["tabla"], fs, res["L"], rutas[6], seleccion=res["banda"],
                       titulo=f"Kurtogram MFD (L={res['L']}): banda elegida ℓ={res['banda'][0]}, m={res['banda'][1]}, "
                              f"K={res['K']:.2f}")
    return rutas


def ficha(reg: dict, res: dict) -> dict:
    """Datos de la ficha de entregable (diapositivas 6 y 7 de la tarea)."""
    b = res["limites"]
    return {
        "Bearing": reg["sensor_falla"],
        "Nro. File": reg["registro"],
        "RPM": round(reg["rpm"], 1),
        "fr (Hz)": round(reg["fr"], 3),
        "Load (hp)": reg["carga_hp"],
        "Damage Size (in)": float(reg["tamano_in"]),
        "Damage Zone": ZONAS[reg["zona"]],
        "Fault Freq. (Hz)": round(reg["fd"], 2),
        "Banda MFD (Hz)": f"{b[0]:.0f}–{b[1]:.0f}",
        "Nivel/nodo": f"ℓ={res['banda'][0]}, m={res['banda'][1]}",
        "K máx": round(res["K"], 2),
        "Diagnóstico señal cruda": NOMBRE_CATEGORIA[res["diag_cruda"]["categoria"]],
        "Diagnóstico señal filtrada": NOMBRE_CATEGORIA[res["diag_filtrada"]["categoria"]],
        "Pico en fd, filtrada (dB)": round(res["diag_filtrada"]["ratio_db_fd"], 1),
    }


def clave_canal(mat, sensor, registro=None):
    """Selecciona el ID declarado; rechaza MAT ambiguos sin ID explícito."""
    candidatas = [k for k in mat if k.endswith(f"_{sensor}_time")]
    if registro is not None:
        clave = f"X{int(registro):03d}_{sensor}_time"
        if clave not in mat:
            raise KeyError(f"Registro/canal esperado ausente: {clave}; disponibles: {candidatas}")
        return clave
    if len(candidatas) != 1:
        raise ValueError(f"MAT ambiguo: indique registro. Canales disponibles: {candidatas}")
    return candidatas[0]


def rpm_registro(mat, registro, nominal=None):
    """RPM del mismo registro que la señal; nunca del primer registro del MAT."""
    claves = [f'X{int(registro):03d}RPM', f'X{int(registro):03d}_RPM']
    valores = [float(np.ravel(mat[k])[0]) for k in claves if k in mat]
    if valores and not all(np.isclose(v, valores[0]) for v in valores):
        raise ValueError('RPM contradictorias para el mismo registro')
    rpm = valores[0] if valores else nominal
    if rpm is None or not np.isfinite(rpm) or rpm <= 0:
        raise ValueError('Faltan RPM del registro y una velocidad nominal válida')
    return float(rpm)
