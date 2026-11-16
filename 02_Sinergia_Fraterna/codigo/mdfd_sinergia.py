"""
mdfd_sinergia.py
=================
MDFD (Multilevel Dyadic Fourier Decomposition) + métricas basadas en la
estructura padre-hijos del árbol diádico.

Notación (igual a las diapositivas del curso, Prof. N. Rodríguez):
    X[k] = FFT{x[n]}
    B_{l,m} = [m*Df_l, (m+1)*Df_l),  Df_l = Fs / 2^(l+1)
    x_{l,m}[n] = IFFT{ H_{l,m}[k] X[k] }
    z_{l,m}[n] = x_{l,m}[n] + j H{x_{l,m}[n]}   (señal analítica, centrada)
    K_{l,m}    = mu4 / mu2^2 - 2                (curtosis compleja)

Métricas nuevas (ver informe):
    * Sinergia fraterna  S_{l,m} = K_{l,m} - w_a^2 K_a - w_b^2 K_b
      (identidad exacta de herencia de curtosis padre -> hijos)
    * Sinergia estandarizada Z_{l,m} por surrogados de desplazamiento circular
    * Espectro de envolvente cruzado fraterno (EECF) y coherencia fraterna
      en las armónicas de la falla (versión cíclica / dirigida de la sinergia)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import csd, welch

# ---------------------------------------------------------------------------
# 1. Árbol MDFD
# ---------------------------------------------------------------------------

def band_edges(l: int, m: int, fs: float):
    df = fs / 2 ** (l + 1)
    return m * df, (m + 1) * df


def analytic_node_spectrum(X: np.ndarray, fs: float, l: int, m: int) -> np.ndarray:
    """Espectro de la señal analítica del nodo (l,m).

    Z_{l,m}[k] = h[k] H_{l,m}[k] X[k], con h = 1 (k=0, k=N/2), 2 (0<k<N/2), 0 (k>N/2).
    Es lineal en X y los soportes de nodos hermanos son disjuntos, por lo que
    z_padre = z_hijo_izq + z_hijo_der EXACTAMENTE.
    """
    N = X.size
    f = np.fft.fftfreq(N, d=1.0 / fs)
    fl, fu = band_edges(l, m, fs)
    af = np.abs(f)
    last = m == 2 ** l - 1
    mask = (af >= fl) & ((af < fu) | (last & (af <= fu + 1e-9)))
    h = np.zeros(N)
    h[0] = 1.0
    if N % 2 == 0:
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[1:(N + 1) // 2] = 2.0
    return X * mask * h


class MDFDTree:
    """Calcula (y cachea) las señales analíticas de todos los nodos hasta el nivel L."""

    def __init__(self, x: np.ndarray, fs: float, L: int):
        self.x = np.asarray(x, float).ravel()
        self.x = self.x - self.x.mean()
        self.fs = fs
        self.L = L
        self.N = self.x.size
        self.X = np.fft.fft(self.x)
        self._z = {}

    def z(self, l: int, m: int) -> np.ndarray:
        key = (l, m)
        if key not in self._z:
            zz = np.fft.ifft(analytic_node_spectrum(self.X, self.fs, l, m))
            self._z[key] = zz - zz.mean()          # centrado (diapositiva 15)
        return self._z[key]

    def real(self, l: int, m: int) -> np.ndarray:
        return self.z(l, m).real

    def nodes(self, L: int | None = None):
        L = self.L if L is None else L
        for l in range(L + 1):
            for m in range(2 ** l):
                yield l, m

    def fc_bw(self, l: int, m: int):
        fl, fu = band_edges(l, m, self.fs)
        return 0.5 * (fl + fu), fu - fl


# ---------------------------------------------------------------------------
# 2. Momentos y curtosis compleja
# ---------------------------------------------------------------------------

def moments(z: np.ndarray):
    e = np.abs(z) ** 2
    return e.mean(), (e ** 2).mean()


def complex_kurtosis(z: np.ndarray) -> float:
    mu2, mu4 = moments(z)
    return mu4 / mu2 ** 2 - 2.0 if mu2 > 0 else 0.0


def kurtogram(tree: MDFDTree) -> dict:
    return {(l, m): complex_kurtosis(tree.z(l, m)) for l, m in tree.nodes()}


# ---------------------------------------------------------------------------
# 3. Identidad de herencia de curtosis y sinergia fraterna
# ---------------------------------------------------------------------------

def heredity_terms(za: np.ndarray, zb: np.ndarray) -> dict:
    """Descompone la curtosis del padre p = a + b (hijos con soporte espectral disjunto).

        K_p = w_a^2 K_a + w_b^2 K_b + S
        S   = S_cov + S_fase
        S_cov  = 4 cov(|a|^2, |b|^2) / P^2              (co-modulación de envolventes)
        S_fase = [2 Re<a^2 b*^2> + 4 <(|a|^2+|b|^2) Re(a b*)>] / P^2   (acoplamiento de fase)
    """
    ea, eb = np.abs(za) ** 2, np.abs(zb) ** 2
    Pa, Pb = ea.mean(), eb.mean()
    P = Pa + Pb
    wa, wb = Pa / P, Pb / P
    Ka = (ea ** 2).mean() / Pa ** 2 - 2
    Kb = (eb ** 2).mean() / Pb ** 2 - 2
    cov = (ea * eb).mean() - Pa * Pb
    cross = za * np.conj(zb)
    s_cov = 4 * cov / P ** 2
    s_fase = (2 * np.real((za ** 2 * np.conj(zb) ** 2).mean())
              + 4 * ((ea + eb) * cross.real).mean()) / P ** 2
    return dict(wa=wa, wb=wb, Ka=Ka, Kb=Kb, S_cov=s_cov, S_fase=s_fase,
                S=s_cov + s_fase, inherited=wa ** 2 * Ka + wb ** 2 * Kb)


def synergy_map(tree: MDFDTree, n_surr: int = 16, rng=None) -> dict:
    """Para cada nodo padre (l < L) devuelve la sinergia S y su versión estandarizada Z.

    Surrogado: se desplaza circularmente z_b un retardo aleatorio tau. El desplazamiento
    no cambia |Z_b[k]| -> conserva energía, curtosis del hijo y ortogonalidad, pero destruye
    la sincronía temporal entre hermanos. Así
        K_p^surr = w_a^2 K_a + w_b^2 K_b + S^surr
    y Z = (S - media(S^surr)) / std(S^surr) mide cuánto más impulsivo es el padre de lo que
    sería si sus hijos fuesen independientes.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    out = {}
    N = tree.N
    for l in range(tree.L):
        min_shift = int(4 * tree.fs / (tree.fs / 2 ** (l + 2)))   # > 4 tiempos de correlación del hijo
        for m in range(2 ** l):
            za, zb = tree.z(l + 1, 2 * m), tree.z(l + 1, 2 * m + 1)
            t = heredity_terms(za, zb)
            ss = []
            for _ in range(n_surr):
                tau = int(rng.integers(min_shift, N - min_shift))
                ss.append(heredity_terms(za, np.roll(zb, tau))["S"])
            ss = np.asarray(ss)
            sd = ss.std(ddof=1) if n_surr > 1 else np.nan
            t["S_surr_mean"], t["S_surr_sd"] = ss.mean(), sd
            t["Z"] = (t["S"] - ss.mean()) / sd if sd > 0 else 0.0
            t["Kp"] = complex_kurtosis(tree.z(l, m))
            out[(l, m)] = t
    return out


# ---------------------------------------------------------------------------
# 4. Envolventes y espectros (auto y cruzado fraterno)
# ---------------------------------------------------------------------------

def squared_envelope(z: np.ndarray) -> np.ndarray:
    e = np.abs(z) ** 2
    return e - e.mean()


def ses(z: np.ndarray, fs: float, seg_sec: float = 1.0):
    """Espectro de la envolvente cuadrática (Welch) -> (alpha, S)."""
    e = squared_envelope(z)
    nper = int(min(e.size, seg_sec * fs))
    return welch(e, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="constant")


def sibling_cross_ses(za: np.ndarray, zb: np.ndarray, fs: float, seg_sec: float = 1.0):
    """Espectro de envolvente cruzado fraterno (EECF).

    C_ab(alpha) = CSD(|z_a|^2, |z_b|^2). Para ruido en bandas disjuntas las envolventes son
    independientes -> E[C_ab] = 0: el ruido NO se acumula como en el SES (auto),
    sólo sobrevive la modulación común a ambos hermanos.
    Devuelve alpha, C (complejo), S_aa, S_bb.
    """
    ea, eb = squared_envelope(za), squared_envelope(zb)
    nper = int(min(ea.size, seg_sec * fs))
    kw = dict(fs=fs, nperseg=nper, noverlap=nper // 2, detrend="constant")
    a, C = csd(ea, eb, **kw)
    _, Saa = welch(ea, **kw)
    _, Sbb = welch(eb, **kw)
    return a, C, Saa, Sbb


def _harmonic_peak(alpha, spec, f0, h, tol):
    lo, hi = h * f0 * (1 - tol), h * f0 * (1 + tol)
    idx = np.where((alpha >= lo) & (alpha <= hi))[0]
    if idx.size == 0:
        return None
    return idx[np.argmax(spec[idx])]


def targeted_auto_index(z, fs, fd, H=3, tol=0.02, seg_sec=1.0):
    """Índice dirigido clásico (tipo ICS2/RCC): energía del SES en armónicas / energía total."""
    a, S = ses(z, fs, seg_sec)
    band = (a > 0.5 * fd) & (a <= (H + 0.5) * fd)
    num = 0.0
    for h in range(1, H + 1):
        i = _harmonic_peak(a, S, fd, h, tol)
        if i is not None:
            num += S[i]
    den = S[band].sum()
    return num / den if den > 0 else 0.0


def targeted_sibling_coherence(za, zb, fs, fd, H=3, tol=0.02, seg_sec=1.0):
    """Coherencia fraterna dirigida: media sobre armónicas h*fd de
       rho(alpha) = Re C_ab / sqrt(S_aa S_bb)  (coherencia real, nula esperada = 0)."""
    a, C, Saa, Sbb = sibling_cross_ses(za, zb, fs, seg_sec)
    den = np.sqrt(Saa * Sbb) + 1e-300
    rho = np.real(C) / den
    vals = []
    for h in range(1, H + 1):
        i = _harmonic_peak(a, np.abs(C) / den, fd, h, tol)
        if i is not None:
            vals.append(rho[i])
    return float(np.mean(vals)) if vals else 0.0


def synergy_spectrum(za, zb, fs, seg_sec=1.0):
    """Espectro de sinergia s(alpha) = 4 Re C_ab(alpha) / P_p^2.

    Por Parseval, integral de s(alpha) d alpha = S_cov (parte de co-modulación de la sinergia):
    la sinergia se DESCOMPONE en frecuencias cíclicas.
    """
    a, C, Saa, Sbb = sibling_cross_ses(za, zb, fs, seg_sec)
    P = (np.abs(za) ** 2).mean() + (np.abs(zb) ** 2).mean()
    return a, 4 * np.real(C) / P ** 2


def targeted_synergy(za, zb, fs, fd, H=3, tol=0.02, seg_sec=1.0):
    """Sinergia dirigida S_fd: parte de la curtosis del padre explicada por la
    co-modulación de los hijos en las armónicas h*fd (unidades de curtosis), descontando
    el fondo cíclico local (mediana en h*fd ± 0.3 fd) para no premiar co-modulación de
    banda ancha (p. ej. impulsos aleatorios)."""
    a, s = synergy_spectrum(za, zb, fs, seg_sec)
    da = a[1] - a[0]
    tot = 0.0
    for h in range(1, H + 1):
        d = np.abs(a - h * fd)
        w = d <= max(tol * h * fd, da)
        nb = (d <= 0.3 * fd) & (d > 2 * tol * h * fd)      # fondo cíclico local
        base = np.median(s[nb]) if nb.any() else 0.0
        tot += (s[w] - base).sum() * da                     # sinergia NETA sobre el fondo
    return tot


def cyclic_maps(tree: MDFDTree, fd: float, H=3, tol=0.02, seg_sec=1.0):
    """Mapas dirigidos: índice auto (por nodo), coherencia fraterna y sinergia dirigida (por padre)."""
    auto, sib, sfd = {}, {}, {}
    for l, m in tree.nodes():
        auto[(l, m)] = targeted_auto_index(tree.z(l, m), tree.fs, fd, H, tol, seg_sec)
        if l < tree.L:
            za, zb = tree.z(l + 1, 2 * m), tree.z(l + 1, 2 * m + 1)
            sib[(l, m)] = targeted_sibling_coherence(za, zb, tree.fs, fd, H, tol, seg_sec)
            sfd[(l, m)] = targeted_synergy(za, zb, tree.fs, fd, H, tol, seg_sec)
    return auto, sib, sfd


# ---------------------------------------------------------------------------
# 5. Selección y diagnóstico
# ---------------------------------------------------------------------------

def argmax_node(metric: dict, valid=None):
    items = [(k, v) for k, v in metric.items() if (valid is None or valid(k)) and np.isfinite(v)]
    return max(items, key=lambda kv: kv[1])[0]


def synergy_descent(tree: MDFDTree, syn: dict, z_thr: float = 3.0):
    """Descenso por sinergia (O(L) decisiones, estilo árbol de decisión).

    Desde la raíz: si el nodo actual es sinérgico (Z >= z_thr) su ancho de banda es
    'indivisible' -> se detiene; si no, baja al hijo que aporta más curtosis heredada
    (w^2 K). Devuelve el camino recorrido y el nodo final.
    """
    l, m = 0, 0
    path = [(l, m)]
    while l < tree.L:
        t = syn[(l, m)]
        if t["Z"] >= z_thr and l > 0:
            break
        ca, cb = t["wa"] ** 2 * t["Ka"], t["wb"] ** 2 * t["Kb"]
        l, m = l + 1, 2 * m + (0 if ca >= cb else 1)
        path.append((l, m))
    return path[-1], path


def diagnostic_score(z, fs, fd, bw=None, H=3, tol=0.02, seg_sec=1.0, thr_db=6.0):
    """Relación armónica/fondo (HBR, dB) en el SES del nodo elegido.

    Para cada armónica h (sólo las que caben en el ancho de banda del nodo, h*fd < bw):
        HBR_h = 10 log10( pico en h*fd±tol / mediana del SES en h*fd ± 0.3 fd (sin el pico) ).
    Devuelve (media de HBR_h, detectado) con detectado = HBR_1 >= thr_db.
    """
    a, S = ses(z, fs, seg_sec)
    bw = fs / 2 if bw is None else bw
    vals = []
    for h in range(1, H + 1):
        if h * fd * (1 + tol) >= bw:
            break
        i = _harmonic_peak(a, S, fd, h, tol)
        if i is None:
            break
        loc = (np.abs(a - h * fd) <= 0.3 * fd) & (np.abs(a - h * fd) > 2 * tol * h * fd)
        bg = np.median(S[loc]) + 1e-300
        vals.append(10 * np.log10(S[i] / bg))
    if not vals:
        return -np.inf, False
    return float(np.mean(vals)), vals[0] >= thr_db
