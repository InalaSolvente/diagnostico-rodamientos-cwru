"""Simulador de señales de rodamiento con interferencias (verdad conocida)."""
import numpy as np


def resonance_ir(fs, fn, zeta, dur=0.05):
    t = np.arange(int(dur * fs)) / fs
    wn = 2 * np.pi * fn
    return np.exp(-zeta * wn * t) * np.sin(wn * np.sqrt(1 - zeta ** 2) * t)


def impulse_train(N, fs, rate, jitter=0.01, rng=None, amp_mod=None):
    """Tren de impulsos a 'rate' Hz con deslizamiento aleatorio (jitter relativo)."""
    rng = np.random.default_rng() if rng is None else rng
    T = 1.0 / rate
    t, times = rng.uniform(0, T), []
    while t < N / fs:
        times.append(t)
        t += T * (1 + jitter * rng.standard_normal())
    d = np.zeros(N)
    for tk in times:
        k = int(round(tk * fs))
        if k < N:
            d[k] += 1.0 if amp_mod is None else amp_mod(tk)
    return d


def convolve_ir(d, ir):
    return np.convolve(d, ir)[: d.size]


def simulate(fs=12000, dur=10.0, fd=107.3, fr=29.95, fn=3500.0, zeta=0.04,
             snr_db=-10.0, rng=None,
             rand_impulses=0.0, rand_fn=1200.0, rand_rate=3.0,
             looseness=0.0, loose_fn=5200.0,
             emi=0.0, tones=0.0):
    """Señal = falla de rodamiento (+ interferencias opcionales) + ruido gaussiano.

    snr_db       : potencia(falla)/potencia(ruido gaussiano)
    rand_impulses: amplitud relativa de impulsos aleatorios (Poisson) en otra resonancia (rand_fn)
    looseness    : amplitud relativa de impactos periódicos a fr (holgura) en loose_fn
    emi          : amplitud relativa de un tren de armónicos de línea (60 Hz, 1..40)
    tones        : amplitud relativa de armónicos de giro k*fr + un tono de engrane de alta frecuencia
    """
    rng = np.random.default_rng() if rng is None else rng
    N = int(fs * dur)
    t = np.arange(N) / fs
    s = convolve_ir(impulse_train(N, fs, fd, 0.01, rng), resonance_ir(fs, fn, zeta))
    s /= s.std()
    x = s.copy()
    if rand_impulses > 0:
        n_ev = rng.poisson(rand_rate * dur)
        d = np.zeros(N)
        d[rng.integers(0, N, n_ev)] = rng.uniform(0.5, 1.5, n_ev)
        r = convolve_ir(d, resonance_ir(fs, rand_fn, 0.03))
        x += rand_impulses * r / (r.std() + 1e-12)
    if looseness > 0:
        r = convolve_ir(impulse_train(N, fs, fr, 0.002, rng), resonance_ir(fs, loose_fn, 0.03))
        x += looseness * r / r.std()
    if emi > 0:
        e = sum(np.sin(2 * np.pi * 60 * k * t + rng.uniform(0, 2 * np.pi)) / k ** 0.5 for k in range(1, 41))
        x += emi * e / e.std()
    if tones > 0:
        g = sum(np.sin(2 * np.pi * k * fr * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 6))
        g += 2 * np.sin(2 * np.pi * 57 * fr * t)
        x += tones * g / g.std()
    x += rng.standard_normal(N) * 10 ** (-snr_db / 20)
    return x
