"""ACP en rejilla cíclica densa mediante FFT compartida entre hipótesis."""
from pathlib import Path
import json
import numpy as np
from scipy.fft import fft, next_fast_len
from scipy.signal import hilbert
from coherencia_ciclica import FS, NW, PASO, DELTA, mapa as mapa_directo, agrupar
from modelo_ciego import F


def longitud_fft(n):
    # Conserva EXACTAMENTE los nodos de portadora de la FFT corta original.
    return NW*next_fast_len(int(np.ceil(n/NW)))


def mapa(z, alfas):
    """Devuelve alfas realmente muestreadas y coherencia, sin crear resolución.

    La resolución física sigue siendo FS/len(z). El zero-padding solo permite
    evaluar todas las separaciones cíclicas con las mismas FFT de ventana.
    """
    z, alfas = np.asarray(z), np.asarray(alfas, dtype=float)
    if z.ndim != 1 or not np.iscomplexobj(z) or len(z) < NW or not np.isfinite(z).all():
        raise ValueError('Se requiere señal analítica compleja finita')
    if alfas.ndim != 1 or not np.isfinite(alfas).all() or np.any((alfas < 0) | (alfas >= FS/2)):
        raise ValueError('Frecuencias cíclicas fuera de dominio')
    nf = longitud_fft(len(z))
    bins, inversa = np.unique(np.rint(alfas*nf/FS).astype(int), return_inverse=True)
    if np.any(bins >= nf//2):
        raise ValueError('Separación cíclica redondeada fuera de dominio')
    usados = bins*FS/nf
    portadora = np.arange(NW//2+1)*(nf//NW)
    inicios = np.arange(0, len(z)-NW+1, PASO)
    n = np.arange(NW)
    w = np.sin(np.pi*(n+.5)/NW)
    cruzada = np.zeros((len(bins), len(portadora)), dtype=complex)
    pa = np.zeros(len(portadora))
    pb = np.zeros_like(cruzada.real)
    # Memoria acotada: 32 ventanas y 16 separaciones por bloque.
    for i in range(0, len(inicios), 32):
        tiempos = inicios[i:i+32]
        segmentos = z[tiempos[:, None]+n]*w
        espectros = fft(segmentos, n=nf, axis=-1)
        a = espectros[:, portadora]
        pa += (abs(a)**2).sum(axis=0)
        for j in range(0, len(bins), 16):
            b = espectros[:, portadora[None, :]+bins[j:j+16, None]]
            fase = np.exp(-2j*np.pi*tiempos[:, None]*usados[None, j:j+16]/FS)
            cruzada[j:j+16] += (b*a.conj()[:, None, :]*fase[..., None]).sum(axis=0)
            pb[j:j+16] += (abs(b)**2).sum(axis=0)
    denominador = pb*pa
    seguro = np.maximum(denominador, denominador.max(axis=1, keepdims=True)*1e-14)
    c = np.divide(abs(cruzada)**2, seguro, out=np.zeros_like(seguro), where=seguro > 0)
    return usados[inversa], np.clip(c[inversa], 0., 1.)


def comprobar():
    rng = np.random.default_rng(163)
    n = 30000
    z = hilbert(rng.normal(size=n))
    consultas = np.array([12., 70.71, 107.5, 162.45, 599.123, 975.1])
    usados, c = mapa(z, consultas)
    directo = mapa_directo(z, usados)
    assert np.allclose(c, directo, rtol=1e-8, atol=1e-10)
    _, escalado = mapa(7*z, consultas)
    assert np.allclose(c, escalado, rtol=1e-9, atol=1e-10)
    # Contraejemplo ciego al desplazamiento real: consulta todo el intervalo del catálogo.
    base = 30*F.MULTIPLICADORES['DE']['BPFI']*6
    real_solo_fixture = base*(1+.007)
    t = np.arange(n)/FS
    x = rng.normal(size=n)*(1+.8*np.cos(2*np.pi*real_solo_fixture*t))
    z = hilbert(x)
    gruesos = np.rint(base*(1+DELTA)*n/FS)*FS/n
    potencia_gruesa = agrupar(mapa_directo(z, gruesos), gruesos)[:, 0]
    nf = longitud_fft(n)
    k = np.arange(int(np.floor(.98*base*nf/FS)), int(np.ceil(1.02*base*nf/FS))+1)
    densos, cd = mapa(z, k*FS/nf)
    potencia_densa = agrupar(cd, densos)[:, 0]
    assert potencia_densa.max() > 5*potencia_gruesa.max()
    pico = float(densos[potencia_densa.argmax()])
    assert abs(pico-real_solo_fixture) < FS/n
    resultado = dict(estimator_equivalente_a_ACP_directo=True, escala_invariante=True,
        resolucion_fisica=FS/n, paso_rejilla_densa=FS/nf,
        frecuencia_fixture=real_solo_fixture, pico_denso=pico,
        coherencia_maxima_gruesa=float(potencia_gruesa.max()),
        coherencia_maxima_densa=float(potencia_densa.max()),
        conclusiones='La rejilla de 21 desplazamientos puede omitir armónicas altas. No es rendimiento de un modelo.',
        datos_reales_evaluados=0)
    p = Path(__file__).parent/'resultados_acp_denso'
    p.mkdir(exist_ok=True)
    (p/'autoprueba.json').write_text(json.dumps(resultado, indent=2), encoding='utf-8')
    print(json.dumps(resultado), flush=True)


if __name__ == '__main__':
    comprobar()
