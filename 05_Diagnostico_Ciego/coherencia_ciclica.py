"""Rasgos de coherencia cíclica ACP; solo señal, fs y catálogo completo."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.io import loadmat
from scipy.signal import hilbert
from ventanas_verificadas import prefijo
from modelo_ciego import F, validar_catalogo
from modelo_por_dominio import firma
from modelo_espectral import FAMILIAS
from rasgos_fisicos import giro_catalogo

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_coherencia_ciclica'
FS, NW, PASO = 12000, 256, 128
DELTA = np.linspace(-.02, .02, 21)


def mapa(z, alfas):
    """ACP no simétrico: corrige la fase ABSOLUTA de cada ventana.

    Equivale a pares f±alfa/2 con f central desplazado. Ventana semiseno
    y 50% de solape: suma de cuadrados constante en el interior.
    """
    z = np.asarray(z)
    alfas = np.asarray(alfas, dtype=float)
    if z.ndim != 1 or len(z) < NW or not np.isfinite(z).all():
        raise ValueError('Se requiere señal analítica finita')
    if alfas.ndim != 1 or not np.isfinite(alfas).all() or np.any((alfas < 0) | (alfas >= FS/2)):
        raise ValueError('Frecuencias cíclicas fuera del dominio')
    inicios = np.arange(0, len(z)-NW+1, PASO)
    n = np.arange(NW)
    w = np.sin(np.pi*(n+.5)/NW)
    segmentos = z[inicios[:, None]+n] * w
    a = fft(segmentos, axis=-1)[:, :NW//2+1]
    pa = (abs(a)**2).sum(axis=0)
    salida = np.zeros((len(alfas), NW//2+1))
    for inicio in range(0, len(alfas), 8):
        alfa = alfas[inicio:inicio+8]
        modulacion = np.exp(-2j*np.pi*alfa[:, None]*n/FS)
        b = fft(segmentos[None]*modulacion[:, None], axis=-1)[..., :NW//2+1]
        fase = np.exp(-2j*np.pi*alfa[:, None]*inicios/FS)
        cruzada = (b*a.conj()[None]*fase[..., None]).sum(axis=1)
        denominador = (abs(b)**2).sum(axis=1)*pa
        seguro = np.maximum(denominador, denominador.max(axis=1, keepdims=True)*1e-14)
        c = np.divide(abs(cruzada)**2, seguro, out=np.zeros_like(seguro), where=seguro > 0)
        salida[inicio:inicio+len(alfa)] = np.clip(c, 0., 1.)
    return salida


def agrupar(c, alfas):
    f = np.arange(NW//2+1)*FS/NW
    salida = np.zeros((len(alfas), 3))
    for i, alfa in enumerate(alfas):
        # Dominio principal: ambos componentes positivos y por debajo de Nyquist.
        valido = (f+alfa <= FS/2) & (f+alfa/2 >= 500) & (f+alfa/2 <= 5500)
        if alfa == 0 or not np.any(valido):
            continue  # la coherencia a alfa=0 es trivial y no informa de una falla
        v = np.sort(c[i, valido])
        salida[i] = [v.mean(), v[-max(1, len(v)//4):].mean(), v[-1]]
    return salida


def resumir(curvas):
    curvas = np.asarray(curvas).reshape(-1, len(DELTA))
    return np.column_stack([curvas[:, len(DELTA)//2], curvas.max(axis=1), curvas.mean(axis=1),
        np.where(np.ptp(curvas, axis=1) > 1e-12, abs(DELTA[curvas.argmax(axis=1)])/.02, 0.)]).ravel()


def medir(x, fs, catalogo):
    validar_catalogo(catalogo, FS)
    x = prefijo(x, fs)
    x = x-x.mean()
    rms = float(np.sqrt(np.mean(x*x)))
    if rms == 0:
        return np.zeros(600)
    z = hilbert(x/rms)
    bases = np.array([2*catalogo['BSF'] if k == '2BSF' else catalogo[k] for k in FAMILIAS])
    armonicas = bases[:, None, None]*np.arange(1, 7)[None, :, None]*(1+DELTA)
    moduladores = np.array([giro_catalogo(catalogo), catalogo['FTF']])
    laterales = abs(bases[:, None, None, None]*(1+DELTA)[None, None, None, :] +
                    moduladores[None, :, None, None]*np.array([-1, 1])[None, None, :, None])
    solicitudes = np.r_[armonicas.ravel(), laterales.ravel()]
    # Resolución cíclica física Fs/L; no inventa resolución mediante zero-padding.
    bins, inversa = np.unique(np.rint(solicitudes*len(z)/FS).astype(int), return_inverse=True)
    alfas = bins*FS/len(z)
    valores = agrupar(mapa(z, alfas), alfas)[inversa]
    n = armonicas.size
    c = valores[:n].reshape(5, 6, 21, 3).transpose(0, 1, 3, 2)
    soporte = np.stack([c.max(axis=1), np.sort(c, axis=1)[:, -2:].mean(axis=1),
                        np.minimum(c[:, 0], c[:, 1:].max(axis=1))], axis=1)
    l = valores[n:].reshape(5, 2, 2, 21, 3).transpose(0, 1, 2, 4, 3)
    simetrico = np.minimum(l[:, :, 0], l[:, :, 1])
    salida = np.r_[resumir(c), resumir(soporte), simetrico.max(axis=-1).ravel(), simetrico.mean(axis=-1).ravel()]
    assert salida.shape == (600,) and np.isfinite(salida).all()
    return salida


def comprobar():
    rng = np.random.default_rng(60561)
    n = np.arange(30000)
    catalogo = {k: 30*v for k, v in F.MULTIPLICADORES['DE'].items()}
    a = np.zeros(30000)
    # Ruido de banda ancha modulado: la propiedad es de segundo orden.
    a = rng.normal(size=30000)*(1+.8*np.cos(2*np.pi*162.4*n/FS))
    z = hilbert(a)
    alfas = np.array([107.6, 162.4, 200.])
    c = mapa(z, alfas)
    assert np.all((c >= 0) & (c <= 1))
    p = agrupar(c, alfas)
    assert p[1, 0] > 5*max(p[0, 0], p[2, 0])
    assert np.allclose(c, mapa(7*z, alfas), atol=1e-12)
    w = np.sin(np.pi*(np.arange(NW)+.5)/NW)
    assert np.allclose(w[:PASO]**2+w[PASO:]**2, 1.)
    # Referencia directa de la ecuación simétrica con tiempos absolutos.
    inicios = np.arange(0, len(z)-NW+1, PASO)
    indices = inicios[:, None]+np.arange(NW)
    f = 40*FS/NW
    u = (z[indices]*w*np.exp(-2j*np.pi*(f+162.4)*indices/FS)).sum(axis=1)
    v = (z[indices]*w*np.exp(-2j*np.pi*f*indices/FS)).sum(axis=1)
    directo = abs(np.sum(u*v.conj()))**2/np.sum(abs(u)**2)/np.sum(abs(v)**2)
    assert np.isclose(c[1, 40], directo, rtol=1e-10)
    for fs in (12000, 48000):
        x = rng.normal(size=6*fs)
        antes = medir(x, fs, catalogo)
        x[3*fs:] = rng.normal(scale=10, size=3*fs)
        assert np.array_equal(antes, medir(x, fs, catalogo))
    print('OK: ACP coincide con ecuacion directa, fase absoluta, escala, dominio y futuro 12/48 kHz.', flush=True)


def preparar():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_plantillas_coherentes/tipos_133.csv'
    d = pd.read_csv(fuente)
    protocolo = dict(codigo=firma(Path(__file__)), fuente=firma(fuente),
        ventanas=firma(AQUI / 'ventanas_verificadas.py'), catalogo=firma(AQUI / 'modelo_ciego.py'),
        referencia='Antoni 2007, doi:10.1016/j.jsv.2007.02.029; ACP ecuaciones 13,17,18',
        receta='prefijo 2.5 s; fs12k; semiseno256 solape50%; fase absoluta; alfa cuantizada Fs/L',
        familias='todas las cinco, H1..6, delta comun +/-2%; laterales +/-giro y +/-FTF',
        rasgos='600; coherencia normalizada, sin identidad de portadora ni amplitud absoluta',
        entrada='senal, fs, catalogo completo; no etiquetas ni frecuencia esperada',
        proposito='identificacion del tipo; no reemplaza el espectro ni el juez D/PD/ND')
    rp = SALIDA / 'protocolo_preparacion.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    progreso = SALIDA / 'progreso.npz'
    x, hechos = np.full((133, 600), np.nan), np.zeros(133, dtype=bool)
    if progreso.exists():
        with np.load(progreso, allow_pickle=False) as z:
            x, hechos = z['x'], z['hechos']
    assert len(d) == 133 and x.shape == (133, 600) and hechos.shape == (133,)
    for i, r in enumerate(d.itertuples()):
        if hechos[i]:
            continue
        if r.suficiente:
            mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
            senal = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            x[i] = medir(senal, int(r.fs), json.loads(r.catalogo))
        hechos[i] = True
        if (i+1) % 10 == 0 or i == len(d)-1:
            np.savez_compressed(progreso, x=x, hechos=hechos)
            print('Coherencia:', int(hechos.sum()), '/133', flush=True)
    assert hechos.all() and np.isfinite(x[d.suficiente]).all()
    d.to_csv(SALIDA / 'muestra.csv', index=False)
    np.savez_compressed(SALIDA / 'rasgos.npz', x=x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparar', action='store_true')
    args = parser.parse_args()
    comprobar()
    if args.preparar:
        preparar()
