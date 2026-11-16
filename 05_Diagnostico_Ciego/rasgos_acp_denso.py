"""Coherencia densa de hasta 5 s de selección, desplazamiento firmado y giro."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert, resample_poly
from acp_denso import mapa, longitud_fft
from coherencia_ciclica import agrupar, FS
from modelo_ciego import F, validar_catalogo
from modelo_espectral import FAMILIAS
from modelo_por_dominio import firma
from rasgos_fisicos import giro_catalogo

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_acp_denso'


def resumir(curvas, delta):
    c = np.asarray(curvas).reshape(-1, len(delta))
    peso = np.maximum(c-np.median(c, axis=1, keepdims=True), 0.)
    suma = peso.sum(axis=1, keepdims=True)
    peso = np.divide(peso, suma, out=np.full_like(peso, 1/len(delta)), where=suma > 0)
    mu = (peso*delta).sum(axis=1)
    ancho = np.sqrt((peso*(delta-mu[:, None])**2).sum(axis=1))/.02
    pico = np.where(np.ptp(c, axis=1) > 1e-12, delta[c.argmax(axis=1)]/.02, 0.)
    return np.column_stack([c[:, len(delta)//2], c.max(axis=1), c.mean(axis=1), pico, mu/.02, ancho]).ravel()


def medir(x, fs, catalogo):
    x = np.asarray(x, dtype=float)
    if fs not in (12000, 48000) or x.ndim != 1 or len(x) < 5*fs or not np.isfinite(x).all():
        raise ValueError('Se necesita señal finita de al menos 5 s, fs12/48 kHz')
    validar_catalogo(catalogo, FS)
    # Corta ANTES de remuestrear o calcular Hilbert: nunca mira la segunda mitad.
    z = x[:min(len(x)//2, 5*fs)]
    if fs == 48000:
        z = resample_poly(z, 1, 4)
    z = z-z.mean()
    rms = float(np.sqrt(np.mean(z*z)))
    if rms == 0:
        return np.zeros(990)
    z = hilbert(z/rms)
    bases = np.array([2*catalogo['BSF'] if f == '2BSF' else catalogo[f] for f in FAMILIAS])
    nf = longitud_fft(len(z))
    paso = FS/nf
    # Una rejilla común suficientemente fina incluso para H6 de la base más alta.
    puntos = int(np.ceil(.04/(paso/(2*6*bases.max()))))+1
    if puntos % 2 == 0:
        puntos += 1
    delta = np.linspace(-.02, .02, puntos)
    h = bases[:, None, None]*np.arange(1, 7)[None, :, None]*(1+delta)
    giro = giro_catalogo(catalogo)
    moduladores = np.array([giro, catalogo['FTF'], 2*catalogo['FTF']])
    laterales = abs(bases[:, None, None, None]*(1+delta)[None, None, None, :] +
        moduladores[None, :, None, None]*np.array([-1, 1])[None, None, :, None])
    rotacion = np.rint(bases[:, None]*np.arange(1, 7)/giro)*giro
    pedidos = np.r_[h.ravel(), laterales.ravel(), rotacion.ravel()]
    bins, inversa = np.unique(np.rint(pedidos/paso).astype(int), return_inverse=True)
    usados, c = mapa(z, bins*paso)
    valores = agrupar(c, usados)[inversa]
    n_h, n_l = h.size, laterales.size
    curvas = valores[:n_h].reshape(5, 6, puntos, 3).transpose(0, 1, 3, 2)
    soporte = np.stack([curvas.max(axis=1), np.sort(curvas, axis=1)[:, -2:].mean(axis=1),
        np.minimum(curvas[:, 0], curvas[:, 1:].max(axis=1))], axis=1)
    l = valores[n_h:n_h+n_l].reshape(5, 3, 2, puntos, 3).transpose(0, 1, 2, 4, 3)
    simetrico = np.minimum(l[:, :, 0], l[:, :, 1])
    giro_valor = valores[n_h+n_l:].reshape(5, 6, 3)
    pico = curvas.max(axis=-1)
    especificidad = (pico-giro_valor)/np.maximum(pico+giro_valor, np.finfo(float).tiny)
    salida = np.r_[resumir(curvas, delta), resumir(soporte, delta),
        simetrico.max(axis=-1).ravel(), simetrico.mean(axis=-1).ravel(), especificidad.ravel()]
    assert salida.shape == (990,) and np.isfinite(salida).all()
    return salida


def comprobar():
    rng = np.random.default_rng(990)
    c = {k: 30*v for k, v in F.MULTIPLICADORES['DE'].items()}
    for fs in (12000, 48000):
        x = rng.normal(size=6*fs+5)
        a = medir(x, fs, c)
        y = x.copy()
        y[len(x)//2:] = rng.normal(scale=100, size=len(x)-len(x)//2)
        assert np.array_equal(a, medir(y, fs, c))
        assert np.allclose(a, medir(7*x, fs, c), rtol=1e-8, atol=1e-8)
    print('OK: ACP990 invariante al futuro y escala, 12/48 kHz; todas las familias y rejilla común.', flush=True)


def preparar():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_temporales_acp/comparacion_133.csv'
    d = pd.read_csv(fuente)
    protocolo = dict(codigo=firma(Path(__file__)), nucleo=firma(AQUI / 'acp_denso.py'),
        fuente=firma(fuente), giro=firma(AQUI / 'rasgos_fisicos.py'),
        agregacion=firma(AQUI / 'coherencia_ciclica.py'),
        entrada='senal, fs y catalogo completo; ninguna etiqueta o frecuencia verdadera',
        tramo='hasta 5 s desde inicio, siempre dentro de la primera mitad; corte antes de remuestreo',
        rejilla='común +/-2%, paso <=mitad del bin cíclico en H6; FFT compartida exacta',
        resolucion='física FS/L, distinta del paso de muestreo FS/NFFT',
        rasgos='990: soporte H1..6, desviación firmada, +/-giro/FTF/2FTF, comparación con orden de eje',
        proposito='solo identificar tipo; no modifica el juez ni su espectro')
    rp = SALIDA / 'protocolo_preparacion.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    p = SALIDA / 'progreso.npz'
    x, hechos = np.full((133, 990), np.nan), np.zeros(133, dtype=bool)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            x, hechos = z['x'], z['hechos']
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
            np.savez_compressed(p, x=x, hechos=hechos)
            print('ACP denso:', int(hechos.sum()), '/133', flush=True)
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
