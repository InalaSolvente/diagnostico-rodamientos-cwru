"""Tres vistas temporales de selección; sin etiquetas ni acceso al tramo verificador."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from modelo_espectral import extraer
from ventanas_verificadas import prefijo
from coherencia_ciclica import medir as acp

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_temporales_acp'


def posiciones(n, fs):
    mitad = n//2
    largo = int(2.5*fs)
    if mitad < largo:
        raise ValueError('Se requieren al menos cinco segundos')
    return [0, (mitad-largo)//2, mitad-largo]


def segmento(x, fs, catalogo):
    z = prefijo(x, fs)
    c = acp(x, fs, catalogo)
    filas = []
    for inicio in (0, 6000, 12000, 18000):
        e = extraer(z[inicio:inicio+12000], 12000, catalogo)
        filas.append(np.r_[e['arbol'], e['globales'], c])
    salida = np.asarray(filas)
    assert salida.shape == (4, 2621) and np.isfinite(salida).all()
    return salida


def medir(x, fs, catalogo):
    inicios = posiciones(len(x), fs)
    largo = int(2.5*fs)
    assert all(i+largo <= len(x)//2 for i in inicios)
    # La API de prefijo recibe señal cruda >=5 s y corta ANTES de transformar.
    # Desde estos inicios, los 2.5 s consumidos terminan dentro de selección.
    return np.concatenate([segmento(x[i:], fs, catalogo) for i in inicios])


def comprobar():
    rng = np.random.default_rng(2621)
    c = {k: 30*v for k, v in F.MULTIPLICADORES['DE'].items()}
    for fs in (12000, 48000):
        n = 6*fs+1
        assert all(i+int(2.5*fs) <= n//2 for i in posiciones(n, fs))
        x = rng.normal(size=n)
        a = medir(x, fs, c)
        x[n//2:] = rng.normal(scale=100, size=n-n//2)
        assert np.array_equal(a, medir(x, fs, c))
        assert a.shape == (12, 2621)
    print('OK: tres vistas dentro de seleccion; futuro independiente a 12/48 kHz.', flush=True)


def preparar():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_crudo_coherencia/comparacion_133.csv'
    d = pd.read_csv(fuente)
    with np.load(AQUI / 'resultados_globales_arbol/rasgos.npz', allow_pickle=False) as z:
        original = z['x']
    with np.load(AQUI / 'resultados_coherencia_ciclica/rasgos.npz', allow_pickle=False) as z:
        coherencia = z['x']
    anterior = np.concatenate([original, np.repeat(coherencia[:, None, :], 4, axis=1)], axis=2)
    protocolo = dict(codigo=firma(Path(__file__)), fuente=firma(fuente),
        crudos=firma(AQUI / 'resultados_globales_arbol/rasgos.npz'),
        acp=firma(AQUI / 'resultados_coherencia_ciclica/rasgos.npz'),
        extractor=firma(AQUI / 'modelo_espectral.py'), coherencia=firma(AQUI / 'coherencia_ciclica.py'),
        prefijo=firma(AQUI / 'ventanas_verificadas.py'),
        vistas='tres ventanas de 2.5 s: inicio, centro y final de la primera mitad',
        entrada='senal, fs, catalogo completo; ningun tipo ni frecuencia verdadera',
        cuenta='133 archivos; 12 vistas correlacionadas por archivo, no casos independientes')
    rp = SALIDA / 'protocolo_preparacion.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    p = SALIDA / 'progreso.npz'
    x, hechos = np.full((133, 12, 2621), np.nan), np.zeros(133, dtype=bool)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            x, hechos = z['x'], z['hechos']
    comprobados = set()
    for i, r in enumerate(d.itertuples()):
        if hechos[i]:
            continue
        if r.suficiente:
            mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
            senal = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            c = json.loads(r.catalogo)
            inicios, largo = posiciones(len(senal), int(r.fs)), int(2.5*r.fs)
            if int(r.fs) not in comprobados:
                recalculado = segmento(senal, int(r.fs), c)
                assert np.allclose(recalculado, anterior[i], rtol=1e-10, atol=1e-10)
                comprobados.add(int(r.fs))
            x[i] = np.concatenate([anterior[i]] + [segmento(senal[j:], int(r.fs), c) for j in inicios[1:]])
        hechos[i] = True
        if (i+1) % 20 == 0 or i == len(d)-1:
            np.savez_compressed(p, x=x, hechos=hechos)
            print('Vistas temporales:', int(hechos.sum()), '/133', flush=True)
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
