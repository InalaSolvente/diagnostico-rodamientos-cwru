"""Cuatro ventanas iguales por registro; nunca se separan al validar por grupos."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from modelo_ciego import F
from modelo_espectral import extraer

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_ventanas_verificadas'


def prefijo(x, fs):
    if fs not in (12000,48000) or len(x)//2 < 2.5*fs:
        raise ValueError('Se necesita primera mitad de al menos 2,5 s, fs 12/48 kHz')
    z = np.asarray(x[:int(2.5*fs)], dtype=float)
    return z if fs == 12000 else resample_poly(z,1,4,padtype='line')


def rasgos(x, fs, catalogo):
    z = prefijo(x,fs)
    return np.asarray([extraer(z[inicio:inicio+12000],12000,catalogo,L=4)['arbol']
                       for inicio in (0,6000,12000,18000)])


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    meta = pd.read_csv(AQUI/'resultados_desarrollo_verificado_v2/muestra.csv')
    protocolo = dict(version='ventanas_1s_hop05_v1', ventanas_por_registro=4,
        duracion=1, salto=.5, fs=12000, seleccion_hasta_segundos=2.5,
        proposito='estimar variación temporal y promediar evidencia; mismos registros de desarrollo',
        negativos_independientes=2, negativos_ventanas=8,
        advertencia='ocho ventanas normales no equivalen a ocho rodamientos; todas se excluyen junto con su registro',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        muestra_sha256=hashlib.sha256((AQUI/'resultados_desarrollo_verificado_v2/muestra.csv').read_bytes()).hexdigest())
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    bloques = []
    for i,r in enumerate(meta.itertuples()):
        mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
        x = np.ravel(mat[r.clave_canal])
        assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        bloques.append(rasgos(x,int(r.fs_original),json.loads(r.catalogo)))
        print(f'Ventanas {i+1}/54',flush=True)
    z = np.concatenate(bloques); origen = np.repeat(np.arange(len(meta)),4)
    assert z.shape == (216,1953) and np.isfinite(z).all()
    assert np.array_equal(np.bincount(origen),np.full(54,4))
    np.savez_compressed(SALIDA/'rasgos.npz',x=z,origen=origen)
    meta.to_csv(SALIDA/'muestra.csv',index=False)
    print('216 ventanas, 54 registros; se puntúan registros, nunca ventanas como casos independientes.',flush=True)
