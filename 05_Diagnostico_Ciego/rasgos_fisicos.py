"""Interacciones físicas para TODAS las familias, sin recibir el tipo real.

No elimina picos ni modifica el espectro usado para calificar. La cercanía a
órdenes de giro solo aporta al clasificador una medida de evidencia inespecífica.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from modelo_ciego import F, validar_catalogo

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_ventanas_fisicas'


def giro_catalogo(catalogo):
    """Recupera el giro implícito en uno de los dos catálogos ya configurados."""
    validar_catalogo(catalogo,12000)
    posibles = []
    for multiplicadores in F.MULTIPLICADORES.values():
        giros = np.array([catalogo[k]/multiplicadores[k] for k in multiplicadores])
        if np.allclose(giros, np.median(giros),rtol=1e-6,atol=1e-9):
            posibles.append(float(np.median(giros)))
    if len(posibles) != 1:
        raise ValueError('Catálogo fuera de las dos geometrías configuradas; no se inventa una velocidad de giro')
    return posibles[0]


def ampliar(x,catalogo):
    x = np.atleast_2d(np.asarray(x,dtype=float))
    if x.shape[1] != 1953 or not np.isfinite(x).all():
        raise ValueError('Rasgos arbol L4 finitos requeridos')
    fr = giro_catalogo(catalogo)
    frecuencias = np.array([catalogo['BPFI'],catalogo['BPFO'],catalogo['BSF'],2*catalogo['BSF'],catalogo['FTF']])
    ordenes = frecuencias[:,None]*np.arange(1,7)/fr
    especificidad = np.clip(abs(ordenes-np.rint(ordenes))/(.02*ordenes),0,1)
    plantillas = x[:,:1860].reshape(-1,31,5,6,2)
    estados = x[:,1860:].reshape(-1,31,3)
    extras = []
    for evidencia,estado in zip(plantillas,estados):
        partes = []
        primeras = evidencia[:,:,0,0].max(axis=0)
        for f in range(5):
            c = evidencia[:,f,:,0]
            coherencia = np.minimum(c[:,0],c[:,1:].max(axis=1))
            indices = np.argsort(-coherencia,kind='stable')
            propio = (np.maximum(c,0)*especificidad[f]).mean(axis=1)
            mejores_primeras = np.sort(c[:,0])[-3:]
            # BSF y 2BSF son el mismo tipo; no compiten entre sí.
            otros = [j for j in range(5) if j != f and {j,f} != {2,3}]
            partes.extend([float(c[:,0].max()),float(mejores_primeras.mean()),
                float(coherencia.max()),float(coherencia[indices[:3]].mean()),
                float(propio.max()),float(np.sort(propio)[-3:].mean()),
                float(primeras[f]-primeras[otros].max()),
                float(estado[indices[0],1]),float(estado[indices[0],2])])
        extras.append(partes)
    z = np.column_stack([x,np.asarray(extras)])
    assert z.shape == (len(x),1998) and np.isfinite(z).all()
    return z


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_ventanas_verificadas'
    meta = pd.read_csv(fuente/'muestra.csv')
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:
        x,origen = z['x'],z['origen']
    protocolo = dict(version='coherencia_ordenes_v1', dimensiones=1998,
        descripcion='1953 rasgos originales + 45 interacciones: fundamental/otra armónica en misma banda, especificidad frente a órdenes de giro, contexto de curtosis/entropía',
        entradas='solo rasgos y catálogo completo; giro implícito en geometría preconfigurada',
        regla_grados_modificada=False, n_normales_unicos=2, independencia_fisica_normales='no verificada',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fuente_sha256=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    ampliados = np.empty((len(x),1998))
    for i,r in enumerate(meta.itertuples()):
        c = json.loads(r.catalogo)
        assert np.isclose(giro_catalogo(c),r.rpm/60,rtol=1e-10)
        ampliados[origen == i] = ampliar(x[origen == i],c)
    np.savez_compressed(SALIDA/'rasgos.npz',x=ampliados,origen=origen)
    meta.to_csv(SALIDA/'muestra.csv',index=False)
    print('Interacciones físicas calculadas para 216 ventanas/54 registros, sin tipo real ni referencia del paper.')
