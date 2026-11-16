"""Recalcula TODA la evidencia84 usada por el clasificador desde la primera mitad."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from bandas_estables import seleccionar
from clasificar_evidencia_seleccion import extraer, comprobar, TIPOS

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_juez_aprendido_acp'


def main():
    d = pd.read_csv(AQUI / 'resultados_crudo_coherencia/comparacion_133.csv')
    with np.load(SALIDA / 'rasgos_juez84.npz', allow_pickle=False) as z:
        guardados = z['x']
    recalculados = np.zeros((133, 84))
    usados = d.suficiente & d.detector_auditoria.ne('N')
    n, maximo = 0, 0.
    for i, r in enumerate(d.itertuples()):
        if not usados.iloc[i]:
            continue
        mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
        senal = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
        assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        mitad = len(senal)//2
        # Las TRES familias, sin consultar la etiqueta ni el tipo predicho.
        alternativas = {t: seleccionar(senal[:mitad], int(r.fs), json.loads(r.catalogo), t) for t in TIPOS}
        recalculados[i] = extraer(alternativas)
        diferencia = float(np.max(abs(recalculados[i]-guardados[i])))
        maximo = max(maximo, diferencia)
        assert np.allclose(recalculados[i], guardados[i], rtol=1e-10, atol=1e-10), (r.archivo, diferencia)
        if n == 0:
            comprobar(alternativas)
        n += 1
        if n % 20 == 0:
            print('Juez84 desde MAT:', n, '/', int(usados.sum()), flush=True)
    assert n == usados.sum() and np.allclose(recalculados, guardados, rtol=1e-10, atol=1e-10)
    np.savez_compressed(SALIDA / 'rasgos_juez84_desde_MAT.npz', x=recalculados)
    resultado = dict(codigo=firma(Path(__file__)), extractor=firma(AQUI / 'clasificar_evidencia_seleccion.py'),
        selector=firma(AQUI / 'bandas_estables.py'), criterio=firma(AQUI / 'criterio_usuario.py'),
        rasgos_guardados=firma(SALIDA / 'rasgos_juez84.npz'),
        registros_recalculados=n, hipotesis_por_registro=3, diferencia_maxima=maximo,
        etiqueta_o_tipo_predicho_para_elegir=False, tramo='solo primera mitad', verificado=True)
    (SALIDA / 'verificacion_fuente84.json').write_text(json.dumps(resultado, indent=2), encoding='utf-8')
    print(json.dumps(resultado), flush=True)


if __name__ == '__main__':
    main()
