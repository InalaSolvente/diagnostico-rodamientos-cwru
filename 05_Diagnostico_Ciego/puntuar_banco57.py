"""Puntúa tipos ya congelados con banco57; ninguna etiqueta elige una banda."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from banco_desplazado import seleccionar_57
from doble_banda_fijada import verificar_dos
from puntuar_tipos_fijos import contar

AQUI = Path(__file__).parent


def evaluar(d, salida):
    salida = Path(salida)
    congelado = pd.read_csv(salida / 'tipos_133.csv')
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    assert congelado.archivo.equals(d.archivo) and congelado.prediccion.equals(d.prediccion)
    p = AQUI / 'resultados_banco_desplazado'
    original = pd.read_csv(p / 'comparacion_133.csv').set_index('archivo')
    cache = {q['archivo']: q for q in json.loads((p / 'evidencia.json').read_text(encoding='utf-8'))}
    procedencia = dict(codigo=firma(Path(__file__)), tipos_congelados=firma(salida / 'tipos_133.csv'),
        cache_evidencia=firma(p / 'evidencia.json'), cache_datos=firma(p / 'comparacion_133.csv'),
        selector=firma(AQUI / 'banco_desplazado.py'), verificador=firma(AQUI / 'doble_banda_fijada.py'),
        clave_cache='archivo, sha256 de señal y tipo predicho; no etiqueta verdadera',
        etiquetas_para_seleccionar=False)
    (salida / 'procedencia_grados.json').write_text(json.dumps(procedencia, ensure_ascii=False, indent=2), encoding='utf-8')
    grados, evidencia = [], []
    for r in d.itertuples():
        assert original.loc[r.archivo, 'sha256_senal'] == r.sha256_senal
        if r.prediccion in ('N', 'sin_diagnostico'):
            v = dict(grado='ND', ramas=[])
        elif cache[r.archivo]['tipo'] == r.prediccion:
            v = {k: cache[r.archivo][k] for k in ('grado', 'ramas')}
        else:
            mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
            x = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            mitad = len(x)//2
            catalogo = json.loads(r.catalogo)
            s = seleccionar_57(x[:mitad], int(r.fs), catalogo, r.prediccion)
            v = verificar_dos(x[mitad:], int(r.fs), catalogo, s)
            print('Banco57 para tipo congelado:', r.archivo, r.prediccion, flush=True)
        grados.append(v['grado'])
        evidencia.append(dict(archivo=r.archivo, tipo=r.prediccion, sha256_senal=r.sha256_senal, **v))
    d = d.copy()
    d['grado'] = grados
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d['tipo_diagnosticado'] = d.prediccion.where(d.grado.ne('ND'))
    d.to_csv(salida / 'comparacion_133.csv', index=False)
    (salida / 'evidencia.json').write_text(json.dumps(evidencia, ensure_ascii=False, indent=2), encoding='utf-8')
    resumen = contar(d)
    resumen.update(conteo_ND_emitido=contar(d, nd_emitido=True),
        normales_correctos=int(d.loc[d.real.eq('N'), 'tipo_correcto'].sum()),
        validacion_externa=False, desplegado=False)
    (salida / 'resumen.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resumen), flush=True)
    return d, resumen
