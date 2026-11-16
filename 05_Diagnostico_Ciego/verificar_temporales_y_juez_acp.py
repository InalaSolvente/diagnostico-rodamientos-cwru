"""Comprueba exclusiones, reproduce ajustes y vuelve a extraer TODAS las vistas."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from evaluar_retirada_control_pd import controlar
from rasgos_temporales_acp import medir, posiciones, comprobar
from banco_desplazado import seleccionar_57
from doble_banda_fijada import verificar_dos
from puntuar_tipos_fijos import contar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
TEMP = AQUI / 'resultados_temporales_acp'
JUEZ = AQUI / 'resultados_juez_aprendido_acp'


def main():
    comprobar()
    referencia = pd.read_csv(AQUI / 'resultados_crudo_coherencia/comparacion_133.csv')
    d = pd.read_csv(TEMP / 'comparacion_133.csv')
    j = pd.read_csv(JUEZ / 'comparacion_133.csv')
    for p, datos, codigo, nombre in [
        (TEMP, d, 'evaluar_temporales_acp.py', 'protocolo_evaluacion.json'),
        (JUEZ, j, 'evaluar_juez_aprendido_acp.py', 'protocolo.json')]:
        assert len(datos) == 133 and datos.archivo.is_unique
        for c in ('archivo', 'sha256_senal', 'sensor', 'fs', 'real'):
            assert datos[c].equals(referencia[c])
        assert datos.paper.fillna('').equals(referencia.paper.fillna(''))
        assert datos.tipo_correcto.equals(datos.real.eq(datos.prediccion))
        pr = json.loads((p / nombre).read_text(encoding='utf-8'))
        assert pr['codigo'] == firma(AQUI / codigo)
        assert pr['puntuador'] == firma(AQUI / 'puntuar_banco57.py')
        congelado = json.loads((p / 'procedencia_grados.json').read_text(encoding='utf-8'))
        assert congelado['tipos_congelados'] == firma(p / 'tipos_133.csv')
        resumen = json.loads((p / 'resumen.json').read_text(encoding='utf-8'))
        assert all(resumen[k] == v for k, v in contar(datos).items())
        assert int(datos.loc[datos.real.eq('N'), 'tipo_correcto'].sum()) == 4
    fuente84 = json.loads((JUEZ / 'verificacion_fuente84.json').read_text(encoding='utf-8'))
    assert fuente84['verificado'] and fuente84['rasgos_guardados'] == firma(JUEZ / 'rasgos_juez84.npz')
    with np.load(TEMP / 'rasgos.npz', allow_pickle=False) as z:
        xt = z['x']
    with np.load(JUEZ / 'rasgos_juez84_desde_MAT.npz', allow_pickle=False) as z:
        evidencia84 = z['x']
    # La primera vista es exactamente el modelo crudo+ACP anterior.
    xj = np.concatenate([xt[:, :4], np.repeat(evidencia84[:, None], 4, axis=1)], axis=2)
    assert xt.shape == (133, 12, 2621) and xj.shape == (133, 4, 2705)
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    assert dev.sum() == 52
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert not any('@' in q for q in g)
    alternativas = {q['archivo']: q['alternativas_seleccion'] for q in json.loads(
        (AQUI / 'resultados_correccion_d_persistente/evidencia.json').read_text(encoding='utf-8'))}
    ajustes = 0
    with threadpool_limits(limits=1):
        for p, datos, x, vistas, con_control in [(TEMP, d, xt, 12, True), (JUEZ, j, xj, 4, False)]:
            folds = json.loads((p / 'pliegues.json').read_text(encoding='utf-8'))
            for q in folds:
                tr = dev & (g != q['grupo_exterior'])
                assert q['train'] == d.loc[tr, 'archivo'].tolist()
                te = np.flatnonzero((g == q['grupo_exterior']) & d.suficiente.to_numpy() & d.detector_auditoria.ne('N').to_numpy())
                clf = crear().fit(x[tr].reshape(-1, x.shape[-1]), np.repeat(y[tr], vistas))
                base = [predecir_tipo(clf, x[i]) for i in te]
                pred = ([controlar(t, d.iloc[i].sensor, alternativas[d.iloc[i].archivo], usar_d=True, usar_pd=True)
                         for t, i in zip(base, te)] if con_control else base)
                assert pred == datos.loc[te, 'prediccion'].tolist()
                if con_control:
                    assert base == datos.loc[te, 'tipo_base_temporal'].tolist()
                ajustes += 1
    assert ajustes == 22
    evidencia = {q['archivo']: q for q in json.loads((TEMP / 'evidencia.json').read_text(encoding='utf-8'))}
    cambios_tipo = d.prediccion.ne(referencia.prediccion)
    revisados, grados, futuro = 0, 0, set()
    for i, r in enumerate(d.itertuples()):
        if not r.suficiente:
            assert r.prediccion == 'sin_diagnostico'
            continue
        mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
        x = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
        assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        c = json.loads(r.catalogo)
        assert all(v+int(2.5*r.fs) <= len(x)//2 for v in posiciones(len(x), int(r.fs)))
        actual = medir(x, int(r.fs), c)
        diferencia = float(np.max(abs(actual-xt[i])))
        assert np.allclose(actual, xt[i], rtol=1e-10, atol=1e-10), (r.archivo, diferencia)
        if int(r.fs) not in futuro:
            otro = x.copy()
            otro[len(x)//2:] = np.random.default_rng(193).normal(scale=50, size=len(x)-len(x)//2)
            assert np.array_equal(actual, medir(otro, int(r.fs), c))
            futuro.add(int(r.fs))
        if cambios_tipo.iloc[i] and r.prediccion in ('B', 'IR', 'OR'):
            mitad = len(x)//2
            s = seleccionar_57(x[:mitad], int(r.fs), c, r.prediccion)
            v = verificar_dos(x[mitad:], int(r.fs), c, s)
            assert v == {k: evidencia[r.archivo][k] for k in ('grado', 'ramas')}
            grados += 1
        revisados += 1
        if revisados % 20 == 0:
            print('Tres vistas desde MAT:', revisados, '/132', flush=True)
    assert revisados == 132 and futuro == {12000, 48000}
    resultado = dict(codigo=firma(Path(__file__)), ajustes_reproducidos=ajustes,
        archivos_recalculados=revisados, vistas_por_archivo=12, nuevas_vistas_sin_futuro=True,
        futuro_invariante=sorted(futuro), grados_recalculados=grados,
        fuente84=firma(JUEZ / 'verificacion_fuente84.json'), entrenamiento_52_grupos_completos=True)
    f = d.paper.notna()
    antes = referencia.tipo_correcto & referencia.grado.map(ORDEN).ge(referencia.paper.map(ORDEN))
    for p, datos in [(TEMP, d), (JUEZ, j)]:
        ahora = datos.tipo_correcto & datos.grado.map(ORDEN).ge(datos.paper.map(ORDEN))
        pares = datos[['archivo', 'real', 'paper', 'prediccion', 'grado']].copy()
        pares['prediccion_previa'], pares['grado_previo'] = referencia.prediccion, referencia.grado
        pares['acierto_previo'], pares['acierto_ahora'] = antes, ahora
        pares[f & (datos.prediccion.ne(referencia.prediccion) | datos.grado.ne(referencia.grado))].to_csv(p / 'cambios_vs_82.csv', index=False)
        balance = dict(recuperados=int((f & ~antes & ahora).sum()), perdidos=int((f & antes & ~ahora).sum()))
        informe = dict(resultado, comparacion_pareada=balance, sin_retroceso=contar(datos)['sin_retroceso'])
        (p / 'verificacion.json').write_text(json.dumps(informe, indent=2), encoding='utf-8')
        print(json.dumps(dict(carpeta=p.name, **informe)), flush=True)


if __name__ == '__main__':
    main()
