"""Reproduce ACP desde todos los MAT y los ajustes exteriores sin etiqueta evaluada."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from coherencia_ciclica import medir, comprobar, SALIDA as CS
from evaluar_coherencia_ciclica import predecir
from evaluar_selector_controles_anidado import elegir
from evaluar_retirada_control_pd import controlar
from modelo_ciego import F
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from banco_desplazado import seleccionar_57
from doble_banda_fijada import verificar_dos
from puntuar_tipos_fijos import contar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
COMBO = AQUI / 'resultados_crudo_coherencia'


def main():
    comprobar()
    solo = pd.read_csv(CS / 'comparacion_133.csv')
    d = pd.read_csv(COMBO / 'comparacion_133.csv')
    base = pd.read_csv(AQUI / 'resultados_banco_desplazado/comparacion_133.csv')
    assert len(d) == len(solo) == len(base) == 133
    for columna in ('archivo', 'sha256_senal', 'sensor', 'fs', 'real'):
        assert d[columna].equals(base[columna]) and solo[columna].equals(base[columna])
    assert d.paper.fillna('').equals(base.paper.fillna(''))
    assert solo.paper.fillna('').equals(base.paper.fillna(''))
    for p, datos, codigo, protocolo in [
        (CS, solo, 'evaluar_coherencia_ciclica.py', 'protocolo_evaluacion.json'),
        (COMBO, d, 'evaluar_crudo_coherencia.py', 'protocolo.json')]:
        pr = json.loads((p / protocolo).read_text(encoding='utf-8'))
        assert pr['codigo'] == firma(AQUI / codigo)
        assert pr['puntuador'] == firma(AQUI / 'puntuar_banco57.py')
        trazabilidad = json.loads((p / 'procedencia_grados.json').read_text(encoding='utf-8'))
        assert trazabilidad['tipos_congelados'] == firma(p / 'tipos_133.csv')
        assert trazabilidad['selector'] == firma(AQUI / 'banco_desplazado.py')
        resumen = json.loads((p / 'resumen.json').read_text(encoding='utf-8'))
        assert datos.tipo_correcto.equals(datos.real.eq(datos.prediccion))
        assert all(resumen[k] == v for k, v in contar(datos).items())
    with np.load(CS / 'rasgos.npz', allow_pickle=False) as z:
        x = z['x']
    with np.load(AQUI / 'resultados_globales_arbol/rasgos.npz', allow_pickle=False) as z:
        original = z['x']
    conjunto = np.concatenate([original, np.repeat(x[:, None, :], 4, axis=1)], axis=2)
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    assert dev.sum() == 52
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert not any('@' in q for q in g)
    alternativas = {q['archivo']: q['alternativas_seleccion'] for q in json.loads(
        (AQUI / 'resultados_correccion_d_persistente/evidencia.json').read_text(encoding='utf-8'))}
    ajustes = 0
    with threadpool_limits(limits=1):
        for q in json.loads((CS / 'auditoria_anidada.json').read_text(encoding='utf-8')):
            tr = np.flatnonzero(dev & (g != q['grupo_exterior']))
            assert q['train'] == d.loc[tr, 'archivo'].tolist()
            for interior in q['internos']:
                vi = tr[g[tr] == interior['grupo_interior']]
                ajuste = tr[g[tr] != interior['grupo_interior']]
                assert interior['validacion'] == d.loc[vi, 'archivo'].tolist()
                assert interior['train'] == d.loc[ajuste, 'archivo'].tolist()
            pred_internas = {k: np.array(v) for k, v in q['oof_interno'].items()}
            mejor, scores = elegir(pred_internas, y[tr])
            assert q['modelo'] == mejor and q['scores'] == scores
            te = np.flatnonzero((g == q['grupo_exterior']) & d.suficiente.to_numpy() & d.detector_auditoria.ne('N').to_numpy())
            assert predecir(mejor, x[tr], y[tr], x[te]).tolist() == solo.loc[te, 'prediccion'].tolist()
            ajustes += 1
        for q in json.loads((COMBO / 'pliegues.json').read_text(encoding='utf-8')):
            tr = dev & (g != q['grupo_exterior'])
            assert q['train'] == d.loc[tr, 'archivo'].tolist()
            te = np.flatnonzero((g == q['grupo_exterior']) & d.suficiente.to_numpy() & d.detector_auditoria.ne('N').to_numpy())
            clf = crear().fit(conjunto[tr].reshape(-1, 2621), np.repeat(y[tr], 4))
            tb = [predecir_tipo(clf, conjunto[i]) for i in te]
            pred = [controlar(t, d.iloc[i].sensor, alternativas[d.iloc[i].archivo], usar_d=True, usar_pd=True)
                    for t, i in zip(tb, te)]
            assert tb == d.loc[te, 'tipo_base_crudo_acp'].tolist()
            assert pred == d.loc[te, 'prediccion'].tolist()
            ajustes += 1
    assert ajustes == 22
    evidencia = {q['archivo']: q for q in json.loads((COMBO / 'evidencia.json').read_text(encoding='utf-8'))}
    cambios_tipo = d.prediccion.ne(base.prediccion)
    recalculos_rasgos = recalculos_grados = 0
    futuro = set()
    for i, r in enumerate(d.itertuples()):
        if not r.suficiente:
            assert r.prediccion == 'sin_diagnostico'
            continue
        mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
        senal = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
        assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        catalogo = json.loads(r.catalogo)
        calculado = medir(senal, int(r.fs), catalogo)
        assert np.allclose(calculado, x[i], rtol=1e-12, atol=1e-12)
        recalculos_rasgos += 1
        if int(r.fs) not in futuro:
            cambiado = senal.copy()
            mitad = len(senal)//2
            cambiado[mitad:] = np.random.default_rng(291).normal(size=len(senal)-mitad)
            assert np.array_equal(calculado, medir(cambiado, int(r.fs), catalogo))
            futuro.add(int(r.fs))
        if cambios_tipo.iloc[i] and r.prediccion in ('B', 'IR', 'OR'):
            mitad = len(senal)//2
            s = seleccionar_57(senal[:mitad], int(r.fs), catalogo, r.prediccion)
            v = verificar_dos(senal[mitad:], int(r.fs), catalogo, s)
            assert v == {k: evidencia[r.archivo][k] for k in ('grado', 'ramas')}
            recalculos_grados += 1
        if recalculos_rasgos % 20 == 0:
            print('ACP desde MAT:', recalculos_rasgos, '/132', flush=True)
    assert recalculos_rasgos == 132 and futuro == {12000, 48000}
    f = d.paper.notna()
    antes = base.tipo_correcto & base.grado.map(ORDEN).ge(base.paper.map(ORDEN))
    ahora = d.tipo_correcto & d.grado.map(ORDEN).ge(d.paper.map(ORDEN))
    pares = d[['archivo', 'paper', 'real', 'prediccion', 'grado']].copy()
    pares['prediccion_previa'], pares['grado_previo'] = base.prediccion, base.grado
    pares['acierto_previo'], pares['acierto_ahora'] = antes, ahora
    pares[f & (cambios_tipo | d.grado.ne(base.grado))].to_csv(COMBO / 'cambios_vs_banco57.csv', index=False)
    balance = dict(recuperados=int((f & ~antes & ahora).sum()), perdidos=int((f & antes & ~ahora).sum()))
    assert int(ahora[f].sum()-antes[f].sum()) == balance['recuperados']-balance['perdidos'] == 2
    verificacion = dict(codigo=firma(Path(__file__)), coherencia=firma(AQUI / 'coherencia_ciclica.py'),
        extractor_crudo=firma(AQUI / 'modelo_espectral.py'), giro=firma(AQUI / 'rasgos_fisicos.py'),
        ajustes_reproducidos=ajustes, rasgos_desde_MAT=recalculos_rasgos,
        grados_recalculados_para_todos_los_tipos_nuevos=recalculos_grados,
        futuro_invariante=sorted(futuro), entrenamiento_52_y_grupos_excluidos=True,
        comparacion_pareada=balance, normal_correctos=4, objetivo_alcanzado=False)
    (COMBO / 'verificacion.json').write_text(json.dumps(verificacion, indent=2), encoding='utf-8')
    print(json.dumps(verificacion), flush=True)


if __name__ == '__main__':
    main()
