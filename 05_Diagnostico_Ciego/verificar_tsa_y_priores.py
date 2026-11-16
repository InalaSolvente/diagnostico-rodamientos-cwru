"""Verifica TSA desde MAT, controles nulos y el clasificador con priors uniformes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from tsa_controlado import aplicar, comprobar, FS
from evaluar_temporal_priores import crear_equilibrado
from modelo_ciego import F
from modelo_por_dominio import firma
from ensayar_tipo_por_sensor import predecir_tipo
from evaluar_retirada_control_pd import controlar
from banco_desplazado import seleccionar_57
from doble_banda_fijada import verificar_dos
from puntuar_tipos_fijos import contar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
TSA = AQUI / 'resultados_tsa_grados'
PRI = AQUI / 'resultados_temporal_priores'
BASE = AQUI / 'resultados_temporales_acp'


def diccionarios(v):
    if isinstance(v, dict):
        yield v
        for a in v.values():
            yield from diccionarios(a)
    elif isinstance(v, list):
        for a in v:
            yield from diccionarios(a)


def main():
    comprobar()
    base = pd.read_csv(BASE / 'comparacion_133.csv')
    tsa = pd.read_csv(TSA / 'comparacion_133.csv')
    pri = pd.read_csv(PRI / 'comparacion_133.csv')
    assert len(base) == len(tsa) == len(pri) == 133
    for datos in (tsa, pri):
        for c in ('archivo', 'sha256_senal', 'sensor', 'fs', 'real'):
            assert datos[c].equals(base[c])
        assert datos.paper.fillna('').equals(base.paper.fillna(''))
        assert datos.tipo_correcto.equals(datos.real.eq(datos.prediccion))
        esperado = datos.prediccion.where(datos.grado.ne('ND'))
        assert datos.tipo_diagnosticado.fillna('').equals(esperado.fillna(''))
    assert tsa.prediccion.equals(base.prediccion)
    p_tsa = json.loads((TSA / 'protocolo.json').read_text(encoding='utf-8'))
    assert p_tsa['codigo'] == firma(AQUI / 'evaluar_tsa_grados.py')
    assert p_tsa['filtro'] == firma(AQUI / 'tsa_controlado.py')
    migracion = json.loads((TSA / 'migracion_metadato.json').read_text(encoding='utf-8'))
    despues = {n: firma(TSA / n) for n in migracion['antes']}
    assert despues['evidencia.json'] == migracion['antes']['evidencia.json']
    antes_csv = pd.read_csv(TSA / 'comparacion_133_v2_diagnostico_previo.csv')
    assert antes_csv.drop(columns=['tipo_diagnosticado']).equals(tsa.drop(columns=['tipo_diagnosticado']))
    antes_tipos = pd.read_csv(TSA / 'tipos_133_v2_diagnostico_previo.csv')
    assert antes_tipos.drop(columns=['tipo_diagnosticado']).equals(pd.read_csv(TSA / 'tipos_133.csv'))
    migracion.update(despues=despues, salidas_numericas_sin_cambios=True, codigo_actual=p_tsa['codigo'],
                     diagnostico_exportado_corregido=True, decisiones_y_grados_sin_cambios=True)
    (TSA / 'migracion_metadato.json').write_text(json.dumps(migracion, indent=2), encoding='utf-8')
    p_pri = json.loads((PRI / 'protocolo.json').read_text(encoding='utf-8'))
    assert p_pri['codigo'] == firma(AQUI / 'evaluar_temporal_priores.py')
    assert p_pri['rasgos'] == firma(BASE / 'rasgos.npz')
    assert p_pri['verificacion_rasgos'] == firma(BASE / 'verificacion.json')
    for p, datos in ((TSA, tsa), (PRI, pri)):
        r = json.loads((p / 'resumen.json').read_text(encoding='utf-8'))
        assert all(r[k] == v for k, v in contar(datos).items())
        assert datos.loc[datos.real.eq('N'), 'tipo_correcto'].sum() == 4
    with np.load(BASE / 'rasgos.npz', allow_pickle=False) as z:
        x = z['x']
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = base.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    assert dev.sum() == 52
    y, g = base.real.to_numpy(), base.grupo_auditoria.to_numpy()
    alternativas = {q['archivo']: q['alternativas_seleccion'] for q in json.loads(
        (AQUI / 'resultados_correccion_d_persistente/evidencia.json').read_text(encoding='utf-8'))}
    ajustes = 0
    with threadpool_limits(limits=1):
        for q in json.loads((PRI / 'pliegues.json').read_text(encoding='utf-8')):
            tr = dev & (g != q['grupo_exterior'])
            assert q['train'] == base.loc[tr, 'archivo'].tolist()
            assert q['cuentas_train'] == {c: int(np.sum(y[tr] == c)) for c in ('B', 'IR', 'OR')}
            clf = crear_equilibrado().fit(x[tr].reshape(-1, 2621), np.repeat(y[tr], 12))
            assert np.allclose(clf.steps[-1][1].priors_, np.full(3, 1/3))
            te = np.flatnonzero((g == q['grupo_exterior']) & base.suficiente.to_numpy() & base.detector_auditoria.ne('N').to_numpy())
            tb = [predecir_tipo(clf, x[i]) for i in te]
            pred = [controlar(t, base.iloc[i].sensor, alternativas[base.iloc[i].archivo], usar_d=True, usar_pd=True)
                    for t, i in zip(tb, te)]
            assert tb == pri.loc[te, 'tipo_base_priores'].tolist()
            assert pred == pri.loc[te, 'prediccion'].tolist()
            ajustes += 1
    assert ajustes == 11
    evidencia = {q['archivo']: q for q in json.loads((TSA / 'evidencia.json').read_text(encoding='utf-8'))}
    plantillas, grados, futuro, chequeos_d = 0, 0, set(), 0
    for r in tsa.itertuples():
        q = evidencia[r.archivo]
        if q['tsa'] is None:
            assert r.prediccion in ('N', 'sin_diagnostico')
            continue
        mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
        original = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
        assert hashlib.sha256(original.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        c = json.loads(r.catalogo)
        a, b, m = aplicar(original, int(r.fs), c)
        assert m == q['tsa']
        coef = np.fft.rfft(np.array(m['plantilla']))
        assert np.max(abs(coef[m['ordenes_protegidos']]), initial=0.) < 1e-9
        if int(r.fs) not in futuro:
            otro = original.copy()
            otro[len(otro)//2:] = np.random.default_rng(793).normal(size=len(otro)-len(otro)//2)
            aa, _, mm = aplicar(otro, int(r.fs), c)
            assert np.array_equal(a, aa) and mm == m
            futuro.add(int(r.fs))
        viejo = base.loc[base.archivo.eq(r.archivo)].iloc[0]
        if r.grado != viejo.grado or plantillas == 0:
            s = seleccionar_57(a, FS, c, r.prediccion)
            v = verificar_dos(b, FS, c, s)
            assert v == {k: q[k] for k in ('grado', 'ramas')}
            grados += 1
        for v in diccionarios(q):
            if v.get('grado') == 'D' and 'armonicas' in v:
                primera = next(a for a in v['armonicas'] if a['h'] == 1)
                assert primera['significativa'] and primera['potencia'] >= v['max_izquierda']
                assert sum(a['significativa'] for a in v['armonicas']) >= 2
                chequeos_d += 1
        plantillas += 1
        if plantillas % 30 == 0:
            print('Plantillas TSA desde MAT:', plantillas, flush=True)
    assert plantillas == 127 and futuro == {12000, 48000}
    # Caso que SÍ tiene frontera nativa no divisible por cuatro.
    c = {k: 30*v for k, v in F.MULTIPLICADORES['DE'].items()}
    n = 8*48000+5
    t = np.arange(n)/48000
    _, residual, m = aplicar(np.cos(2*np.pi*1200*t), 48000, c)
    assert (n//2) % 4 != 0 and 40 in m['ordenes_retirados']
    assert np.sqrt(np.mean(residual[1200:-1200]**2)) < .1
    # Control nulo pequeño; no es una estimación concluyente de falsos positivos.
    nulos = []
    for sensor in ('DE', 'FE'):
        c = {k: 30*v for k, v in F.MULTIPLICADORES[sensor].items()}
        for semilla in range(3):
            rng = np.random.default_rng(950+semilla)
            t = np.arange(6*FS)/FS
            z = rng.normal(size=len(t))+2*np.cos(2*np.pi*30*t)
            a, b, _ = aplicar(z, FS, c)
            for tipo in ('B', 'IR', 'OR'):
                s = seleccionar_57(a, FS, c, tipo)
                grado = verificar_dos(b, FS, c, s)['grado']
                nulos.append(dict(sensor=sensor, semilla=semilla, hipotesis=tipo, grado=grado))
    assert len(nulos) == 18 and not any(q['grado'] == 'D' for q in nulos)
    (TSA / 'control_nulo.json').write_text(json.dumps(nulos, indent=2), encoding='utf-8')
    for p, datos in ((TSA, tsa), (PRI, pri)):
        f = datos.paper.notna()
        antes = base.tipo_correcto & base.grado.map(ORDEN).ge(base.paper.map(ORDEN))
        ahora = datos.tipo_correcto & datos.grado.map(ORDEN).ge(datos.paper.map(ORDEN))
        cambio = datos.prediccion.ne(base.prediccion) | datos.grado.ne(base.grado)
        pares = datos[['archivo', 'real', 'paper', 'prediccion', 'grado']].copy()
        pares['prediccion_previa'], pares['grado_previo'] = base.prediccion, base.grado
        pares[f & cambio].to_csv(p / 'cambios_vs_temporal84.csv', index=False)
        r = dict(codigo=firma(Path(__file__)), plantillas_desde_MAT=plantillas,
            grados_reproducidos=grados, evidencias_D_comprobadas=chequeos_d,
            ajustes_prior_uniforme_reproducidos=ajustes, futuro_invariante=sorted(futuro),
            origen_fs48_no_divisible_por_4=True, controles_nulos=len(nulos), D_nulos=0,
            tipos_iguales_a_referencia=bool(datos.prediccion.equals(base.prediccion)),
            recuperados=int((f & ~antes & ahora).sum()), perdidos=int((f & antes & ~ahora).sum()),
            sin_retroceso=contar(datos)['sin_retroceso'])
        (p / 'verificacion.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
        print(json.dumps(dict(carpeta=p.name, **r)), flush=True)


if __name__ == '__main__':
    main()
