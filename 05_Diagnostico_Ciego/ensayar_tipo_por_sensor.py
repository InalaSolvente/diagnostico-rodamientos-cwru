"""Experimento de desarrollo: tipo por sensor, detector y juez sin cambios.

No elige variantes ni umbrales con el total previamente observado. La muestra
de desarrollo ya fue usada en otros ensayos: estos resultados son exploratorios.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import crear, calidad, firma
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
FUENTE = AQUI/'resultados_dominios_desarrollo'
SALIDA = AQUI/'resultados_tipo_por_sensor'


def predecir_tipo(clf, ventanas):
    scores = clf.decision_function(ventanas).mean(axis=0)
    return str(clf.classes_[scores.argmax()])


def main():
    SALIDA.mkdir(exist_ok=True)
    meta = pd.read_csv(FUENTE/'muestra.csv')
    with np.load(FUENTE/'rasgos.npz', allow_pickle=False) as z:
        x, origen = z['x'], z['origen']
    assert x.shape == (224, 1953) and meta.vista_uid.is_unique
    assert meta.archivo.nunique() == 54 and meta.real.ne('N').sum() == 52
    assert np.array_equal(np.bincount(origen), np.full(len(meta), 4))
    gates = pd.read_csv(AQUI/'resultados_por_dominio/tipos_oof.csv').set_index('vista_uid')
    gates = gates.prediccion.reindex(meta.vista_uid).to_numpy()
    assert not pd.isna(gates).any()
    protocolo = dict(version='tipo_por_sensor_lda_fijo_v1',
        hipotesis='evitar mezclar transferencias DE/FE en el clasificador del tipo',
        modelo='StandardScaler + LDA lsqr shrinkage auto; solo B/IR/OR por sensor',
        detector='OOF historico por sensor, sin modificar',
        exterior='condicion completa fuera de ambos sensores; todas sus ventanas juntas',
        seleccion='una unica variante fijada antes de calcular resultados',
        limitacion='desarrollo reutilizado; no validacion externa ni reserva virgen',
        codigo=firma(Path(__file__)), fuente=firma(FUENTE/'rasgos.npz'),
        muestra=firma(FUENTE/'muestra.csv'),
        detector_sha256=firma(AQUI/'resultados_por_dominio/tipos_oof.csv'),
        bandas=firma(AQUI/'bandas_estables.py'), criterio=firma(AQUI/'criterio_usuario.py'))
    ruta = SALIDA/'protocolo.json'
    if ruta.exists():
        assert json.loads(ruta.read_text(encoding='utf-8')) == protocolo
    else:
        ruta.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy(); sensor = meta.sensor.to_numpy()
    pred = np.full(len(meta), '', dtype=object); folds = []
    with threadpool_limits(limits=1):
        for grupo in sorted(set(grupos)):
            for dominio in ('DE', 'FE'):
                test = np.flatnonzero((grupos == grupo) & (sensor == dominio))
                if not len(test):
                    continue
                train = (grupos != grupo) & (sensor == dominio) & (y != 'N')
                ft = train[origen]
                assert grupo not in set(grupos[train])
                assert set(y[origen[ft]]) == {'B', 'IR', 'OR'}
                assert set(meta.loc[train, 'archivo']).isdisjoint(meta.loc[test, 'archivo'])
                clf = crear().fit(x[ft], y[origen[ft]])
                for i in test:
                    pred[i] = 'N' if gates[i] == 'N' else predecir_tipo(clf, x[origen == i])
                folds.append(dict(grupo=grupo, sensor=dominio,
                    train=meta.loc[train, 'vista_uid'].tolist(),
                    test=meta.loc[test, 'vista_uid'].tolist()))
            print('OOF tipo por sensor:', grupo, flush=True)
    assert all(pred != '')
    d = meta[['archivo', 'vista_uid', 'sensor', 'real']].assign(prediccion=pred)
    d.to_csv(SALIDA/'tipos_oof.csv', index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    # Las decisiones ya estan fijadas. El cache recupera evidencia solo para
    # el mismo archivo y tipo predicho; ninguna etiqueta elige la prediccion.
    previo = AQUI/'comparacion_con_sin_esperada_todos'
    config_previa = json.loads((previo/'configuracion.json').read_text(encoding='utf-8'))
    for nombre in ('bandas_estables.py', 'criterio_usuario.py'):
        assert config_previa['hashes'][nombre] == firma(AQUI/nombre)
    huellas = pd.read_csv(previo/'comparacion_133.csv').set_index('archivo').sha256_senal
    evidencia = json.loads((previo/'evidencia.json').read_text(encoding='utf-8'))
    cache = {}
    for fila in evidencia:
        for escenario in ('sin_esperada', 'con_esperada'):
            q = fila[escenario]
            if q and 'seleccion' in q:
                cache[(fila['archivo'], q['tipo'])] = {
                    k: q[k] for k in ('tipo', 'grado', 'seleccion', 'verificacion')}
    base = pd.read_csv(AQUI/'resultados_jerarquico/comparacion.csv').set_index('vista_uid')
    filas = []
    for i, r in enumerate(meta.itertuples()):
        q = cache.get((r.archivo, pred[i])) if r.sensor == ('FE' if '/12k_FE/' in '/'+r.archivo else 'DE') else None
        if q is not None:
            assert r.sha256_senal == huellas.loc[r.archivo]
        else:
            mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            senal = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            q = calidad(senal, int(r.fs_original), json.loads(r.catalogo), str(pred[i]))
        filas.append(dict(vista_uid=r.vista_uid, **q))
        print(f'Juez fijo {i+1}/{len(meta)}: {pred[i]} {q["grado"]}', flush=True)
    d['grado'] = [q['grado'] for q in filas]
    d['paper'] = d.vista_uid.map(base.paper)
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'comparacion.csv', index=False)
    (SALIDA/'predicciones.json').write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding='utf-8')
    f = d[d.paper.notna()]; delta = f.grado.map(ORDEN)-f.paper.map(ORDEN); ok = f.tipo_correcto
    resumen = dict(total=len(f), tipo_correcto=int(ok.sum()),
        mejoran=int((ok & delta.gt(0)).sum()), iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()), tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),
        normales_vistas_correctas=int(d.loc[d.real.eq('N'), 'tipo_correcto'].sum()),
        despliegue=False, evaluacion='desarrollo OOF por condicion; exploratoria')
    assert resumen['total'] == 52
    assert sum(resumen[k] for k in ('mejoran', 'iguales', 'empeoran', 'tipo_incorrecto')) == 52
    (SALIDA/'resumen.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resumen, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
