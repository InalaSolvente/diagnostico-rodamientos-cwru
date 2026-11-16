"""LDA de tipo entrenado con un promedio por registro, igual que en inferencia.

Las cuatro ventanas solapadas no se tratan como cuatro adquisiciones distintas.
Una unica variante fija; detector, filtro y juez se conservan.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
FUENTE = AQUI/'resultados_dominios_desarrollo'
SALIDA = AQUI/'resultados_tipo_promedio'


def main():
    SALIDA.mkdir(exist_ok=True)
    meta = pd.read_csv(FUENTE/'muestra.csv')
    with np.load(FUENTE/'rasgos.npz', allow_pickle=False) as z:
        x, origen = z['x'], z['origen']
    assert x.shape == (224, 1953) and np.array_equal(np.bincount(origen), np.full(56, 4))
    promedios = np.asarray([x[origen == i].mean(axis=0) for i in range(len(meta))])
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    gate = pd.read_csv(AQUI/'resultados_por_dominio/tipos_oof.csv').set_index('vista_uid')
    gate = gate.prediccion.reindex(meta.vista_uid).to_numpy()
    assert not pd.isna(gate).any()
    protocolo = dict(version='tipo_promedio_fijo_v1',
        hipotesis='ajustar covarianza sobre registros promediados como en inferencia',
        entrenamiento='un promedio por registro; LDA lsqr shrinkage auto, solo B/IR/OR',
        exterior='condicion completa fuera, ambos sensores y todas las ventanas juntos',
        seleccion='una variante fija; no se cambian detector, juez ni filtros',
        limitacion='desarrollo ya reutilizado; evidencia exploratoria',
        codigo=firma(Path(__file__)), rasgos=firma(FUENTE/'rasgos.npz'),
        muestra=firma(FUENTE/'muestra.csv'), detector=firma(AQUI/'resultados_por_dominio/tipos_oof.csv'))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    pred = np.full(len(meta), '', dtype=object); folds = []
    with threadpool_limits(limits=1):
        for grupo in sorted(set(grupos)):
            train = (grupos != grupo) & (y != 'N'); test = np.flatnonzero(grupos == grupo)
            assert set(grupos[train]).isdisjoint(grupos[test])
            assert set(meta.loc[train, 'archivo']).isdisjoint(meta.loc[test, 'archivo'])
            assert set(y[train]) == {'B', 'IR', 'OR'}
            clf = crear().fit(promedios[train], y[train])
            for i in test:
                pred[i] = 'N' if gate[i] == 'N' else predecir_tipo(clf, promedios[i:i+1])
            folds.append(dict(grupo=grupo, train=meta.loc[train, 'vista_uid'].tolist(),
                test=meta.loc[test, 'vista_uid'].tolist()))
            print('OOF promedio:', grupo, flush=True)
    assert all(pred != '')
    d = meta[['archivo', 'vista_uid', 'sensor', 'real', 'grupo']].assign(prediccion=pred)
    d.to_csv(SALIDA/'tipos_oof.csv', index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    # Solo se puntua la identificacion de tipo aqui. No se inventa grado ND
    # para un error ni se reutiliza la frecuencia verdadera para predecir.
    f = d[d.real.ne('N')]
    resumen = dict(total=len(f), tipo_correcto=int(f.real.eq(f.prediccion).sum()),
        tipo_incorrecto=int(f.real.ne(f.prediccion).sum()),
        normales_vistas_correctas=int(d.loc[d.real.eq('N')].prediccion.eq('N').sum()),
        evaluacion='solo tipo, OOF por condicion; falta evaluar calidad espectral',
        desplegado=False)
    assert len(f) == 52
    (SALIDA/'resumen_tipo.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resumen, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
