"""Selección de clasificador dentro de cada pliegue, nunca con su resultado exterior."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_tipo_anidado_desarrollo'


def modelos():
    return {
        'logistica': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=2000)),
        'lda_120': lambda: make_pipeline(VarianceThreshold(), SelectKBest(f_classif, k=120), StandardScaler(),
                                       LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        'ridge': lambda: make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, class_weight='balanced', solver='lsqr')),
    }


def elegir(x, y, grupos):
    """Las validaciones internas puntúan fallas; normales conservan clase N al ajustar."""
    fallas = np.flatnonzero(y != 'N')
    scores = {nombre: [] for nombre in modelos()}
    trazas = []
    for _, prueba_rel in GroupKFold(n_splits=4).split(x[fallas], y[fallas], grupos[fallas]):
        test = fallas[prueba_rel]
        train = np.setdiff1d(np.arange(len(y)), test)
        assert set(grupos[train]).isdisjoint(set(grupos[test]))
        assert set(y[train]) == set(y)
        for nombre, crear in modelos().items():
            clf = crear().fit(x[train], y[train])
            pred = clf.predict(x[test])
            score = float(np.mean([np.mean(pred[y[test] == c] == c) for c in np.unique(y[test])]))
            scores[nombre].append(score)
        trazas.append(dict(grupos_train=sorted(set(grupos[train])), grupos_test=sorted(set(grupos[test]))))
    medios = {nombre: float(np.mean(valores)) for nombre, valores in scores.items()}
    ganador = max(medios, key=medios.get)
    return ganador, medios, trazas


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_tipo_seleccion_desarrollo'
    meta = pd.read_csv(fuente/'predicciones_oof.csv')
    with np.load(fuente/'rasgos.npz', allow_pickle=False) as f:
        x = f['x']
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    assert len(meta) == 54 and meta.sha256_senal.is_unique and x.shape == (54, 1953)
    protocolo = dict(version='tipo_anidado_v1', fuente='prefijos 2 s, fs canónica 12k, arbol L4',
        modelos=list(modelos()), objetivo_interno='exactitud equilibrada de tipo en fallas',
        exterior='cada condición de falla o señal normal única fuera del ajuste y de la selección',
        interior='4 pliegues por condiciones de falla; normales solo entrenamiento interno',
        nota_normales='OOF de cada normal contra el otro; no implica rodamientos normales independientes',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fuente_sha256=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    ruta = SALIDA/'protocolo.json'
    if ruta.exists():
        assert json.loads(ruta.read_text(encoding='utf-8')) == protocolo
    else:
        ruta.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    pred = np.empty(len(y), dtype=object)
    elegidos = np.empty(len(y), dtype=object)
    auditoria = []
    for grupo in sorted(set(grupos)):
        test = grupos == grupo; train = ~test
        assert grupo not in set(grupos[train])
        nombre, scores, internos = elegir(x[train], y[train], grupos[train])
        clf = modelos()[nombre]().fit(x[train], y[train])
        pred[test] = clf.predict(x[test]); elegidos[test] = nombre
        auditoria.append(dict(grupo_exterior=grupo, train=meta.loc[train,'archivo'].tolist(),
            test=meta.loc[test,'archivo'].tolist(), modelo_elegido=nombre, scores_internos=scores,
            validaciones_internas=internos))
        meta.assign(prediccion_anidada=pred, modelo_anidado=elegidos).iloc[0:0]  # sin guardar filas no predichas
        print(f'Grupo {grupo[:25]}: {nombre}', flush=True)
    d = meta.drop(columns=['prediccion','acierto']).assign(prediccion=pred, modelo=elegidos, acierto=pred == y)
    d.to_csv(SALIDA/'predicciones_oof.csv', index=False)
    (SALIDA/'auditoria_pliegues.json').write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding='utf-8')
    print(d.groupby(d.real.eq('N').map({True:'normales',False:'fallas'})).acierto.agg(['sum','count']).to_string())
