"""Control de hipótesis: clasificación con las 45 interacciones físicas solamente.

Mismos datos y particiones; parámetros fijos. No usa firma espacial de los 31
nodos como entrada separada ni la referencia ND/PD/D para ajustar el tipo.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_tipo_fisico'


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_ventanas_fisicas'
    meta = pd.read_csv(fuente/'muestra.csv')
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:
        x,origen = z['x'][:,-45:],z['origen']
    assert x.shape == (216,45) and np.array_equal(np.bincount(origen),np.full(54,4))
    protocolo = dict(version='tipo_fisico45_lr_v1', clasificador='StandardScaler + LogisticRegression(C=1, balanced)',
        datos='45 interacciones físicas; cuatro ventanas/reg.; catálogo completo sin defecto real esperado',
        parametros='fijos, sin búsqueda de C ni referencia del paper',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rasgos_sha256=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    y=meta.real.to_numpy(); grupos=meta.grupo.to_numpy(); pred=np.empty(len(meta),dtype=object); folds=[]
    with threadpool_limits(limits=1):
        for grupo in sorted(set(grupos)):
            test=grupos==grupo; train=~test; ft=train[origen]
            assert grupo not in set(grupos[train]) and set(y[origen[ft]])==set(y)
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000))
            clf.fit(x[ft],y[origen[ft]])
            for i in np.flatnonzero(test):
                scores=clf.decision_function(x[origen==i]).mean(axis=0)
                pred[i]=clf.classes_[scores.argmax()]
            folds.append(dict(grupo=grupo,train=meta.loc[train,'archivo'].tolist(),test=meta.loc[test,'archivo'].tolist()))
    d=meta.assign(prediccion=pred,acierto=pred==y)
    d.to_csv(SALIDA/'tipos_oof.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    print(d.groupby('real').acierto.agg(['sum','count']).to_string())
