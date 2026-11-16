"""Selección anidada de dos clasificadores no lineales, solo desarrollo.

La selección interna usa tipos, no los grados del paper. Todas las ventanas de
una condición excluida permanecen fuera del ajuste y de la selección exterior.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_no_lineal'


def modelos():
    return {
        'svm': lambda: make_pipeline(StandardScaler(), SVC(C=1,gamma='scale',class_weight='balanced')),
        'arboles': lambda: ExtraTreesClassifier(n_estimators=200,min_samples_leaf=2,
            max_features='sqrt',class_weight='balanced',random_state=2026,n_jobs=1),
    }


def predecir_registros(clf,x,origen,indices):
    salida=[]
    for i in indices:
        z=x[origen==i]
        scores=clf.decision_function(z) if hasattr(clf,'decision_function') else clf.predict_proba(z)
        salida.append(clf.classes_[scores.mean(axis=0).argmax()])
    return np.asarray(salida)


def elegir(x,y,grupos,origen,permitidos):
    fallas=permitidos[y[permitidos]!='N']; valores={k:[] for k in modelos()}; trazas=[]
    for _,indices in GroupKFold(n_splits=4).split(fallas,y[fallas],grupos[fallas]):
        test=fallas[indices];train=np.setdiff1d(permitidos,test)
        assert set(grupos[train]).isdisjoint(set(grupos[test]))
        filas=np.isin(origen,train)
        assert set(y[origen[filas]])==set(y)
        for nombre,crear in modelos().items():
            clf=crear().fit(x[filas],y[origen[filas]])
            pred=predecir_registros(clf,x,origen,test)
            valores[nombre].append(float(np.mean([np.mean(pred[y[test]==c]==c) for c in np.unique(y[test])])))
        trazas.append(dict(train=train.tolist(),test=test.tolist()))
    medias={k:float(np.mean(v)) for k,v in valores.items()}
    return max(medias,key=medias.get),medias,trazas


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente=AQUI/'resultados_ventanas_fisicas'
    meta=pd.read_csv(fuente/'muestra.csv')
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:x,origen=z['x'],z['origen']
    assert x.shape==(216,1998) and np.array_equal(np.bincount(origen),np.full(54,4))
    protocolo=dict(version='no_lineal_anidado_v1',candidatos=list(modelos()),
        seleccion='4 pliegues internos por condición, exactitud equilibrada de tipo en fallas',
        exterior='condición completa fuera; cada normal único fuera individualmente',
        limitacion='normales se conservan en ajuste interno: no hay suficientes adquisiciones para seleccionar por especificidad independiente',
        referencia_paper_usada_para_elegir=False,
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rasgos_sha256=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    rp=SALIDA/'protocolo.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    y=meta.real.to_numpy();grupos=meta.grupo.to_numpy();pred=np.empty(len(y),dtype=object);auditoria=[]
    with threadpool_limits(limits=1):
        for grupo in sorted(set(grupos)):
            test=np.flatnonzero(grupos==grupo);train=np.flatnonzero(grupos!=grupo)
            nombre,scores,trazas=elegir(x,y,grupos,origen,train)
            ft=np.isin(origen,train);clf=modelos()[nombre]().fit(x[ft],y[origen[ft]])
            pred[test]=predecir_registros(clf,x,origen,test)
            auditoria.append(dict(grupo=grupo,train=meta.loc[train,'archivo'].tolist(),test=meta.loc[test,'archivo'].tolist(),
                indices_train=train.tolist(),indices_test=test.tolist(),modelo=nombre,scores_internos=scores,internos=trazas))
            print(f'Anidado no lineal {grupo[:20]}: {nombre}',flush=True)
    d=meta.assign(prediccion=pred,acierto=pred==y)
    d.to_csv(SALIDA/'tipos_oof.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(auditoria,indent=2),encoding='utf-8')
    print(d.groupby('real').acierto.agg(['sum','count']).to_string())
