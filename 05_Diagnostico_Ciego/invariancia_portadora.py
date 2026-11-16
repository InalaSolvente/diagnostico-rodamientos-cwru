"""Tipo sin identidad de la banda portadora; selección de modelo anidada."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import crear,firma
from ensayar_tipo_por_sensor import predecir_tipo
from control_ordenes_giro import elegir as elegir_d
from ensayar_pd_persistente import elegir as elegir_pd
from doble_banda_fijada import seleccionar_dos,verificar_dos
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_invariancia_portadora'


def compactar(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2021 or not np.isfinite(x).all():
        raise ValueError('Se requieren rasgos árbol+globales finitos')
    h=x[:,:1860].reshape(-1,31,60)
    # Los cuatro momentos globales se conservan, la PSD por portadora no.
    z=np.column_stack([x[:,1953:1957],h.max(axis=1),np.median(h,axis=1),h.mean(axis=1)])
    assert z.shape==(len(x),184) and np.isfinite(z).all()
    return z


def modelos():
    return dict(lda=crear,svm=lambda:make_pipeline(StandardScaler(),SVC(C=1.,gamma='scale',class_weight='balanced')))


def seleccionar_modelo(x,y,g):
    """Solo recibe los registros de entrenamiento del pliegue exterior."""
    pred={k:np.full(len(y),'',dtype=object) for k in modelos()}; trazas=[]
    for grupo in sorted(set(g)):
        tr=g!=grupo;te=np.flatnonzero(~tr)
        assert set(g[tr]).isdisjoint(set(g[te])) and set(y[tr])=={'B','IR','OR'}
        for nombre,fabrica in modelos().items():
            clf=fabrica().fit(x[tr].reshape(-1,184),np.repeat(y[tr],4))
            for i in te:pred[nombre][i]=predecir_tipo(clf,x[i])
        trazas.append(dict(grupo_interno_fuera=grupo,grupos_train=sorted(set(g[tr]))))
    scores={k:float(np.mean([np.mean(p[y==c]==c) for c in ('B','IR','OR')])) for k,p in pred.items()}
    assert all(np.all(p!='') for p in pred.values())
    ganador=max(scores,key=scores.get)
    return ganador,scores,trazas


def comprobar():
    rng=np.random.default_rng(814)
    x=rng.normal(size=(4,2021));z=compactar(x);otra=x.copy()
    h=otra[:,:1860].reshape(4,31,60);h[:]=h[:,rng.permutation(31)].copy()
    assert np.allclose(z,compactar(otra))
    otra[:,1957:]=rng.normal(size=otra[:,1957:].shape)
    assert np.allclose(z,compactar(otra))


def main():
    comprobar();SALIDA.mkdir(exist_ok=True)
    fuente=AQUI/'resultados_grupos_por_defecto'
    base=pd.read_csv(fuente/'comparacion_133.csv')
    with np.load(AQUI/'resultados_globales_arbol/rasgos.npz',allow_pickle=False) as z:original=z['x']
    x=np.full((133,4,184),np.nan)
    for i,r in enumerate(base.itertuples()):
        if r.suficiente:x[i]=compactar(original[i])
    meta=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    meta=meta[meta.real.ne('N')].copy()
    pos=base.reset_index().set_index('archivo')['index'].reindex(meta.archivo).to_numpy()
    xd=x[pos];y=meta.real.to_numpy();g=meta.grupo.str.split('@').str[0].to_numpy()
    protocolo=dict(version='invariante_184_anidado_v1',
        rasgos='4 momentos globales y maximo/mediana/media de 31 bandas para 60 evidencias armonicas; sin identidad portadora',
        exterior='tipo y tamaño; todas las posiciones OR, cargas, sensores y muestreos fuera juntos',
        interior='leave-one-group-out solo entrenamiento exterior; exactitud equilibrada de tipo, nunca grado del paper',
        candidatos='LDA shrinkage auto y SVM RBF C=1 gamma=scale balanced, ambos fijos',
        detector='mismo detector con grupos OR juntos',control='D sin alias de giro + PD maxmin + dos bandas banco31',
        limitacion='dataset ya inspeccionado; eleccion interna anidada no elimina la exploracion previa del conjunto',
        codigo=firma(Path(__file__)),rasgos_fuente=firma(AQUI/'resultados_globales_arbol/rasgos.npz'),
        detector_fuente=firma(fuente/'comparacion_133.csv'),muestra=firma(AQUI/'resultados_dominios_desarrollo/muestra.csv'))
    rp=SALIDA/'protocolo.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    ep=AQUI/'resultados_correccion_d_persistente/evidencia.json'
    alternativas={q['archivo']:q['alternativas_seleccion'] for q in json.loads(ep.read_text(encoding='utf-8'))}
    ajustes={};auditoria=[];pred=[];raw=[];elegidos=[]
    with threadpool_limits(limits=1):
        for i,r in enumerate(base.itertuples()):
            if not r.suficiente or r.detector_auditoria=='N':
                t='N' if r.suficiente else 'sin_diagnostico';tb=t;nombre='sin_modelo_por_detector'
            else:
                if r.grupo_auditoria not in ajustes:
                    tr=g!=r.grupo_auditoria
                    assert r.grupo_auditoria not in set(g[tr])
                    assert r.archivo not in set(meta.loc[tr,'archivo'])
                    nombre,scores,internos=seleccionar_modelo(xd[tr],y[tr],g[tr])
                    for q in internos:
                        assert r.grupo_auditoria!=q['grupo_interno_fuera']
                        assert r.grupo_auditoria not in q['grupos_train']
                    clf=modelos()[nombre]().fit(xd[tr].reshape(-1,184),np.repeat(y[tr],4))
                    ajustes[r.grupo_auditoria]=(nombre,clf)
                    auditoria.append(dict(grupo_exterior=r.grupo_auditoria,train=meta.loc[tr,'archivo'].tolist(),
                        modelo=nombre,scores_internos=scores,internos=internos))
                    print('Anidado:',r.grupo_auditoria,nombre,scores,flush=True)
                nombre,clf=ajustes[r.grupo_auditoria]
                tb=predecir_tipo(clf,x[i])
                td,_=elegir_d(tb,alternativas[r.archivo],F.MULTIPLICADORES[r.sensor])
                t=elegir_pd(td,alternativas[r.archivo])
            pred.append(t);raw.append(tb);elegidos.append(nombre)
    d=base.drop(columns=['prediccion','tipo_correcto','grado']).assign(prediccion=pred,tipo_invariante=raw,modelo_anidado=elegidos)
    d['tipo_correcto']=d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'auditoria_anidada.json').write_text(json.dumps(auditoria,indent=2),encoding='utf-8')
    cache={}
    for carpeta,banco in [('resultados_grupos_por_defecto',False),('resultados_pd_doble',True),
                          ('resultados_globales_bancos',True),('resultados_doble_banda_fijada',False)]:
        for q in json.loads((AQUI/carpeta/'evidencia.json').read_text(encoding='utf-8')):
            cache[(q['archivo'],q['tipo'])]=q['banco31'] if banco else {k:q[k] for k in ('grado','ramas')}
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    grados=[];evidencias=[]
    for r in d.itertuples():
        if r.prediccion in ('N','sin_diagnostico'):v=dict(grado='ND',ramas=[])
        elif (r.archivo,r.prediccion) in cache:v=cache[(r.archivo,r.prediccion)]
        else:
            fila=idx.loc[r.archivo];mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            senal=np.ravel(mat[F.clave_canal(mat,r.sensor,int(fila.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
            rpm=F.rpm_registro(mat,int(fila.registro),nominal=float(fila.rpm_nominal))
            c={k:float(a*rpm/60) for k,a in F.MULTIPLICADORES[r.sensor].items()}
            mitad=len(senal)//2;s=seleccionar_dos(senal[:mitad],int(r.fs),c,r.prediccion)
            v=verificar_dos(senal[mitad:],int(r.fs),c,s)
            print('Bandas nuevas:',r.archivo,r.prediccion,flush=True)
        grados.append(v['grado']);evidencias.append(dict(archivo=r.archivo,tipo=r.prediccion,**v))
    d['grado']=grados;d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    f=d[d.paper.notna()];ok=f.tipo_correcto;delta=f.grado.map(ORDEN)-f.paper.map(ORDEN)
    resumen=dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,seleccion_anidada=True)
    assert len(d)==133 and len(f)==105 and d.sha256_senal.is_unique
    assert sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto'))==105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__=='__main__':main()
