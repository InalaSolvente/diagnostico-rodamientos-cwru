"""Clasificador anidado sobre curvas coherentes; sin reglas que fuerzan el tipo."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import firma
from invariancia_portadora import modelos
from doble_banda_fijada import seleccionar_dos,verificar_dos
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_plantillas_coherentes'


def elegir_modelo(x,y,g):
    pred={k:np.full(len(y),'',dtype=object) for k in modelos()};trazas=[]
    for grupo in sorted(set(g)):
        tr=g!=grupo;te=~tr
        assert set(g[tr]).isdisjoint(g[te]) and set(y[tr])=={'B','IR','OR'}
        for nombre,fabrica in modelos().items():
            clf=fabrica().fit(x[tr],y[tr]);pred[nombre][te]=clf.predict(x[te])
        trazas.append(dict(grupo_fuera=grupo,grupos_train=sorted(set(g[tr]))))
    scores={k:float(np.mean([np.mean(p[y==c]==c) for c in ('B','IR','OR')])) for k,p in pred.items()}
    assert all(np.all(p!='') for p in pred.values())
    return max(scores,key=scores.get),scores,trazas


def main():
    d=pd.read_csv(SALIDA/'muestra.csv')
    with np.load(SALIDA/'rasgos.npz',allow_pickle=False) as z:x=z['x']
    prep=json.loads((SALIDA/'protocolo_preparacion.json').read_text(encoding='utf-8'))
    assert prep['codigo']==firma(AQUI/'plantillas_coherentes.py')
    assert x.shape==(133,724) and d.archivo.is_unique and d.sha256_senal.is_unique
    muestra=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    dev=d.archivo.isin(muestra.loc[muestra.real.ne('N'),'archivo']).to_numpy()
    assert int(dev.sum())==52 and set(d.loc[dev,'real'])=={'B','IR','OR'}
    y=d.real.to_numpy();g=d.grupo_auditoria.to_numpy()
    assert not any('@' in s for s in g)
    protocolo=dict(version='curvas724_anidado_sin_forzar_tipo_v1',
        exterior='tipo+tamaño, posiciones OR juntas, todas las cargas/sensores/muestreos fuera',
        interior='leave-one-group-out sobre entrenamiento exterior, exactitud equilibrada de tipo',
        candidatos='LDA shrinkage auto o SVM RBF C=1 gamma=scale balanced; mismos candidatos fijos',
        tipo='solo clasificador aprendido; no sobrescribir mediante D/PD de una familia alternativa',
        detector='mismo detector crudo con grupos completos de la referencia estricta',
        grado='mismo juez y dos bandas banco31 fijadas antes de abrir verificacion',
        limitacion='diseño exploratorio con dataset ya inspeccionado; selección de clasificador anidada, sin validación externa',
        codigo=firma(Path(__file__)),rasgos=firma(SALIDA/'rasgos.npz'),muestra=firma(SALIDA/'muestra.csv'),
        preparador=firma(AQUI/'plantillas_coherentes.py'),modelos=firma(AQUI/'invariancia_portadora.py'),
        bandas=firma(AQUI/'doble_banda_fijada.py'),criterio=firma(AQUI/'criterio_usuario.py'))
    rp=SALIDA/'protocolo_evaluacion.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    ajustes={};auditoria=[];pred=[];elegidos=[]
    with threadpool_limits(limits=1):
        for i,r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria=='N':
                t='N' if r.suficiente else 'sin_diagnostico';nombre='sin_modelo_por_detector'
            else:
                if r.grupo_auditoria not in ajustes:
                    tr=dev & (g!=r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr,'archivo'])
                    nombre,scores,internos=elegir_modelo(x[tr],y[tr],g[tr])
                    assert all(r.grupo_auditoria!=q['grupo_fuera'] and r.grupo_auditoria not in q['grupos_train'] for q in internos)
                    clf=modelos()[nombre]().fit(x[tr],y[tr]);ajustes[r.grupo_auditoria]=(nombre,clf)
                    auditoria.append(dict(grupo_exterior=r.grupo_auditoria,train=d.loc[tr,'archivo'].tolist(),
                        modelo=nombre,scores_internos=scores,internos=internos))
                    print('Curvas anidadas:',r.grupo_auditoria,nombre,scores,flush=True)
                nombre,clf=ajustes[r.grupo_auditoria];t=str(clf.predict(x[i:i+1])[0])
            pred.append(t);elegidos.append(nombre)
    salida=d.drop(columns=['prediccion','tipo_correcto','grado']).assign(prediccion=pred,modelo_anidado=elegidos)
    salida['tipo_correcto']=salida.real.eq(salida.prediccion)
    salida.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'auditoria_anidada.json').write_text(json.dumps(auditoria,indent=2),encoding='utf-8')
    # Tras congelar todos los tipos se consulta el resultado de la banda fijada.
    cache={}
    for carpeta,banco in [('resultados_grupos_por_defecto',False),('resultados_pd_doble',True),
                          ('resultados_globales_bancos',True),('resultados_doble_banda_fijada',False)]:
        for q in json.loads((AQUI/carpeta/'evidencia.json').read_text(encoding='utf-8')):
            cache[(q['archivo'],q['tipo'])]=q['banco31'] if banco else {k:q[k] for k in ('grado','ramas')}
    grados=[];evidencias=[]
    for r in salida.itertuples():
        if r.prediccion in ('N','sin_diagnostico'):v=dict(grado='ND',ramas=[])
        elif (r.archivo,r.prediccion) in cache:v=cache[(r.archivo,r.prediccion)]
        else:
            mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            senal=np.ravel(mat[F.clave_canal(mat,r.sensor,int(r.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
            mitad=len(senal)//2;c=json.loads(r.catalogo)
            s=seleccionar_dos(senal[:mitad],int(r.fs),c,r.prediccion)
            v=verificar_dos(senal[mitad:],int(r.fs),c,s)
            print('Bandas para tipo nuevo:',r.archivo,r.prediccion,flush=True)
        grados.append(v['grado']);evidencias.append(dict(archivo=r.archivo,tipo=r.prediccion,**v))
    salida['grado']=grados;salida.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    f=salida[salida.paper.notna()];delta=f.grado.map(ORDEN)-f.paper.map(ORDEN);ok=f.tipo_correcto
    resumen=dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(salida.loc[salida.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,seleccion_anidada=True,desplegado=False)
    assert len(salida)==133 and len(f)==105 and salida.sha256_senal.is_unique
    assert sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto'))==105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__=='__main__':main()
