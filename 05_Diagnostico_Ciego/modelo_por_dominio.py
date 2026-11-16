"""Modelos DE/FE seleccionados por catálogo del componente, no por falla real."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from ventanas_verificadas import rasgos
from bandas_estables import seleccionar, verificar
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
FUENTE=AQUI/'resultados_dominios_desarrollo'
SALIDA=AQUI/'resultados_por_dominio'


def firma(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def dominio_catalogo(c):
    compatibles=[]
    for nombre,m in F.MULTIPLICADORES.items():
        fr=np.array([c[k]/m[k] for k in m])
        if np.allclose(fr,np.median(fr),rtol=1e-6,atol=1e-9):compatibles.append(nombre)
    if len(compatibles)!=1:raise ValueError('Geometría del catálogo no configurada')
    return compatibles[0]


def crear():return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))


def calidad(x,fs,c,tipo):
    mitad=len(x)//2
    if mitad<2.5*fs:raise ValueError('Se requieren al menos 5 s')
    s=seleccionar(x[:mitad],fs,c,tipo);v=verificar(x[mitad:],fs,c,s)
    return dict(tipo=tipo,grado=v['grado'],seleccion=s,verificacion=v)


def predecir(x,fs,catalogo):
    dominio=dominio_catalogo(catalogo)
    with np.load(SALIDA/f'modelo_{dominio}.npz',allow_pickle=False) as m:
        for archivo,campo in [('modelo_espectral.py','extractor'),('ventanas_verificadas.py','ventanas')]:
            if str(m[campo])!=firma(AQUI/archivo):raise ValueError('Preprocesado cambiado; reentrenar')
        z=rasgos(x,fs,catalogo).mean(axis=0)
        scores=((z-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
        tipo=str(m['clases'][scores.argmax()])
    return dict(dominio=dominio,**calidad(x,fs,catalogo,tipo))


def clave_cache(r):
    return (r.archivo,r.clave_canal,r.sha256_senal,int(r.fs_original),json.dumps(json.loads(r.catalogo),sort_keys=True))


if __name__=='__main__':
    SALIDA.mkdir(exist_ok=True)
    meta=pd.read_csv(FUENTE/'muestra.csv')
    with np.load(FUENTE/'rasgos.npz',allow_pickle=False) as z:x,origen=z['x'],z['origen']
    assert x.shape==(224,1953) and meta.vista_uid.is_unique and meta.archivo.nunique()==54
    for r in meta.itertuples():assert dominio_catalogo(json.loads(r.catalogo))==r.sensor
    protocolo=dict(version='dominios_lda_fijo_v1',clasificador='LDA shrinkage auto independiente DE/FE',
        seleccion_modelo='solo catálogo completo del rodamiento; misma selección para todas sus fallas',
        negativos='dos archivos sanos, dos canales por archivo; ambos canales excluidos simultáneamente',
        metricas='52 fallas; cuatro vistas normales y dos archivos normales informados por separado',
        codigo=firma(Path(__file__)),rasgos=firma(FUENTE/'rasgos.npz'),
        bandas=firma(AQUI/'bandas_estables.py'),criterio=firma(AQUI/'criterio_usuario.py'),
        reserva_abierta_para_elegir=False)
    rp=SALIDA/'protocolo.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    y=meta.real.to_numpy();g=meta.grupo.to_numpy();dom=meta.sensor.to_numpy();pred=np.empty(len(meta),dtype=object);folds=[]
    with threadpool_limits(limits=1):
        for grupo in sorted(set(g)):
            test=g==grupo;train=~test
            for sensor in ('DE','FE'):
                t=np.flatnonzero(test&(dom==sensor));tr=train&(dom==sensor)
                if not len(t):continue
                ft=tr[origen]
                assert grupo not in set(g[tr]) and set(y[origen[ft]])=={'B','IR','N','OR'}
                clf=crear().fit(x[ft],y[origen[ft]])
                for i in t:
                    scores=clf.decision_function(x[origen==i]).mean(axis=0)
                    pred[i]=clf.classes_[scores.argmax()]
                folds.append(dict(grupo=grupo,sensor=sensor,train=meta.loc[tr,'vista_uid'].tolist(),test=meta.loc[t,'vista_uid'].tolist()))
            print(f'Dominio OOF: {grupo}',flush=True)
    meta.assign(prediccion=pred).to_csv(SALIDA/'tipos_oof.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    cache={}
    padre=AQUI/'resultados_verificados_ventanas'
    cp=json.loads((padre/'protocolo.json').read_text(encoding='utf-8'))
    assert cp['bandas_sha256']==protocolo['bandas'] and cp['criterio_sha256']==protocolo['criterio']
    mp=pd.read_csv(AQUI/'resultados_ventanas_verificadas/muestra.csv').set_index('archivo',drop=False)
    for q in json.loads((padre/'predicciones.json').read_text(encoding='utf-8')):
        r=mp.loc[q['archivo']];cache[clave_cache(r)]=q
    filas=[]
    for i,r in enumerate(meta.itertuples()):
        c=json.loads(r.catalogo);guardado=cache.get(clave_cache(r))
        if guardado and guardado['tipo']==pred[i]:
            q={k:v for k,v in guardado.items() if k!='archivo'}
        else:
            m=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            senal=np.ravel(m[F.clave_canal(m,r.sensor,int(r.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
            q=calidad(senal,int(r.fs_original),c,str(pred[i]))
        filas.append(dict(archivo=r.archivo,vista_uid=r.vista_uid,sensor=r.sensor,**q))
        (SALIDA/'predicciones.json').write_text(json.dumps(filas,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'Dominio verificado {i+1}/56: {pred[i]} {q["grado"]}',flush=True)
    d=meta[['archivo','vista_uid','sensor','real']].assign(prediccion=pred,grado=[q['grado'] for q in filas])
    referencia=pd.read_csv(AQUI/'resultados_criterio_usuario_total/comparacion.csv').set_index('archivo').paper
    d['paper']=d.archivo.map(referencia);d['tipo_correcto']=d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'comparacion.csv',index=False)
    f=d[d.paper.notna()];delta=f.grado.map(ORDEN)-f.paper.map(ORDEN);ok=f.tipo_correcto
    n=d[d.real.eq('N')]
    resumen=dict(total=52,tipo_correcto=int(ok.sum()),mejoran=int((ok&delta.gt(0)).sum()),iguales=int((ok&delta.eq(0)).sum()),
        empeoran=int((ok&delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),sin_retroceso=int((ok&delta.ge(0)).sum()),
        normales_vistas=len(n),normales_vistas_correctas=int(n.tipo_correcto.sum()),
        normales_archivos=n.archivo.nunique(),normales_archivos_ambos_canales_correctos=int(n.groupby('archivo').tipo_correcto.all().sum()))
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    with threadpool_limits(limits=1):
        for sensor in ('DE','FE'):
            ft=(dom==sensor)[origen];clf=crear().fit(x[ft],y[origen[ft]])
            sc,lda=clf.steps[0][1],clf.steps[1][1]
            np.savez_compressed(SALIDA/f'modelo_{sensor}.npz',media=sc.mean_,escala=sc.scale_,coef=lda.coef_,
                intercepto=lda.intercept_,clases=lda.classes_,extractor=firma(AQUI/'modelo_espectral.py'),ventanas=firma(AQUI/'ventanas_verificadas.py'))
    print(json.dumps(resumen),flush=True)
