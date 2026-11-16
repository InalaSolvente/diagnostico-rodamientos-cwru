"""Sano/falla por sensor y tipo de defecto por modelo dedicado, sin oráculo."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from ventanas_verificadas import rasgos
from modelo_por_dominio import crear,calidad,dominio_catalogo,clave_cache,firma
from criterio_usuario import ORDEN
from bandas_laterales import rasgos_laterales

AQUI=Path(__file__).parent
PADRE=AQUI/'resultados_por_dominio'
FUENTE=AQUI/'resultados_dominios_desarrollo'
SALIDA=AQUI/'resultados_jerarquico'


def aplicar(z,ruta):
    with np.load(ruta,allow_pickle=False) as m:
        if str(m['extractor'])!=firma(AQUI/'modelo_espectral.py') or str(m['ventanas'])!=firma(AQUI/'ventanas_verificadas.py'):
            raise ValueError('Preprocesado incompatible con los pesos')
        if 'laterales' in m and str(m['laterales']) and str(m['laterales'])!=firma(AQUI/'bandas_laterales.py'):
            raise ValueError('Modulación cambiada; requiere reentrenamiento')
        s=((z-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
        return str(m['clases'][s.argmax()])


def predecir(x,fs,catalogo,*,modelo=None):
    modelo=Path(modelo) if modelo else SALIDA/'modelo_tipo.npz'
    base=rasgos(x,fs,catalogo);dominio=dominio_catalogo(catalogo)
    gate=aplicar(base.mean(axis=0),PADRE/f'modelo_{dominio}.npz')
    if gate=='N':tipo='N'
    else:
        with np.load(modelo,allow_pickle=False) as m:usa_laterales='laterales' in m and bool(str(m['laterales']))
        z=np.column_stack([base,rasgos_laterales(x,fs,catalogo)]).mean(axis=0) if usa_laterales else base.mean(axis=0)
        tipo=aplicar(z,modelo)
    return dict(dominio=dominio,decision_detector=gate,**calidad(x,fs,catalogo,tipo))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--laterales',action='store_true')
    args=parser.parse_args()
    if args.laterales:
        FUENTE=AQUI/'resultados_laterales_desarrollo'
        SALIDA=AQUI/'resultados_jerarquico_laterales'
    SALIDA.mkdir(exist_ok=True)
    meta=pd.read_csv(FUENTE/'muestra.csv')
    with np.load(FUENTE/'rasgos.npz',allow_pickle=False) as z:x,origen=z['x'],z['origen']
    with np.load(AQUI/'resultados_dominios_desarrollo/rasgos.npz',allow_pickle=False) as z:
        assert np.array_equal(x[:,:1953],z['x']) and np.array_equal(origen,z['origen'])
    gates=pd.read_csv(PADRE/'tipos_oof.csv').set_index('vista_uid').prediccion.reindex(meta.vista_uid).to_numpy()
    assert len(gates)==56 and not pd.isna(gates).any()
    protocolo=dict(version='jerarquico_laterales_v1' if args.laterales else 'jerarquico_fijo_v1',detector='OOF por sensor, ambos canales del mismo normal excluidos juntos',
        tipo='LDA shrinkage auto entrenado solo con B/IR/OR, condición completa excluida',
        entradas='solo señal, fs y catálogo completo; ninguna etiqueta ni frecuencia esperada en inferencia',
        codigo=firma(Path(__file__)),rasgos=firma(FUENTE/'rasgos.npz'),gate=firma(PADRE/'tipos_oof.csv'),
        bandas=firma(AQUI/'bandas_estables.py'),criterio=firma(AQUI/'criterio_usuario.py'),reserva_abierta_para_elegir=False)
    if args.laterales:protocolo['laterales']=firma(AQUI/'bandas_laterales.py')
    rp=SALIDA/'protocolo.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    y=meta.real.to_numpy();g=meta.grupo.to_numpy();pred=np.empty(len(meta),dtype=object);folds=[]
    with threadpool_limits(limits=1):
        for grupo in sorted(set(g)):
            test=np.flatnonzero(g==grupo);train=(g!=grupo)&(y!='N');ft=train[origen]
            assert grupo not in set(g[train]) and set(y[origen[ft]])=={'B','IR','OR'}
            clf=crear().fit(x[ft],y[origen[ft]])
            for i in test:
                scores=clf.decision_function(x[origen==i]).mean(axis=0)
                candidato=clf.classes_[scores.argmax()]
                pred[i]='N' if gates[i]=='N' else candidato
            folds.append(dict(grupo=grupo,train=meta.loc[train,'vista_uid'].tolist(),test=meta.loc[test,'vista_uid'].tolist()))
            print(f'Tipo dedicado OOF: {grupo}',flush=True)
    meta.assign(prediccion=pred,decision_detector=gates).to_csv(SALIDA/'tipos_oof.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    pp=json.loads((PADRE/'protocolo.json').read_text(encoding='utf-8'))
    assert pp['bandas']==protocolo['bandas'] and pp['criterio']==protocolo['criterio']
    m=meta.set_index('vista_uid',drop=False);cache={}
    for q in json.loads((PADRE/'predicciones.json').read_text(encoding='utf-8')):
        cache[(clave_cache(m.loc[q['vista_uid']]),q['tipo'])]={k:q[k] for k in ('tipo','grado','seleccion','verificacion')}
    filas=[]
    for i,r in enumerate(meta.itertuples()):
        q=cache.get((clave_cache(r),pred[i]))
        if q is None:
            mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            senal=np.ravel(mat[F.clave_canal(mat,r.sensor,int(r.registro))])
            q=calidad(senal,int(r.fs_original),json.loads(r.catalogo),str(pred[i]))
        filas.append(dict(archivo=r.archivo,vista_uid=r.vista_uid,sensor=r.sensor,**q))
        (SALIDA/'predicciones.json').write_text(json.dumps(filas,ensure_ascii=False,indent=2),encoding='utf-8')
    d=meta[['archivo','vista_uid','sensor','real']].assign(prediccion=pred,grado=[q['grado'] for q in filas])
    referencia=pd.read_csv(AQUI/'resultados_criterio_usuario_total/comparacion.csv').set_index('archivo').paper
    d['paper']=d.archivo.map(referencia);d['tipo_correcto']=d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'comparacion.csv',index=False)
    f=d[d.paper.notna()];delta=f.grado.map(ORDEN)-f.paper.map(ORDEN);ok=f.tipo_correcto;n=d[d.real.eq('N')]
    resumen=dict(total=52,tipo_correcto=int(ok.sum()),mejoran=int((ok&delta.gt(0)).sum()),iguales=int((ok&delta.eq(0)).sum()),
        empeoran=int((ok&delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),sin_retroceso=int((ok&delta.ge(0)).sum()),
        normales_vistas_correctas=int(n.tipo_correcto.sum()),normales_vistas=len(n),normales_archivos=2,
        normales_archivos_ambos_canales_correctos=int(n.groupby('archivo').tipo_correcto.all().sum()))
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    with threadpool_limits(limits=1):
        ft=(y!='N')[origen];clf=crear().fit(x[ft],y[origen[ft]])
    sc,lda=clf.steps[0][1],clf.steps[1][1]
    np.savez_compressed(SALIDA/'modelo_tipo.npz',media=sc.mean_,escala=sc.scale_,coef=lda.coef_,intercepto=lda.intercept_,
        clases=lda.classes_,extractor=firma(AQUI/'modelo_espectral.py'),ventanas=firma(AQUI/'ventanas_verificadas.py'),
        laterales=firma(AQUI/'bandas_laterales.py') if args.laterales else '')
    print(json.dumps(resumen),flush=True)
