"""LDA propio de 48 kHz: mitad fija y representación alineada L6/375 Hz."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from modelo_ciego import F
from alinear_bandas import rasgos_alineados
from ranking_catalogo import grupo_condicion
from integridad_senales import huellas_senales, exigir_independencia

AQUI=Path(__file__).parent


def crear_particion(idx,particion_previa):
    """Dos cargas por condición de falla; normales conservan la separación previa."""
    rng=np.random.default_rng(2026)
    filas=[]
    for grupo,g in idx.assign(grupo=idx.archivo.map(grupo_condicion)).groupby('grupo',sort=True):
        archivos=sorted(g.archivo)
        elegidos=set(rng.permutation(archivos)[:len(archivos)//2]) if grupo!='Normal' else set()
        for archivo in archivos:
            parte=particion_previa[archivo] if grupo=='Normal' else ('desarrollo' if archivo in elegidos else 'reserva')
            filas.append(dict(archivo=archivo,grupo=grupo,particion=parte))
    return pd.DataFrame(filas).sort_values('archivo').reset_index(drop=True)


def crear_modelo():
    return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))


def ejecutar():
    salida=AQUI/'resultados_modelo_48k'
    salida.mkdir(exist_ok=True);(salida/'cache').mkdir(exist_ok=True)
    datos=AQUI.parent/'data2/CWRU'
    idx=F.leer_indice(str(datos));idx=idx[idx.fs==48000]
    previa=pd.read_csv(AQUI/'particion_desarrollo.csv').set_index('archivo').particion.to_dict()
    particion=crear_particion(idx,previa)
    ruta_particion=salida/'particion.csv'
    if ruta_particion.exists():
        assert pd.read_csv(ruta_particion).equals(particion),'La partición existente no coincide'
    else:particion.to_csv(ruta_particion,index=False)
    dev=set(particion.loc[particion.particion=='desarrollo','archivo'])
    reserva=set(particion.loc[particion.particion=='reserva','archivo'])
    assert len(dev)==len(reserva)==14 and not dev&reserva
    exigir_independencia(huellas_senales(idx, datos), dev, reserva)
    idx=idx[idx.archivo.isin(dev)]
    firma_alineado=hashlib.sha256((AQUI/'alinear_bandas.py').read_bytes()).hexdigest()
    firma_espectral=hashlib.sha256((AQUI/'modelo_espectral.py').read_bytes()).hexdigest()
    filas=[];rasgos=[]
    for _,r in idx.iterrows():
        ruta=datos/r.archivo;mat=loadmat(ruta)
        clave=next(k for k in mat if k.endswith(f'_{r.sensor_falla}_time'))
        claves_rpm=[k for k in mat if k.endswith('RPM')]
        rpm=float(np.ravel(mat[claves_rpm[0]])[0]) if claves_rpm else float(r.rpm_nominal)
        catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
        firma_datos=hashlib.sha256(ruta.read_bytes()).hexdigest()
        clave_cache=hashlib.sha256((firma_datos+firma_alineado+firma_espectral+json.dumps(catalogo,sort_keys=True)).encode()).hexdigest()
        cache=salida/'cache'/f'{clave_cache}.npz'
        if cache.exists():
            with np.load(cache,allow_pickle=False) as c:feature=c['x']
        else:
            feature,nodos,L=rasgos_alineados(np.ravel(mat[clave]),48000,catalogo)
            assert L==6 and len(nodos)==31
            np.savez_compressed(cache,x=feature)
        rasgos.append(feature)
        filas.append(dict(archivo=r.archivo,grupo=grupo_condicion(r.archivo),real=r.zona,
                          sha256_datos=firma_datos,catalogo=catalogo))
        print(f'Rasgos {len(filas)}/14 {r.archivo}',flush=True)
    x=np.asarray(rasgos);meta=pd.DataFrame(filas)
    y=meta.real.to_numpy();grupos=meta.grupo.to_numpy()
    assert x.shape==(14,1953) and set(y)=={'B','IR','N'} and np.isfinite(x).all()
    pred=np.full(len(y),'no_evaluado',dtype=object);auditoria=[]
    for grupo in sorted(set(grupos[y!='N'])):
        test=grupos==grupo;train=~test
        assert set(y[train])=={'B','IR','N'} and grupo not in set(grupos[train])
        clf=crear_modelo();clf.fit(x[train],y[train]);pred[test]=clf.predict(x[test])
        auditoria.append(dict(grupo_excluido=grupo,entrenamiento=meta.loc[train,'archivo'].tolist(),
                              prueba=meta.loc[test,'archivo'].tolist(),clases_entrenadas=sorted(set(y[train]))))
        print(f'Grupo validado: {grupo}',flush=True)
    fallas=y!='N';aciertos=int((pred[fallas]==y[fallas]).sum())
    evaluacion=meta[['archivo','grupo','real']].copy();evaluacion['prediccion_oof']=pred
    evaluacion['evaluado']=fallas
    evaluacion['acierto']=[bool(a==b) if ok else None for a,b,ok in zip(y,pred,fallas)]
    evaluacion.to_csv(salida/'validacion.csv',index=False)
    pd.crosstab(evaluacion.loc[fallas,'real'],evaluacion.loc[fallas,'prediccion_oof']).to_csv(salida/'matriz_confusion.csv')
    final=crear_modelo();final.fit(x,y)
    escala,lda=final.steps[0][1],final.steps[-1][1]
    scores=((x-escala.mean_)/escala.scale_)@lda.coef_.T+lda.intercept_
    assert np.array_equal(lda.classes_[scores.argmax(axis=1)],final.predict(x))
    np.savez_compressed(salida/'modelo_48k.npz',media=escala.mean_,escala=escala.scale_,coef=lda.coef_,
        intercepto=lda.intercept_,clases=lda.classes_,fs=48000,L=6,ancho_final=375,
        extractor_alineado_sha256=firma_alineado,extractor_espectral_sha256=firma_espectral)
    np.savez_compressed(salida/'rasgos_desarrollo.npz',x=x)
    (salida/'entrenamiento.json').write_text(json.dumps(filas,indent=2),encoding='utf-8')
    (salida/'auditoria_grupos.json').write_text(json.dumps(auditoria,indent=2),encoding='utf-8')
    config=dict(semilla=2026,fs=48000,L=6,ancho_final=375,nodos=31,rango_hz=[0,6000],
        modelo='StandardScaler + LDA(lsqr, shrinkage=auto)',parametros_elegidos_con_resultado=False,
        desarrollo=sorted(dev),reserva_no_ejecutada=sorted(reserva),clases=lda.classes_.tolist(),
        validacion='excluir condición de falla completa; normales solo entrenamiento, sin métrica OOF',
        aciertos_oof=aciertos,total_oof=int(fallas.sum()))
    (salida/'configuracion.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    lineas=['# Modelo propio de 48 kHz','',
        'Mismo clasificador LDA regularizado y representación alineada: L=6, 31 nodos 0–6 kHz y bandas finales de 375 Hz.',
        'Entrenamiento: 14 señales (6 bola, 6 interna, 2 normales); las otras 14 no se cargaron para este entrenamiento/ensayo.',
        f'Primera opción correcta fuera de condición: {aciertos}/12 ({100*aciertos/12:.2f}%).',
        'Seis grupos de falla excluidos por turno. Escalado y ajuste solo en entrenamiento; sin selección de variantes contra los resultados.',
        'Los dos normales permanecen en entrenamiento: no existe otro grupo normal para validarlos fuera de grupo. No se informa exactitud en normales.',
        'No hay ejemplos de pista externa o jaula en este subconjunto: esas clases no se entrenan ni se declaran cubiertas.',
        'La reserva comparte condiciones con desarrollo y fue examinada en experimentos anteriores; no es un conjunto externo intacto.',
        'El modelo final se ajusta con las 14 señales. Los porcentajes anteriores no son aciertos de entrenamiento.',
        'Inferencia: señal de 48 kHz y catálogo completo; sin frecuencia ni defecto real esperado.',
        'El árbol conserva solo 0–6 kHz para mantener la representación alineada; no utiliza 6–24 kHz.']
    (salida/'resumen.md').write_text('\n'.join(lineas)+'\n',encoding='utf-8')
    print('\n'.join(lineas))


if __name__=='__main__':ejecutar()
