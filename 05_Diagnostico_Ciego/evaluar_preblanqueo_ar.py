"""Evalúa preblanqueo AR fijo con grupos completos y sin frecuencia esperada."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import crear,firma
from ensayar_tipo_por_sensor import predecir_tipo
from control_ordenes_giro import elegir as elegir_d
from ensayar_pd_persistente import elegir as elegir_pd
from doble_banda_fijada import seleccionar_dos,verificar_dos
from criterio_usuario import ORDEN
from preblanqueo_ar import preblanquear,FS

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_preblanqueo_ar'


def main():
    meta=pd.read_csv(SALIDA/'vistas.csv')
    with np.load(SALIDA/'rasgos.npz',allow_pickle=False) as z:x=z['x']
    assert x.shape==(135,4,2021) and meta.vista_uid.is_unique and meta.primaria.sum()==133
    preparacion=json.loads((SALIDA/'protocolo_preparacion.json').read_text(encoding='utf-8'))
    assert preparacion['codigo']==firma(AQUI/'preblanqueo_ar.py')
    original=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    es_dev=meta.vista_uid.isin(original.vista_uid).to_numpy()
    assert int(es_dev.sum())==56 and set(meta.loc[es_dev,'vista_uid'])==set(original.vista_uid)
    y=meta.real.to_numpy();g=meta.grupo.to_numpy();dom=meta.sensor.to_numpy()
    protocolo=dict(version='AR32_LDA_grupos_completos_v1',
        hipotesis='reducir respuesta de componentes predecibles antes de aprender tipo y buscar bandas',
        filtro='AR32 fijo, ajustado sin etiquetas con primeros 2.5 s; remuestreo de mitades separado',
        tipo='LDA shrinkage auto, arbol+globales; detector LDA por sensor con arbol',
        grupos='tipo y tamaño; todas las posiciones OR, cargas, sensores y muestreos excluidos juntos',
        entrenamiento='solo 52 fallas y dos archivos normales de desarrollo; ambas vistas normales juntas',
        decision='mismas reglas D sin alias de giro y PD maxmin; maximo dos bandas disjuntas banco31',
        limitacion='ensayo interno sobre datos ya inspeccionados; sin validacion externa',
        codigo=firma(Path(__file__)),preblanqueo=firma(AQUI/'preblanqueo_ar.py'),
        rasgos=firma(SALIDA/'rasgos.npz'),vistas=firma(SALIDA/'vistas.csv'),
        control=firma(AQUI/'control_ordenes_giro.py'),pd=firma(AQUI/'ensayar_pd_persistente.py'),
        bandas=firma(AQUI/'doble_banda_fijada.py'),criterio=firma(AQUI/'criterio_usuario.py'))
    rp=SALIDA/'protocolo_evaluacion.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    ajustes={};folds=[];filas=[];evidencias=[]
    referencia=pd.read_csv(AQUI/'resultados_grupos_por_defecto/comparacion_133.csv').set_index('archivo')
    with threadpool_limits(limits=1):
        for i,r in enumerate(meta.itertuples()):
            if not r.primaria:continue
            alternativas={};v=dict(grado='ND',ramas=[]);tb='sin_diagnostico';t=tb;gate=tb
            if r.suficiente:
                if r.grupo not in ajustes:
                    tr=es_dev & (g!=r.grupo)
                    ft=tr & (y!='N')
                    assert r.grupo not in set(g[tr]) and r.archivo not in set(meta.loc[tr,'archivo'])
                    assert set(y[ft])=={'B','IR','OR'}
                    clf=crear().fit(x[ft].reshape(-1,2021),np.repeat(y[ft],4))
                    detectores={}
                    for sensor in ('DE','FE'):
                        td=tr & (dom==sensor)
                        assert set(y[td])=={'N','B','IR','OR'}
                        detectores[sensor]=crear().fit(x[td,:,:1953].reshape(-1,1953),np.repeat(y[td],4))
                    ajustes[r.grupo]=(clf,detectores)
                    folds.append(dict(grupo_excluido=r.grupo,train=meta.loc[tr,'vista_uid'].tolist(),
                        grupos_train=sorted(set(g[tr]))))
                    print('Preblanqueo, grupo fuera:',r.grupo,flush=True)
                clf,detectores=ajustes[r.grupo]
                gate=predecir_tipo(detectores[r.sensor],x[i,:,:1953])
                if gate=='N':tb=t='N'
                else:
                    tb=predecir_tipo(clf,x[i])
                    mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
                    cruda=np.ravel(mat[F.clave_canal(mat,r.sensor,int(r.registro))])
                    assert hashlib.sha256(cruda.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
                    blanca,info=preblanquear(cruda,int(r.fs_original))
                    assert hashlib.sha256(blanca.astype('<f8').tobytes()).hexdigest()==r.sha256_procesada
                    assert info['coef']==json.loads(r.coef)
                    c=json.loads(r.catalogo);mitad=len(blanca)//2
                    pares={tipo:seleccionar_dos(blanca[:mitad],FS,c,tipo) for tipo in ('IR','OR','B')}
                    alternativas={tipo:s[0] for tipo,s in pares.items()}
                    td,_=elegir_d(tb,alternativas,c);t=elegir_pd(td,alternativas)
                    # Primera consulta al segmento reservado tras fijar tipo y ambas bandas.
                    v=verificar_dos(blanca[mitad:],FS,c,pares[t])
            filas.append(dict(archivo=r.archivo,vista_uid=r.vista_uid,sensor=r.sensor,fs=int(r.fs_original),
                fs_procesado=FS,real=r.real,grupo=r.grupo,sha256_senal=r.sha256_senal,
                sha256_procesada=r.sha256_procesada,suficiente=r.suficiente,detector=gate,
                tipo_base=tb,prediccion=t,grado=v['grado'],tipo_correcto=t==r.real,
                paper=referencia.loc[r.archivo,'paper']))
            evidencias.append(dict(archivo=r.archivo,tipo=t,alternativas_seleccion=alternativas,**v))
            if len(filas)%10==0:
                pd.DataFrame(filas).to_csv(SALIDA/'comparacion_parcial.csv',index=False)
                (SALIDA/'evidencia_parcial.json').write_text(json.dumps(evidencias,ensure_ascii=False),encoding='utf-8')
            print(f'{len(filas)}/133 preblanqueo: {r.archivo} {t}/{v["grado"]}',flush=True)
    d=pd.DataFrame(filas)
    assert len(d)==133 and d.archivo.is_unique and d.sha256_senal.is_unique
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    f=d[d.paper.notna()];delta=f.grado.map(ORDEN)-f.paper.map(ORDEN);ok=f.tipo_correcto
    resumen=dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,desplegado=False)
    assert len(f)==105 and sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto'))==105
    assert firma(AQUI/'preblanqueo_ar.py')==protocolo['preblanqueo']
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__=='__main__':main()
