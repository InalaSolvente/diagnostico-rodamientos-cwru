"""Sensibilidad: agrupa todas las posiciones OR del mismo tamaño de defecto.

Reglas y banco31 fijados antes de esta auditoría. Detector y tipo se ajustan
excluyendo el grupo completo, conservando solo el desarrollo histórico.
"""
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

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_grupos_por_defecto'


def grupo_defecto(grupo):
    return str(grupo).split('@')[0]


def main():
    assert grupo_defecto('OR007@12')==grupo_defecto('OR007@6')=='OR007'
    assert grupo_defecto('IR007')=='IR007'
    SALIDA.mkdir(exist_ok=True)
    global_dir=AQUI/'resultados_globales_arbol'
    meta=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    meta['grupo_auditoria']=meta.grupo.map(grupo_defecto)
    with np.load(AQUI/'resultados_dominios_desarrollo/rasgos.npz',allow_pickle=False) as z:
        gate_x,origen=z['x'],z['origen']
    with np.load(global_dir/'rasgos.npz',allow_pickle=False) as z:todos=z['x']
    fuente=AQUI/'resultados_pd_doble'
    anterior=pd.read_csv(fuente/'comparacion_133.csv').set_index('archivo',drop=False)
    previa={q['archivo']:q for q in json.loads((fuente/'evidencia.json').read_text(encoding='utf-8'))}
    ep=AQUI/'resultados_correccion_d_persistente/evidencia.json'
    alternativas={q['archivo']:q['alternativas_seleccion'] for q in json.loads(ep.read_text(encoding='utf-8'))}
    protocolo=dict(version='grupo_tipo_tamano_sin_posicion_OR_v1',
        grupo='tipo y tamaño; todas las posiciones OR, cargas, sensores y muestreos juntos',
        modelo='mismos rasgos globales+arbol, LDA fijo, D sin alias de giro, PD maxmin y dos bandas banco31',
        detector='reajustado por sensor excluyendo el mismo grupo; ambos canales de cada normal juntos',
        seleccion='configuracion fijada antes de conocer los resultados de estos grupos',
        limitacion='auditoria interna; no prueba independencia de rodamientos ni elimina seleccion reiterada del dataset',
        codigo=firma(Path(__file__)),rasgos=firma(global_dir/'rasgos.npz'),
        muestra=firma(AQUI/'resultados_dominios_desarrollo/muestra.csv'),
        control=firma(AQUI/'control_ordenes_giro.py'),pd=firma(AQUI/'ensayar_pd_persistente.py'),
        bandas=firma(AQUI/'doble_banda_fijada.py'))
    rp=SALIDA/'protocolo.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    posiciones=anterior.reset_index(drop=True).reset_index().set_index('archivo')['index']
    fallas=meta.real.ne('N').to_numpy()
    im=posiciones.reindex(meta.loc[fallas,'archivo']).to_numpy()
    xd=todos[im]; yd=meta.loc[fallas,'real'].to_numpy()
    gd=meta.loc[fallas,'grupo_auditoria'].to_numpy()
    y=meta.real.to_numpy();grupos=meta.grupo_auditoria.to_numpy();sensor=meta.sensor.to_numpy()
    ajustes={};folds=[];pred=[];gates=[];tipos_base=[]
    with threadpool_limits(limits=1):
        for i,r in enumerate(anterior.itertuples()):
            g=grupo_defecto(r.grupo)
            if not r.suficiente:
                pred.append('sin_diagnostico');gates.append('sin_diagnostico');tipos_base.append('sin_diagnostico');continue
            if g not in ajustes:
                tr=grupos!=g;ft=gd!=g
                assert g not in set(grupos[tr]) and r.archivo not in set(meta.loc[tr,'archivo'])
                clf=crear().fit(xd[ft].reshape(-1,2021),np.repeat(yd[ft],4))
                detectores={}
                for dominio in ('DE','FE'):
                    filas=(tr & (sensor==dominio))[origen]
                    assert set(y[origen[filas]])=={'N','B','IR','OR'}
                    detectores[dominio]=crear().fit(gate_x[filas],y[origen[filas]])
                ajustes[g]=(clf,detectores)
                folds.append(dict(grupo_excluido=g,train=meta.loc[tr,'vista_uid'].tolist(),
                    grupos_train=sorted(set(grupos[tr]))))
                print('Grupo completo excluido:',g,flush=True)
            clf,detectores=ajustes[g]
            gate=predecir_tipo(detectores[r.sensor],todos[i,:,:1953])
            tb='N' if gate=='N' else predecir_tipo(clf,todos[i])
            if tb=='N':t='N'
            else:
                assert alternativas[r.archivo], 'Falta evidencia de selección; no inventar clasificación'
                td,_=elegir_d(tb,alternativas[r.archivo],F.MULTIPLICADORES[r.sensor])
                t=elegir_pd(td,alternativas[r.archivo])
            pred.append(t);gates.append(gate);tipos_base.append(tb)
    d=anterior.reset_index(drop=True).drop(columns=['prediccion','tipo_correcto','grado31','grado63']).assign(
        grupo_auditoria=anterior.grupo.map(grupo_defecto).to_numpy(),prediccion=pred,
        detector_auditoria=gates,tipo_base_auditoria=tipos_base)
    d['tipo_correcto']=d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    grados=[];evidencias=[]
    for r in d.itertuples():
        if r.prediccion in ('N','sin_diagnostico'):
            v=dict(grado='ND',ramas=[])
        elif r.prediccion==anterior.loc[r.archivo,'prediccion']:
            v=previa[r.archivo]['banco31']
        else:
            fila=idx.loc[r.archivo];mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            x=np.ravel(mat[F.clave_canal(mat,r.sensor,int(fila.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
            rpm=F.rpm_registro(mat,int(fila.registro),nominal=float(fila.rpm_nominal))
            c={k:float(a*rpm/60) for k,a in F.MULTIPLICADORES[r.sensor].items()}
            mitad=len(x)//2;s=seleccionar_dos(x[:mitad],int(r.fs),c,r.prediccion)
            v=verificar_dos(x[mitad:],int(r.fs),c,s)
            print('Bandas recalculadas por cambio:',r.archivo,r.prediccion,flush=True)
        grados.append(v['grado']);evidencias.append(dict(archivo=r.archivo,tipo=r.prediccion,**v))
    d['grado']=grados;d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    f=d[d.paper.notna()];delta=f.grado.map(ORDEN)-f.paper.map(ORDEN);ok=f.tipo_correcto
    resumen=dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,seleccion_posterior=False)
    assert len(d)==133 and len(f)==105 and d.sha256_senal.is_unique
    assert sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto'))==105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__=='__main__':main()
