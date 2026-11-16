"""Arquitectura fija físicamente restringida; grupos OR completos fuera."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ranker_fisico import RankerFisico,comprobar
from modelo_por_dominio import firma
from puntuar_tipos_fijos import evaluar

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_ranker_fisico'


def main():
    comprobar();SALIDA.mkdir(exist_ok=True)
    padre=AQUI/'resultados_plantillas_coherentes'
    d=pd.read_csv(padre/'muestra.csv')
    with np.load(padre/'rasgos.npz',allow_pickle=False) as z:x=z['x']
    assert len(d)==133 and x.shape==(133,724)
    muestra=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    dev=d.archivo.isin(muestra.loc[muestra.real.ne('N'),'archivo']).to_numpy()
    y=d.real.to_numpy();g=d.grupo_auditoria.to_numpy()
    assert dev.sum()==52 and not any('@' in s for s in g)
    protocolo=dict(version='ranker_monotono_fijo_v1',
        arquitectura='un puntaje por familia, compartido; B=logmediaexp BSF/2BSF',
        restricciones='soporte: coef>=0; desviacion absoluta y anchura: coef<=0; L2=1 fijo',
        validacion='tipo+tamaño, todas las posiciones OR/cargas/sensores/muestreos fuera',
        seleccion='una arquitectura fija; sin buscar penalizacion o decidir modelos por resultado exterior',
        detector='crudo con grupos completos, igual a la referencia',
        tipo='clasificador solamente; no forzar D/PD de otra familia',
        limitacion='diseño exploratorio sobre dataset ya inspeccionado; sin validación externa',
        codigo=firma(Path(__file__)),ranker=firma(AQUI/'ranker_fisico.py'),
        objetivo=firma(AQUI/'ranker_plantillas.py'),rasgos=firma(padre/'rasgos.npz'),
        muestra=firma(padre/'muestra.csv'),puntuador=firma(AQUI/'puntuar_tipos_fijos.py'))
    rp=SALIDA/'protocolo_evaluacion.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    ajustes={};folds=[];pred=[]
    with threadpool_limits(limits=1):
        for i,r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria=='N':t='N' if r.suficiente else 'sin_diagnostico'
            else:
                if r.grupo_auditoria not in ajustes:
                    tr=dev & (g!=r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr,'archivo'])
                    ajustes[r.grupo_auditoria]=RankerFisico().fit(x[tr],y[tr])
                    folds.append(dict(grupo_excluido=r.grupo_auditoria,train=d.loc[tr,'archivo'].tolist()))
                    print('Ranker físico, grupo fuera:',r.grupo_auditoria,flush=True)
                t=str(ajustes[r.grupo_auditoria].predict(x[i:i+1])[0])
            pred.append(t)
    d=d.drop(columns=['prediccion','grado','tipo_correcto']).assign(prediccion=pred)
    d['tipo_correcto']=d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    evaluar(d,SALIDA)


if __name__=='__main__':main()
