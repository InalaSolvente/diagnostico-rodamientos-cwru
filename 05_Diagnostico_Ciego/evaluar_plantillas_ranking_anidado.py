"""Añade un ranker compartido al menú anidado sin usar resultados exteriores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modelo_por_dominio import firma
from ranker_plantillas import RankerPlantillas,comprobar
from puntuar_tipos_fijos import evaluar

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_plantillas_ranking_anidado'


def validar_interno(x,y,g):
    pred=np.full(len(y),'',dtype=object);trazas=[]
    for grupo in sorted(set(g)):
        tr=g!=grupo;te=~tr
        assert set(g[tr]).isdisjoint(g[te]) and set(y[tr])=={'B','IR','OR'}
        m=RankerPlantillas().fit(x[tr],y[tr]);pred[te]=m.predict(x[te])
        trazas.append(dict(grupo_fuera=grupo,grupos_train=sorted(set(g[tr]))))
    assert np.all(pred!='')
    score=float(np.mean([np.mean(pred[y==c]==c) for c in ('B','IR','OR')]))
    return score,trazas


def main():
    comprobar();SALIDA.mkdir(exist_ok=True)
    padre=AQUI/'resultados_plantillas_coherentes'
    d=pd.read_csv(padre/'tipos_133.csv')
    with np.load(padre/'rasgos.npz',allow_pickle=False) as z:x=z['x']
    assert len(d)==133 and x.shape==(133,724)
    antiguo=json.loads((padre/'protocolo_evaluacion.json').read_text(encoding='utf-8'))
    assert antiguo['codigo']==firma(AQUI/'evaluar_plantillas_coherentes.py')
    assert antiguo['preparador']==firma(AQUI/'plantillas_coherentes.py')
    assert antiguo['rasgos']==firma(padre/'rasgos.npz')
    assert antiguo['modelos']==firma(AQUI/'invariancia_portadora.py')
    previos=json.loads((padre/'auditoria_anidada.json').read_text(encoding='utf-8'))
    muestra=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    dev=d.archivo.isin(muestra.loc[muestra.real.ne('N'),'archivo']).to_numpy()
    y=d.real.to_numpy();g=d.grupo_auditoria.to_numpy()
    assert dev.sum()==52 and not any('@' in s for s in g)
    protocolo=dict(version='curvas724_menu_tres_anidado_v1',
        candidatos='LDA, SVM RBF fijos y ranker log-media-exp compartido L2=1 fijo',
        eleccion='exactitud equilibrada interna, dejando el grupo exterior fuera de todo ajuste y eleccion',
        reutilizacion='scores internos de LDA/SVM y tipos de su ganador bajo exactamente las mismas particiones',
        tipo='ninguna correccion manual ni frecuencia esperada; candidato B agrupa BSF/2BSF',
        parametros='sin busqueda de regularizacion o semillas',
        limitacion='datos ya inspeccionados; eleccion anidada no elimina exploracion previa',
        codigo=firma(Path(__file__)),ranker=firma(AQUI/'ranker_plantillas.py'),
        puntuador=firma(AQUI/'puntuar_tipos_fijos.py'),
        predicciones_previas=firma(padre/'tipos_133.csv'),auditoria_previa=firma(padre/'auditoria_anidada.json'),
        rasgos=firma(padre/'rasgos.npz'))
    rp=SALIDA/'protocolo_evaluacion.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    pred=d.prediccion.to_numpy().copy();nombres=d.modelo_anidado.to_numpy().copy();auditoria=[]
    with threadpool_limits(limits=1):
        for anterior in previos:
            grupo=anterior['grupo_exterior'];tr=dev & (g!=grupo)
            assert set(anterior['train'])==set(d.loc[tr,'archivo']) and grupo not in set(g[tr])
            for q in anterior['internos']:
                assert grupo!=q['grupo_fuera'] and grupo not in q['grupos_train']
            score,internos=validar_interno(x[tr],y[tr],g[tr])
            assert all(grupo!=q['grupo_fuera'] and grupo not in q['grupos_train'] for q in internos)
            scores=dict(anterior['scores_internos'],compartido=score)
            ganador=max(scores,key=scores.get)
            assert ganador in (anterior['modelo'],'compartido')
            test=(g==grupo) & d.suficiente.to_numpy() & d.detector_auditoria.ne('N').to_numpy()
            if ganador=='compartido':
                modelo=RankerPlantillas().fit(x[tr],y[tr]);pred[test]=modelo.predict(x[test]);nombres[test]=ganador
            auditoria.append(dict(grupo_exterior=grupo,train=d.loc[tr,'archivo'].tolist(),
                modelo=ganador,scores_internos=scores,internos_compartido=internos,
                modelo_previo=anterior['modelo'],internos_previos=anterior['internos']))
            print('Menú anidado:',grupo,ganador,scores,flush=True)
    d=d.assign(prediccion=pred,modelo_anidado=nombres)
    d['tipo_correcto']=d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'auditoria_anidada.json').write_text(json.dumps(auditoria,indent=2),encoding='utf-8')
    evaluar(d,SALIDA)


if __name__=='__main__':main()
