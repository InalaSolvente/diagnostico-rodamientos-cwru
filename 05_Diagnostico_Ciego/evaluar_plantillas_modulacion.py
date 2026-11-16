"""Selección anidada con curvas y modulación, manteniendo grupos OR completos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modelo_por_dominio import firma
from evaluar_plantillas_coherentes import elegir_modelo,modelos
from puntuar_tipos_fijos import evaluar

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_plantillas_modulacion'


def main():
    d=pd.read_csv(SALIDA/'muestra.csv')
    with np.load(SALIDA/'rasgos.npz',allow_pickle=False) as z:x=z['x']
    assert len(d)==133 and x.shape==(133,964)
    prep=json.loads((SALIDA/'protocolo_preparacion.json').read_text(encoding='utf-8'))
    assert prep['codigo']==firma(AQUI/'plantillas_modulacion.py')
    assert prep['laterales']==firma(AQUI/'bandas_laterales.py')
    original=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    dev=d.archivo.isin(original.loc[original.real.ne('N'),'archivo']).to_numpy()
    y=d.real.to_numpy();g=d.grupo_auditoria.to_numpy()
    assert dev.sum()==52 and not any('@' in s for s in g)
    protocolo=dict(version='curvas_modulacion_964_anidado_v1',
        exterior='tipo+tamaño, posiciones OR/cargas/sensores/muestreos fuera juntos',
        interior='LDA o SVM RBF fijos; elección por exactitud equilibrada solo dentro del entrenamiento exterior',
        tipo='clasificador solamente, sin reglas que sobrescriben por D/PD',
        detector='mismo detector crudo con grupos completos de la referencia',
        grado='mismo banco31 y regla ND/PD/D, bandas fijadas antes de verificar',
        limitacion='dataset ya inspeccionado; selección anidada no es validación externa',
        codigo=firma(Path(__file__)),preparador=firma(AQUI/'plantillas_modulacion.py'),
        rasgos=firma(SALIDA/'rasgos.npz'),muestra=firma(SALIDA/'muestra.csv'),
        seleccion_modelo=firma(AQUI/'evaluar_plantillas_coherentes.py'),
        modelos=firma(AQUI/'invariancia_portadora.py'),puntuador=firma(AQUI/'puntuar_tipos_fijos.py'))
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
                    print('Modulación anidada:',r.grupo_auditoria,nombre,scores,flush=True)
                nombre,clf=ajustes[r.grupo_auditoria];t=str(clf.predict(x[i:i+1])[0])
            pred.append(t);elegidos.append(nombre)
    d=d.drop(columns=['prediccion','grado','tipo_correcto']).assign(prediccion=pred,modelo_anidado=elegidos)
    d['tipo_correcto']=d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'auditoria_anidada.json').write_text(json.dumps(auditoria,indent=2),encoding='utf-8')
    evaluar(d,SALIDA)


if __name__=='__main__':main()
