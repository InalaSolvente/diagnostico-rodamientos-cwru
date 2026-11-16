from pathlib import Path
import inspect,json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from entrenar_48k import crear_particion,crear_modelo,F
from predictor_48k import predecir,MODELO_48K


if __name__=='__main__':
    assert set(inspect.signature(predecir).parameters)=={'x','fs','frecuencias_fabricante','modelo'}
    datos=Path(F.RAIZ_PROYECTO)/'data2/CWRU'
    idx=F.leer_indice(str(datos));idx=idx[idx.fs==48000]
    aqui=Path(__file__).parent
    prev=pd.read_csv(aqui/'particion_desarrollo.csv').set_index('archivo').particion.to_dict()
    a=crear_particion(idx,prev);b=crear_particion(idx.sample(frac=1,random_state=3),prev)
    assert a.equals(b) and a.particion.eq('desarrollo').sum()==14 and a.particion.eq('reserva').sum()==14
    if MODELO_48K.exists():
        salida=MODELO_48K.parent
        meta=json.loads((salida/'entrenamiento.json').read_text(encoding='utf-8'))
        assert {r['archivo'] for r in meta}==set(a.loc[a.particion=='desarrollo','archivo'])
        auditoria=json.loads((salida/'auditoria_grupos.json').read_text(encoding='utf-8'))
        for g in auditoria:
            assert set(g['entrenamiento']).isdisjoint(g['prueba'])
            assert g['clases_entrenadas']==['B','IR','N']
        with np.load(salida/'rasgos_desarrollo.npz',allow_pickle=False) as banco:x=banco['x']
        y=np.array([r['real'] for r in meta]);grupos=np.array([r['grupo'] for r in meta])
        valid=pd.read_csv(salida/'validacion.csv').set_index('archivo')
        for grupo in sorted(set(grupos[y!='N'])):
            test=grupos==grupo;clf=crear_modelo();clf.fit(x[~test],y[~test])
            esperado=valid.loc[[meta[i]['archivo'] for i in np.flatnonzero(test)],'prediccion_oof'].to_numpy()
            assert np.array_equal(clf.predict(x[test]),esperado)
        r=meta[0];fila=idx[idx.archivo==r['archivo']].iloc[0]
        mat=loadmat(datos/r['archivo']);clave=next(k for k in mat if k.endswith(f'_{fila.sensor_falla}_time'))
        pred=predecir(np.ravel(mat[clave]),48000,r['catalogo'])
        with np.load(MODELO_48K,allow_pickle=False) as m:
            scores=((x[0]-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
            assert pred['tipo']==m['clases'][int(scores.argmax())]
        try:predecir(np.ravel(mat[clave]),12000,r['catalogo'])
        except ValueError:pass
        else:raise AssertionError('Aceptó fs incorrecta')
    print('OK: mitad fija, seis grupos, clases entrenadas por fold, inferencia sin fd y modelo propio')
