"""Comprueba causalidad y exclusión de ventanas por registro/condición."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ventanas_verificadas import prefijo


if __name__ == '__main__':
    p = Path(__file__).parent
    fuente = p/'resultados_ventanas_verificadas'
    salida = p/'resultados_verificados_ventanas'
    meta = pd.read_csv(fuente/'muestra.csv').set_index('archivo')
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:
        origen = z['origen']; assert z['x'].shape == (216,1953)
    assert np.array_equal(np.bincount(origen),np.full(54,4))
    folds = json.loads((salida/'pliegues.json').read_text(encoding='utf-8'))
    cubiertos = []
    for f in folds:
        train,test = set(f['train']),set(f['test'])
        assert train.isdisjoint(test) and train|test == set(meta.index)
        assert f['grupo'] not in set(meta.loc[list(train),'grupo'])
        assert set(meta.loc[list(train),'sha256_senal']).isdisjoint(set(meta.loc[list(test),'sha256_senal']))
        ti = np.flatnonzero(meta.index.isin(train));vi = np.flatnonzero(meta.index.isin(test))
        assert set(origen[np.isin(origen,ti)]).isdisjoint(set(vi))
        cubiertos.extend(test)
    assert len(cubiertos) == len(set(cubiertos)) == 54
    rng = np.random.default_rng(143)
    for fs in (12000,48000):
        x = rng.normal(size=6*fs);antes = prefijo(x,fs)
        x[int(2.5*fs):] = rng.normal(size=len(x)-int(2.5*fs))*1000
        assert np.array_equal(antes,prefijo(x,fs))
    tipos = pd.read_csv(salida/'tipos_oof.csv')
    assert len(tipos) == 54 and tipos.archivo.is_unique
    print(tipos.assign(acierto=tipos.real.eq(tipos.prediccion)).groupby('real').acierto.agg(['sum','count']).to_string())
    (salida/'integridad.json').write_text(json.dumps(dict(registros=54,ventanas=216,
        ventanas_por_registro=4,exclusion_completa_por_grupo=True, prefijo_no_usa_futuro=True),indent=2),encoding='utf-8')
    print('OK: 54 casos OOF; ventanas del mismo registro excluidas juntas; prefijo causal a 12/48k. No son 216 casos de validación.')
