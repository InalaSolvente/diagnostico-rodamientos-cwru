"""Prueba de inferencia real y equivalencia con rasgos exportados, sin consultar etiqueta."""
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
from scipy.io import loadmat
from predictor_espectral import predecir, MODELO
from modelo_ciego import F


if __name__=="__main__":
    assert set(inspect.signature(predecir).parameters)=={"x","fs","frecuencias_fabricante","modelo"}
    salida=MODELO.parent
    meta=pd.read_csv(salida/"muestra.csv")
    datos=Path(F.RAIZ_PROYECTO)/"data2/CWRU"
    idx=F.leer_indice(str(datos))
    r=idx[idx.archivo==meta.iloc[0].archivo].iloc[0]
    d=loadmat(datos/r.archivo)
    clave=F.clave_canal(d, r.sensor_falla, int(r.registro))
    rpm_keys=[k for k in d if k.endswith("RPM")]
    rpm=F.rpm_registro(d, int(r.registro), nominal=float(r.rpm_nominal))
    catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
    def prohibido(*args,**kwargs):
        raise AssertionError("La inferencia intentó consultar fd real")
    anterior=F.frecuencia_falla; F.frecuencia_falla=prohibido
    try:
        a=predecir(np.ravel(d[clave]),float(r.fs),catalogo)
        b=predecir(np.ravel(d[clave]),float(r.fs),dict(reversed(list(catalogo.items()))))
    finally:
        F.frecuencia_falla=anterior
    assert a==b
    with np.load(MODELO,allow_pickle=False) as m, np.load(salida/"bancos_desarrollo.npz",allow_pickle=False) as bancos:
        x=bancos[str(m['banco'])][0]
        scores=((x-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
        assert a['tipo']==m['clases'][int(scores.argmax())]
        assert np.allclose([a['scores'][c] for c in m['clases']],scores)
    try:
        predecir(np.ravel(d[clave]),float(r.fs),catalogo,fd=100)
    except TypeError:
        pass
    else:
        raise AssertionError("Aceptó fd como entrada")
    print("OK: senal real, rasgos y modelo; catalogo completo, sin fd y exportacion equivalente")
