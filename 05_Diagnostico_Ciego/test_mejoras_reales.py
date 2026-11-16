"""Integridad del desarrollo: futuro aislado, grupos excluidos y salidas completas."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from ensayar_tipo_seleccion import canonizar_prefijo
from modelo_ciego import F
from scipy.io import loadmat


if __name__ == '__main__':
    p = Path(__file__).parent
    rng = np.random.default_rng(45)
    for fs in (12000, 48000):
        x = rng.normal(size=4*fs)
        prefijo = canonizar_prefijo(x, fs)
        x[2*fs:] = 1000*rng.normal(size=2*fs)
        assert np.array_equal(prefijo, canonizar_prefijo(x, fs))
        assert prefijo.shape == (24000,)
    fuente = p/'resultados_tipo_anidado_desarrollo'
    meta = pd.read_csv(fuente/'predicciones_oof.csv').set_index('archivo')
    folds = json.loads((fuente/'auditoria_pliegues.json').read_text(encoding='utf-8'))
    evaluados = []
    for fold in folds:
        train, test = set(fold['train']), set(fold['test'])
        assert train.isdisjoint(test) and train|test == set(meta.index)
        assert set(meta.loc[list(train),'sha256_senal']).isdisjoint(set(meta.loc[list(test),'sha256_senal']))
        assert fold['grupo_exterior'] not in set(meta.loc[list(train),'grupo'])
        for interior in fold['validaciones_internas']:
            a, b = set(interior['grupos_train']), set(interior['grupos_test'])
            assert a.isdisjoint(b) and fold['grupo_exterior'] not in a|b
        evaluados.extend(test)
    assert len(evaluados) == 54 and len(set(evaluados)) == 54
    split = pd.read_csv(p/'particion_desarrollo.csv')
    assert set(meta.index) == set(split.loc[split.particion.eq('desarrollo'),'archivo'])
    indice = F.leer_indice(str(p.parent/'data2/CWRU')).set_index('archivo', drop=False)
    entradas = []
    for archivo, fila in meta.iterrows():
        r = indice.loc[archivo]
        ruta = p.parent/'data2/CWRU'/archivo
        mat = loadmat(ruta)
        clave = F.clave_canal(mat, r.sensor_falla, int(r.registro))
        x = np.ravel(mat[clave])
        huella = hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        prefijo = canonizar_prefijo(x, int(r.fs))
        assert huella == fila.sha256_senal
        assert hashlib.sha256(prefijo.astype('<f8').tobytes()).hexdigest() == fila.sha256_prefijo
        assert r.zona == fila.real
        krpm = [k for k in mat if k.endswith('RPM')]
        rpm = F.rpm_registro(mat, int(r.registro), nominal=float(r.rpm_nominal))
        entradas.append(dict(archivo=archivo, fs=int(r.fs), sensor=r.sensor_falla,
            sha256_mat=hashlib.sha256(ruta.read_bytes()).hexdigest(), sha256_senal=huella,
            rpm=rpm, rpm_fuente='MAT' if krpm else 'nominal del índice',
            catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}))
    (p/'resultados_tipo_seleccion_desarrollo/entradas_verificadas.json').write_text(
        json.dumps(entradas, ensure_ascii=False, indent=2), encoding='utf-8')
    d = pd.read_csv(p/'resultados_candidato_causal_desarrollo/comparacion.csv')
    assert len(d) == 54 and d.archivo.is_unique and d.grado.isin(['ND','PD','D']).all()
    assert d.set_index('archivo').tipo_predicho.eq(meta.prediccion).all()
    print('OK: 54 prefijos contrastados con señal cruda; catálogos registrados; futuro aislado; grupos anidados sin copias; reserva excluida')
