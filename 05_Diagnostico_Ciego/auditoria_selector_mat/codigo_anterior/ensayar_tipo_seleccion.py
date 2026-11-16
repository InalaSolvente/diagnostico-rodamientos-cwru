"""Candidato de desarrollo: tipo desde primeros 2 s, sin acceso al tramo verificador.

Una receta fija, sin escoger clasificadores por su puntuación exterior. No abre
la reserva. Normales por adquisición única; fallas por condición nominal completa.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from modelo_ciego import F
from modelo_espectral import extraer
from ranking_catalogo import grupo_condicion

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_tipo_seleccion_desarrollo'


def canonizar_prefijo(x, fs):
    """Misma duración, fs y bandas físicas; no lee muestras posteriores a 2 s."""
    if fs not in (12000, 48000) or len(x) < 2*fs:
        raise ValueError('Se requieren al menos 2 s y muestreo 12/48 kHz')
    prefijo = np.asarray(x[:int(2*fs)], dtype=float)
    return prefijo if fs == 12000 else resample_poly(prefijo, 1, 4, padtype='line')


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    protocolo = dict(version='tipo_prefijo_v1', segundos=2, fs_canonica=12000, L=4,
        clasificador='StandardScaler + LogisticRegression(C=.1, class_weight=balanced, max_iter=2000)',
        justificacion='Regularización fuerte fija con p>>n; no hay selección de hiperparámetros por resultados',
        validacion='cada condición de falla fuera completa; cada señal normal única fuera individualmente',
        limitacion='Desarrollo exploratorio, dos normales; no validación externa ni prueba de sano/falla por rodamiento nuevo',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        extractor_sha256=hashlib.sha256((AQUI/'modelo_espectral.py').read_bytes()).hexdigest())
    ruta_protocolo = SALIDA/'protocolo.json'
    if ruta_protocolo.exists():
        assert json.loads(ruta_protocolo.read_text(encoding='utf-8')) == protocolo
    else:
        ruta_protocolo.write_text(json.dumps(protocolo, indent=2, ensure_ascii=False), encoding='utf-8')
    split = pd.read_csv(AQUI/'particion_desarrollo.csv')
    dev = set(split.loc[split.particion.eq('desarrollo'), 'archivo'])
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU'))
    idx = idx[idx.archivo.isin(dev)].sort_values('archivo')
    assert len(idx) == 54 and set(idx.archivo) == dev
    filas = []
    rasgos = []
    for i, r in enumerate(idx.itertuples()):
        ruta = AQUI.parent/'data2/CWRU'/r.archivo
        mat = loadmat(ruta)
        clave = next(k for k in mat if k.endswith(f'_{r.sensor_falla}_time'))
        x = np.ravel(mat[clave])
        krpm = [k for k in mat if k.endswith('RPM')]
        rpm = float(np.ravel(mat[krpm[0]])[0]) if krpm else float(r.rpm_nominal)
        c = {k: float(v*rpm/60) for k, v in F.MULTIPLICADORES[r.sensor_falla].items()}
        z = canonizar_prefijo(x, int(r.fs))
        assert len(z) == 24000
        f = extraer(z, 12000, c, L=4)['arbol']
        rasgos.append(f)
        huella = hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        filas.append(dict(archivo=r.archivo, real=r.zona, sensor=r.sensor_falla,
            grupo=('normal:'+huella) if r.zona == 'N' else grupo_condicion(r.archivo),
            sha256_senal=huella, sha256_prefijo=hashlib.sha256(z.astype('<f8').tobytes()).hexdigest()))
        print(f'Prefijo {i+1}/54', flush=True)
    meta = pd.DataFrame(filas)
    assert meta.sha256_senal.is_unique
    x = np.asarray(rasgos)
    np.savez_compressed(SALIDA/'rasgos.npz', x=x)
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    pred = np.empty(len(y), dtype=object)
    auditoria = []
    for grupo in sorted(set(grupos)):
        test = grupos == grupo; train = ~test
        assert grupo not in set(grupos[train]) and set(y[train]) == set(y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=2000))
        clf.fit(x[train], y[train])
        pred[test] = clf.predict(x[test])
        auditoria.append(dict(grupo=grupo, train=meta.loc[train, 'archivo'].tolist(), test=meta.loc[test, 'archivo'].tolist()))
    meta['prediccion'] = pred
    meta['acierto'] = pred == y
    meta.to_csv(SALIDA/'predicciones_oof.csv', index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(auditoria, indent=2), encoding='utf-8')
    print(meta.groupby(meta.real.eq('N').map({True:'normales',False:'fallas'})).acierto.agg(['sum','count']).to_string())
    # Prueba de causalidad del preprocesado: variar el futuro no cambia el prefijo.
    rng = np.random.default_rng(123)
    prueba = rng.normal(size=60000)
    antes = canonizar_prefijo(prueba, 12000)
    prueba[24000:] *= 100
    assert np.array_equal(antes, canonizar_prefijo(prueba, 12000))
    print('OOF completo; ningún predictor usa el tramo posterior a 2 s.')
