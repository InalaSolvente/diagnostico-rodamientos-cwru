"""LDA fijo + bandas estables con datos corregidos, solo desarrollo.

Tipo desde primeros 2,5 s. Banda/familia/transformación desde primera mitad.
Verificación de la misma elección en segunda mitad. Nunca recibe la falla real.
"""
from pathlib import Path
import hashlib
import inspect
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_espectral import extraer, VERSION_DATOS
from ventanas_verificadas import rasgos as rasgos_ventanas
from bandas_estables import seleccionar, verificar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
FUENTE = AQUI/'resultados_ventanas_verificadas'
SALIDA = AQUI/'resultados_verificados_ventanas'


def hash_archivo(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def firma_lector():
    texto = ''.join(inspect.getsource(f) for f in (F.leer_indice, F.clave_canal, F.rpm_registro))
    return hashlib.sha256(texto.encode()).hexdigest()


def clasificador():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))


def diagnosticar(x, fs, catalogo, tipo):
    mitad = len(x)//2
    if mitad < 2.5*fs:
        raise ValueError('Se necesitan al menos 5 s para separar selección y verificación')
    seleccion = seleccionar(x[:mitad], fs, catalogo, tipo)
    verificacion = verificar(x[mitad:], fs, catalogo, seleccion)
    return dict(tipo=tipo, grado=verificacion['grado'], seleccion=seleccion, verificacion=verificacion)


def predecir(x, fs, catalogo, *, modelo=SALIDA/'modelo.npz'):
    """Señal, fs y catálogo completo; sin registro, etiqueta ni frecuencia esperada."""
    with np.load(modelo, allow_pickle=False) as m:
        if str(m['extractor_sha256']) != hash_archivo(AQUI/'modelo_espectral.py'):
            raise ValueError('Extractor cambiado; requiere nuevo entrenamiento')
        if str(m['ventanas_sha256']) != hash_archivo(AQUI/'ventanas_verificadas.py'):
            raise ValueError('Ventanas cambiadas; requiere nuevo entrenamiento')
        z = rasgos_ventanas(x,fs,catalogo).mean(axis=0)
        scores = ((z-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
        tipo = str(m['clases'][scores.argmax()])
        resultado = diagnosticar(x, fs, catalogo, tipo)
        resultado['scores'] = {str(c):float(s) for c,s in zip(m['clases'],scores)}
        resultado['nota'] = 'Scores discriminantes; no probabilidades calibradas'
        return resultado


def ejecutar():
    SALIDA.mkdir(exist_ok=True)
    meta = pd.read_csv(FUENTE/'muestra.csv')
    with np.load(FUENTE/'rasgos.npz', allow_pickle=False) as f:
        x = f['x']
        origen = f['origen']
    assert x.shape == (216,1953) and meta.sha256_senal.is_unique
    assert np.array_equal(np.bincount(origen),np.full(54,4))
    assert meta.version_datos.eq(VERSION_DATOS).all()
    split = pd.read_csv(AQUI/'particion_desarrollo.csv')
    dev = set(split.loc[split.particion.eq('desarrollo'),'archivo'])
    assert set(meta.archivo) == dev
    protocolo = dict(version='verificado_ventanas_v1', etapa='desarrollo 52 fallas y dos normales',
        clasificador='LDA lsqr shrinkage auto con 1953 rasgos; receta única, sin elegir por resultado exterior',
        inferencia='tipo promediando cuatro ventanas de 1 s en primeros 2,5 s; elección de banda en primera mitad; verificación fija en segunda',
        grupos='condición nominal completa; cada normal único excluido individualmente',
        codigo_sha256=hash_archivo(Path(__file__)), rasgos_sha256=hash_archivo(FUENTE/'rasgos.npz'),
        lector_sha256=firma_lector(), extractor_sha256=hash_archivo(AQUI/'modelo_espectral.py'),
        bandas_sha256=hash_archivo(AQUI/'bandas_estables.py'), criterio_sha256=hash_archivo(AQUI/'criterio_usuario.py'),
        ventanas_sha256=hash_archivo(AQUI/'ventanas_verificadas.py'),
        reserva_abierta_para_elegir=False, validacion_externa=False)
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo, 'Receta cambiada: crear otro experimento'
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    pred = np.empty(len(y), dtype=object); folds = []
    with threadpool_limits(limits=1):
        for grupo in sorted(set(grupos)):
            test = grupos == grupo; train = ~test
            assert grupo not in set(grupos[train]) and set(y[train]) == set(y)
            filas_train = train[origen]
            assert set(origen[filas_train]).isdisjoint(set(np.flatnonzero(test)))
            clf = clasificador().fit(x[filas_train], y[origen[filas_train]])
            for i in np.flatnonzero(test):
                scores = clf.decision_function(x[origen == i]).mean(axis=0)
                pred[i] = clf.classes_[scores.argmax()]
            folds.append(dict(grupo=grupo, train=meta.loc[train,'archivo'].tolist(), test=meta.loc[test,'archivo'].tolist()))
            print(f'OOF correcto: {grupo[:25]}', flush=True)
    meta.assign(prediccion=pred).to_csv(SALIDA/'tipos_oof.csv', index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    filas = []
    for i, r in enumerate(meta.itertuples()):
        mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
        clave = F.clave_canal(mat, r.sensor if hasattr(r,'sensor') else ('FE' if r.archivo.startswith('12k_FE/') else 'DE'), int(r.registro))
        senal = np.ravel(mat[clave])
        assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        q = diagnosticar(senal, int(r.fs_original), json.loads(r.catalogo), str(pred[i]))
        filas.append(dict(archivo=r.archivo, **q))
        (SALIDA/'predicciones.json').write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Verificación fija {i+1}/54: {q["tipo"]} {q["grado"]}', flush=True)
    # La referencia se consulta después de fijar todas las predicciones.
    referencia = pd.read_csv(AQUI/'resultados_criterio_usuario_total/comparacion.csv').set_index('archivo').paper
    d = meta[['archivo','real']].copy()
    d['prediccion'] = pred
    d['grado'] = [r['grado'] for r in filas]
    d['paper'] = d.archivo.map(referencia)
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'comparacion.csv', index=False)
    fallas = d[d.paper.notna()]; delta = fallas.grado.map(ORDEN)-fallas.paper.map(ORDEN); ok = fallas.tipo_correcto
    normales = d[d.real.eq('N')]
    resumen = dict(total=52, tipo_correcto=int(ok.sum()), mejoran=int((ok&delta.gt(0)).sum()),
        iguales=int((ok&delta.eq(0)).sum()), empeoran=int((ok&delta.lt(0)).sum()),
        tipo_incorrecto=int((~ok).sum()), sin_retroceso=int((ok&delta.ge(0)).sum()),
        normales_correctos=int(normales.tipo_correcto.sum()), normales_total=len(normales),
        falsas_alarmas_con_evidencia=int((normales.prediccion.ne('N')&normales.grado.ne('ND')).sum()))
    assert sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 52
    (SALIDA/'resumen.json').write_text(json.dumps(resumen, indent=2), encoding='utf-8')
    with threadpool_limits(limits=1):
        final = clasificador().fit(x,y[origen])
    scaler, lda = final.steps[0][1], final.steps[1][1]
    np.savez_compressed(SALIDA/'modelo.npz', media=scaler.mean_, escala=scaler.scale_, coef=lda.coef_,
        intercepto=lda.intercept_, clases=lda.classes_, extractor_sha256=protocolo['extractor_sha256'],
        ventanas_sha256=protocolo['ventanas_sha256'])
    assert np.array_equal(lda.classes_[(((x-scaler.mean_)/scaler.scale_)@lda.coef_.T+lda.intercept_).argmax(axis=1)], final.predict(x))
    print(json.dumps(resumen), flush=True)


if __name__ == '__main__':
    ejecutar()
