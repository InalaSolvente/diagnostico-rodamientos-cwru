"""Clasificador no lineal por archivo; exclusión estricta y evidencia84 sin gate."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import firma
from bandas_estables import seleccionar
from clasificar_evidencia_seleccion import extraer, comprobar, TIPOS
from evaluar_retirada_control_pd import controlar
from evaluar_acp_denso import guardar_resumen_tipos

AQUI = Path(__file__).parent
FUENTE = AQUI / 'resultados_temporales_acp'
DATOS = AQUI / 'resultados_hist_temporal'


def crear():
    return HistGradientBoostingClassifier(learning_rate=.05, max_iter=120,
        max_depth=3, max_leaf_nodes=7, min_samples_leaf=3, l2_regularization=1.,
        max_bins=63, early_stopping=False, random_state=2026)


def main():
    DATOS.mkdir(exist_ok=True)
    original = pd.read_csv(FUENTE / 'comparacion_133.csv')
    columnas = ['archivo', 'sensor', 'fs', 'suficiente', 'sha256_senal', 'registro', 'catalogo',
        'real', 'grupo', 'grupo_auditoria', 'detector_auditoria', 'paper']
    d = original[columnas].copy()
    with np.load(FUENTE / 'rasgos.npz', allow_pickle=False) as z:
        temporales = z['x']
    assert temporales.shape == (133, 12, 2621)
    ep = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(ep.read_text(encoding='utf-8'))}
    calidad = np.full((133, 84), np.nan)
    for i, r in enumerate(d.itertuples()):
        if not r.suficiente:
            continue
        q = alternativas[r.archivo]
        if not q:
            mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
            signal = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
            assert hashlib.sha256(signal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            q = {t: seleccionar(signal[:len(signal)//2], int(r.fs), json.loads(r.catalogo), t) for t in TIPOS}
        calidad[i] = extraer(q)
    comprobar(next(q for q in alternativas.values() if q))
    # Las 12 vistas se agregan ANTES de ajustar: un único ejemplo por archivo.
    x = np.column_stack([temporales.mean(axis=1), calidad])
    assert x.shape == (133, 2705) and np.isfinite(x[d.suficiente]).all()
    np.savez_compressed(DATOS / 'rasgos_juez84_sin_gate.npz', x=calidad)
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert dev.sum() == 52 and not any('@' in q for q in g)
    protocolo = dict(codigo=firma(Path(__file__)), temporales=firma(FUENTE / 'rasgos.npz'),
        metadata=firma(FUENTE / 'comparacion_133.csv'), evidencia_fuente=firma(ep),
        rasgos84=firma(DATOS / 'rasgos_juez84_sin_gate.npz'),
        extractor84=firma(AQUI / 'clasificar_evidencia_seleccion.py'),
        metrica=firma(AQUI / 'evaluar_acp_denso.py'),
        receta='HistGradientBoosting, parámetros fijos; media temporal2621 +84 de selección',
        entrenamiento='52 archivos con falla; un ejemplo por archivo; pesos equilibrados por tipo',
        etiquetas='solo y_train en ajuste; ningún grado de paper ni frecuencia verdadera en inferencia',
        exclusiones='tipo+tamaño completos, todas las posiciones OR/cargas/sensores/muestreos',
        rasgos84_independientes_del_gate=True, detector='mismo detector por grupo completo',
        variantes='con controles D/PD históricos o sin forzar tipo; ambas fijadas antes de puntuar',
        solo_tipo=True, exploratorio=True, validacion_externa=False)
    rp = DATOS / 'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    modelos, folds, bases, controlados = {}, [], [], []
    with threadpool_limits(limits=1):
        for i, r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria == 'N':
                tb = t = 'N' if r.suficiente else 'sin_diagnostico'
            else:
                if r.grupo_auditoria not in modelos:
                    tr = dev & (g != r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr, 'archivo'])
                    pesos = np.array([tr.sum()/(3*np.sum(y[tr] == c)) for c in y[tr]])
                    clf = crear().fit(x[tr], y[tr], sample_weight=pesos)
                    modelos[r.grupo_auditoria] = clf
                    folds.append(dict(grupo_exterior=r.grupo_auditoria,
                        train=d.loc[tr, 'archivo'].tolist(), n_ejemplos=int(tr.sum())))
                    print('Hist por archivo, fuera:', r.grupo_auditoria, flush=True)
                tb = str(modelos[r.grupo_auditoria].predict(x[i:i+1])[0])
                t = controlar(tb, r.sensor, alternativas[r.archivo], usar_d=True, usar_pd=True)
            bases.append(tb)
            controlados.append(t)
    pendientes = []
    for nombre, pred in [('resultados_hist_temporal_sin_control', bases), ('resultados_hist_temporal_control', controlados)]:
        salida = AQUI / nombre
        salida.mkdir(exist_ok=True)
        v = d.copy()
        v['prediccion'], v['tipo_base_hist'] = pred, bases
        v['tipo_correcto'] = v.real.eq(v.prediccion)
        v.to_csv(salida / 'tipos_133.csv', index=False)
        (salida / 'protocolo.json').write_text(json.dumps(dict(protocolo, variante=nombre), ensure_ascii=False, indent=2), encoding='utf-8')
        (salida / 'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        pendientes.append((v, salida))
    for v, salida in pendientes:
        guardar_resumen_tipos(v, salida)


if __name__ == '__main__':
    main()
