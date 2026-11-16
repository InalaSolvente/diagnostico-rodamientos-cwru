"""Representación compacta: patrones armónicos coherentes dentro de cada banda.

Transformación por registro, sin etiquetas. Reutiliza la selección anidada por
grupos; no escoge un clasificador mirando los resultados exteriores.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from validar_tipo_anidado import elegir, modelos

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_tipo_compacto_desarrollo'


def compactar(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 1953 or not np.isfinite(x).all():
        raise ValueError('Se requiere matriz de rasgos arbol L4 finita')
    armonicos = x[:, :1860].reshape(-1, 31, 5, 6, 2)
    estadisticas = x[:, 1860:].reshape(-1, 31, 3)
    salida = []
    for familias, estados in zip(armonicos, estadisticas):
        partes = []
        for familia in range(5):
            bandas = familias[:, familia]
            # Fundamental y al menos otra armónica fuertes EN LA MISMA banda.
            coherencia = np.minimum(bandas[:, 0, 0], bandas[:, 1:, 0].max(axis=1))
            indices = np.argsort(-coherencia, kind='stable')[:3]
            partes.extend(bandas[indices].ravel())
        partes.extend(np.r_[estados.mean(axis=0), np.median(estados, axis=0), estados.std(axis=0)])
        salida.append(partes)
    z = np.asarray(salida)
    assert z.shape == (len(x), 189) and np.isfinite(z).all()
    return z


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_tipo_seleccion_desarrollo'
    meta = pd.read_csv(fuente/'predicciones_oof.csv')
    with np.load(fuente/'rasgos.npz', allow_pickle=False) as f:
        x = compactar(f['x'])
    protocolo = dict(version='compacto_anidado_v1', dimensiones=189,
        representacion='tres bandas con fundamental y otra armónica coherentes por cada una de cinco familias; nueve estadísticas globales',
        seleccion='misma validación anidada, mismos tres clasificadores y prefijo 2s',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seleccion_sha256=hashlib.sha256((AQUI/'validar_tipo_anidado.py').read_bytes()).hexdigest(),
        rasgos_origen_sha256=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    ruta = SALIDA/'protocolo.json'
    if ruta.exists():
        assert json.loads(ruta.read_text(encoding='utf-8')) == protocolo
    else:
        ruta.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    pred = np.empty(len(y), dtype=object); elegidos = np.empty(len(y), dtype=object)
    auditoria = []
    for grupo in sorted(set(grupos)):
        test = grupos == grupo; train = ~test
        nombre, scores, internos = elegir(x[train], y[train], grupos[train])
        clf = modelos()[nombre]().fit(x[train], y[train])
        pred[test] = clf.predict(x[test]); elegidos[test] = nombre
        auditoria.append(dict(grupo_exterior=grupo, train=meta.loc[train,'archivo'].tolist(),
            test=meta.loc[test,'archivo'].tolist(), modelo_elegido=nombre,
            scores_internos=scores, validaciones_internas=internos))
        print(f'Compacto: {grupo[:20]} {nombre}', flush=True)
    d = meta.drop(columns=['prediccion','acierto']).assign(prediccion=pred, modelo=elegidos, acierto=pred == y)
    d.to_csv(SALIDA/'predicciones_oof.csv', index=False)
    (SALIDA/'auditoria_pliegues.json').write_text(json.dumps(auditoria, indent=2), encoding='utf-8')
    print(d.groupby(d.real.eq('N').map({True:'normales',False:'fallas'})).acierto.agg(['sum','count']).to_string())
