"""ACP600: selección interna de LDA, SVM o peine físico, sin forzar el tipo."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from invariancia_portadora import modelos
from modelo_por_dominio import firma
from evaluar_selector_controles_anidado import elegir
from puntuar_banco57 import evaluar

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_coherencia_ciclica'
NOMBRES = ('lda', 'svm', 'peine')


def peine(x):
    """Soporte común de H1 y otra armónica, misma desviación del catálogo."""
    soporte = x[:, 360:540].reshape(-1, 5, 3, 3, 4)
    # Medida del cuartil superior en portadora, máximo sobre desviación común.
    s = soporte[:, :, 2, 1, 1]
    scores = np.column_stack([np.maximum(s[:, 2], s[:, 3]), s[:, 0], s[:, 1]])
    return np.array(['B', 'IR', 'OR'])[scores.argmax(axis=1)]


def predecir(nombre, xtr, ytr, xtest):
    if nombre == 'peine':
        return peine(xtest)
    return modelos()[nombre]().fit(xtr, ytr).predict(xtest)


def main():
    d = pd.read_csv(SALIDA / 'muestra.csv')
    with np.load(SALIDA / 'rasgos.npz', allow_pickle=False) as z:
        x = z['x']
    assert len(d) == 133 and x.shape == (133, 600)
    prep = json.loads((SALIDA / 'protocolo_preparacion.json').read_text(encoding='utf-8'))
    assert prep['codigo'] == firma(AQUI / 'coherencia_ciclica.py')
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert dev.sum() == 52 and not any('@' in q for q in g)
    assert np.isfinite(x[d.suficiente]).all()
    protocolo = dict(codigo=firma(Path(__file__)), preparador=firma(AQUI / 'coherencia_ciclica.py'),
        rasgos=firma(SALIDA / 'rasgos.npz'), muestra=firma(SALIDA / 'muestra.csv'),
        modelos=firma(AQUI / 'invariancia_portadora.py'), seleccion=firma(AQUI / 'evaluar_selector_controles_anidado.py'),
        puntuador=firma(AQUI / 'puntuar_banco57.py'),
        candidatos='LDA shrinkage, SVM RBF C1 balanced o peine físico fijo',
        criterio='exactitud equilibrada interna, desempate por exactitud y orden fijo',
        particiones='exterior e interior por tipo+tamaño; OR completo, cargas, sensores y muestreos fuera',
        detector='mismo detector crudo con grupos completos',
        controles='ninguno sobrescribe el tipo', grados='banco57 y mismo juez, dos ramas fijadas en selección',
        entrenamiento_fallas=52, exploratorio=True, validacion_externa=False)
    rp = SALIDA / 'protocolo_evaluacion.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    pred = np.full(len(d), '', dtype=object)
    elegidos = np.full(len(d), 'sin_modelo_por_detector', dtype=object)
    auditoria = []
    with threadpool_limits(limits=1):
        for exterior in sorted(set(g)):
            te = np.flatnonzero(g == exterior)
            suficientes = d.loc[te, 'suficiente'].to_numpy()
            gate_n = d.loc[te, 'detector_auditoria'].eq('N').to_numpy()
            pred[te[~suficientes]] = 'sin_diagnostico'
            pred[te[suficientes & gate_n]] = 'N'
            utilizables = te[suficientes & ~gate_n]
            if not len(utilizables):
                continue
            tr = np.flatnonzero(dev & (g != exterior))
            assert set(g[tr]).isdisjoint(g[te]) and set(y[tr]) == {'B', 'IR', 'OR'}
            p = {nombre: np.full(len(d), '', dtype=object) for nombre in NOMBRES}
            internos = []
            for interior in sorted(set(g[tr])):
                vi = tr[g[tr] == interior]
                ajuste = tr[g[tr] != interior]
                assert set(g[ajuste]).isdisjoint({exterior, interior})
                for nombre in NOMBRES:
                    p[nombre][vi] = predecir(nombre, x[ajuste], y[ajuste], x[vi])
                internos.append(dict(grupo_interior=interior, train=d.loc[ajuste, 'archivo'].tolist(),
                                     validacion=d.loc[vi, 'archivo'].tolist()))
            reducidos = {k: v[tr] for k, v in p.items()}
            assert all(np.all(v != '') for v in reducidos.values())
            mejor, scores = elegir(reducidos, y[tr])
            yy = y.copy()
            yy[te] = 'etiqueta_exterior_inaccesible'
            assert elegir(reducidos, yy[tr]) == (mejor, scores)
            pred[utilizables] = predecir(mejor, x[tr], y[tr], x[utilizables])
            elegidos[te] = mejor
            auditoria.append(dict(grupo_exterior=exterior, train=d.loc[tr, 'archivo'].tolist(),
                test=d.loc[te, 'archivo'].tolist(), modelo=mejor, scores=scores, internos=internos,
                oof_interno={k: v.tolist() for k, v in reducidos.items()}))
            print('ACP anidado:', exterior, mejor, scores[mejor], flush=True)
    assert np.all(pred != '')
    d = d.drop(columns=['prediccion', 'tipo_correcto']).assign(prediccion=pred, modelo_anidado=elegidos)
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA / 'tipos_133.csv', index=False)
    (SALIDA / 'auditoria_anidada.json').write_text(json.dumps(auditoria, indent=2), encoding='utf-8')
    evaluar(d, SALIDA)


if __name__ == '__main__':
    main()
