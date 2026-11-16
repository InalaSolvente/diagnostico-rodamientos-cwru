"""LDA temporal con probabilidades previas uniformes; mismos datos y controles."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from evaluar_retirada_control_pd import controlar
from puntuar_banco57 import evaluar

AQUI = Path(__file__).parent
FUENTE = AQUI / 'resultados_temporales_acp'
SALIDA = AQUI / 'resultados_temporal_priores'


def crear_equilibrado():
    clf = crear()
    clf.steps[-1][1].set_params(priors=np.full(3, 1/3))
    return clf


def main():
    SALIDA.mkdir(exist_ok=True)
    d = pd.read_csv(FUENTE / 'muestra.csv').drop(columns=['grado'])
    with np.load(FUENTE / 'rasgos.npz', allow_pickle=False) as z:
        x = z['x']
    assert x.shape == (133, 12, 2621) and np.isfinite(x[d.suficiente]).all()
    verificado = json.loads((FUENTE / 'verificacion.json').read_text(encoding='utf-8'))
    assert verificado['archivos_recalculados'] == 132 and verificado['entrenamiento_52_grupos_completos']
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert dev.sum() == 52 and not any('@' in q for q in g)
    ep = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(ep.read_text(encoding='utf-8'))}
    protocolo = dict(codigo=firma(Path(__file__)), rasgos=firma(FUENTE / 'rasgos.npz'),
        muestra=firma(FUENTE / 'muestra.csv'), verificacion_rasgos=firma(FUENTE / 'verificacion.json'),
        modelo=firma(AQUI / 'modelo_por_dominio.py'), controlador=firma(AQUI / 'evaluar_retirada_control_pd.py'),
        puntuador=firma(AQUI / 'puntuar_banco57.py'),
        cambio='LDA priors=[1/3,1/3,1/3] para B, IR, OR; no cambia la proporción de evaluación',
        resto='mismos rasgos, doce vistas correlacionadas, detector, controles y banco57',
        entrenamiento_fallas=52, exclusiones='tipo+tamaño completos, todas las posiciones OR juntas',
        eleccion_por_archivo=False, exploratorio=True, validacion_externa=False)
    rp = SALIDA / 'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    modelos, folds, pred, bases = {}, [], [], []
    with threadpool_limits(limits=1):
        for i, r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria == 'N':
                tb = t = 'N' if r.suficiente else 'sin_diagnostico'
            else:
                if r.grupo_auditoria not in modelos:
                    tr = dev & (g != r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr, 'archivo'])
                    clf = crear_equilibrado().fit(x[tr].reshape(-1, 2621), np.repeat(y[tr], 12))
                    assert list(clf.classes_) == ['B', 'IR', 'OR']
                    assert np.allclose(clf.steps[-1][1].priors_, np.full(3, 1/3))
                    modelos[r.grupo_auditoria] = clf
                    folds.append(dict(grupo_exterior=r.grupo_auditoria,
                        train=d.loc[tr, 'archivo'].tolist(), priors=clf.steps[-1][1].priors_.tolist(),
                        cuentas_train={c: int(np.sum(y[tr] == c)) for c in clf.classes_}))
                    print('Priores uniformes, fuera:', r.grupo_auditoria, flush=True)
                tb = predecir_tipo(modelos[r.grupo_auditoria], x[i])
                t = controlar(tb, r.sensor, alternativas[r.archivo], usar_d=True, usar_pd=True)
            pred.append(t)
            bases.append(tb)
    d['prediccion'], d['tipo_base_priores'] = pred, bases
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA / 'tipos_133.csv', index=False)
    (SALIDA / 'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    evaluar(d, SALIDA)


if __name__ == '__main__':
    main()
