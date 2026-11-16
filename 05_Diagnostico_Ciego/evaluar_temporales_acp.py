"""LDA sobre vistas de toda la mitad de selección; mismos grupos y controles."""
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
SALIDA = AQUI / 'resultados_temporales_acp'


def main():
    d = pd.read_csv(SALIDA / 'muestra.csv').drop(columns=['grado'])
    with np.load(SALIDA / 'rasgos.npz', allow_pickle=False) as z:
        x = z['x']
    assert x.shape == (133, 12, 2621) and d.archivo.is_unique
    preparacion = json.loads((SALIDA / 'protocolo_preparacion.json').read_text(encoding='utf-8'))
    assert preparacion['codigo'] == firma(AQUI / 'rasgos_temporales_acp.py')
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert dev.sum() == 52 and not any('@' in q for q in g)
    ep = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(ep.read_text(encoding='utf-8'))}
    protocolo = dict(codigo=firma(Path(__file__)), preparador=firma(AQUI / 'rasgos_temporales_acp.py'),
        rasgos=firma(SALIDA / 'rasgos.npz'), muestra=firma(SALIDA / 'muestra.csv'),
        modelo=firma(AQUI / 'modelo_por_dominio.py'), controlador=firma(AQUI / 'evaluar_retirada_control_pd.py'),
        puntuador=firma(AQUI / 'puntuar_banco57.py'),
        receta='LDA shrinkage auto fijo, 12 vistas por registro; media de decision_function',
        entrenamiento_fallas=52, muestras_independientes='archivos, no ventanas',
        exclusiones='tipo+tamaño completos; todas las posiciones OR, cargas, sensores y muestreos fuera',
        detector='sin cambios', controles='D y PD sin cambios', grados='banco57 y mismo juez',
        temporal='solo primera mitad para clasificar y seleccionar; segunda mitad solo verifica',
        exploratorio=True, validacion_externa=False)
    rp = SALIDA / 'protocolo_evaluacion.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    modelos, pliegues, pred, tipos_base = {}, [], [], []
    acuerdo = []
    with threadpool_limits(limits=1):
        for i, r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria == 'N':
                tb = t = 'N' if r.suficiente else 'sin_diagnostico'
                votos = []
            else:
                if r.grupo_auditoria not in modelos:
                    tr = dev & (g != r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr, 'archivo'])
                    modelos[r.grupo_auditoria] = crear().fit(x[tr].reshape(-1, 2621), np.repeat(y[tr], 12))
                    pliegues.append(dict(grupo_exterior=r.grupo_auditoria,
                        train=d.loc[tr, 'archivo'].tolist(), grupos_train=sorted(set(g[tr]))))
                    print('LDA temporal, fuera:', r.grupo_auditoria, flush=True)
                clf = modelos[r.grupo_auditoria]
                tb = predecir_tipo(clf, x[i])
                votos = [predecir_tipo(clf, v) for v in x[i].reshape(3, 4, 2621)]
                t = controlar(tb, r.sensor, alternativas[r.archivo], usar_d=True, usar_pd=True)
            pred.append(t)
            tipos_base.append(tb)
            acuerdo.append(json.dumps(votos))
    d['prediccion'], d['tipo_base_temporal'], d['votos_por_segmento'] = pred, tipos_base, acuerdo
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA / 'tipos_133.csv', index=False)
    (SALIDA / 'pliegues.json').write_text(json.dumps(pliegues, indent=2), encoding='utf-8')
    evaluar(d, SALIDA)


if __name__ == '__main__':
    main()
