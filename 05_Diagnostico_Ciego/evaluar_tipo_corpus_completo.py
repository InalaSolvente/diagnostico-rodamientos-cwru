"""Amplía solo el entrenamiento del tipo; conserva el LDA temporal y el detector."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from evaluar_retirada_control_pd import controlar
from metricas_tipo import guardar

AQUI = Path(__file__).parent
FUENTE = AQUI / 'resultados_temporales_acp'
DATOS = AQUI / 'resultados_tipo_corpus_completo'


def main():
    DATOS.mkdir(exist_ok=True)
    origen = pd.read_csv(FUENTE / 'comparacion_133.csv')
    columnas = ['archivo', 'sensor', 'fs', 'suficiente', 'sha256_senal', 'registro', 'catalogo',
        'real', 'grupo', 'grupo_auditoria', 'detector_auditoria', 'paper']
    d = origen[columnas].copy()
    with np.load(FUENTE / 'rasgos.npz', allow_pickle=False) as z:
        x = z['x']
    assert x.shape == (133, 12, 2621) and np.isfinite(x[d.suficiente]).all()
    v = json.loads((FUENTE / 'verificacion.json').read_text(encoding='utf-8'))
    assert v['archivos_recalculados'] == 132 and v['entrenamiento_52_grupos_completos']
    corpus = d.suficiente.to_numpy() & d.real.ne('N').to_numpy()
    assert corpus.sum() == 128
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert not any('@' in q for q in g)
    ep = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(ep.read_text(encoding='utf-8'))}
    protocolo = dict(codigo=firma(Path(__file__)), rasgos=firma(FUENTE / 'rasgos.npz'),
        metadata=firma(FUENTE / 'comparacion_133.csv'), verificacion_rasgos=firma(FUENTE / 'verificacion.json'),
        modelo=firma(AQUI / 'modelo_por_dominio.py'), controlador=firma(AQUI / 'evaluar_retirada_control_pd.py'),
        metrica=firma(AQUI / 'metricas_tipo.py'), autorizacion='usuario autorizó ampliar con exclusión completa del grupo',
        receta='mismo LDA temporal2621 y doce vistas; solo cambia el corpus de entrenamiento del tipo',
        corpus_fallas=128, etiquetas='solo y_train; no grado de paper ni frecuencia verdadera en inferencia',
        exclusiones='tipo+tamaño completos, todas las posiciones OR, cargas, sensores y muestreos juntos',
        detector='mismo detector estricto anterior para aislar el cambio del clasificador de tipo',
        archivo_corto='fuera de entrenamiento, conservado en evaluación como sin diagnóstico',
        variantes='con controles históricos y sin forzar tipo', solo_tipo=True,
        exploratorio=True, validacion_externa=False)
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
                    tr = corpus & (g != r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr, 'archivo'])
                    assert set(y[tr]) == {'B', 'IR', 'OR'}
                    modelos[r.grupo_auditoria] = crear().fit(x[tr].reshape(-1, 2621), np.repeat(y[tr], 12))
                    folds.append(dict(grupo_exterior=r.grupo_auditoria, n_archivos_train=int(tr.sum()),
                        train=d.loc[tr, 'archivo'].tolist(), grupos_train=sorted(set(g[tr])),
                        n_excluidos_mismo_grupo=int((corpus & (g == r.grupo_auditoria)).sum())))
                    print('Corpus completo, fuera:', r.grupo_auditoria, 'train:', int(tr.sum()), flush=True)
                tb = predecir_tipo(modelos[r.grupo_auditoria], x[i])
                t = controlar(tb, r.sensor, alternativas[r.archivo], usar_d=True, usar_pd=True)
            bases.append(tb)
            controlados.append(t)
    pendientes = []
    for nombre, pred in [('resultados_tipo_completo_sin_control', bases), ('resultados_tipo_completo_control', controlados)]:
        salida = AQUI / nombre
        salida.mkdir(exist_ok=True)
        datos = d.copy()
        datos['prediccion'], datos['tipo_base_completo'] = pred, bases
        datos['tipo_correcto'] = datos.real.eq(datos.prediccion)
        datos.to_csv(salida / 'tipos_133.csv', index=False)
        (salida / 'protocolo.json').write_text(json.dumps(dict(protocolo, variante=nombre), ensure_ascii=False, indent=2), encoding='utf-8')
        (salida / 'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        pendientes.append((datos, salida))
    for datos, salida in pendientes:
        guardar(datos, salida, corpus_fallas=int(corpus.sum()))


if __name__ == '__main__':
    main()
