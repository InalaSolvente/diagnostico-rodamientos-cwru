"""Añade evidencia de selección84 a crudo+ACP; elimina TODOS los forzados D/PD."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from clasificar_evidencia_seleccion import extraer, comprobar
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from puntuar_banco57 import evaluar

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_juez_aprendido_acp'


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_crudo_coherencia/comparacion_133.csv'
    d = pd.read_csv(fuente).drop(columns=['grado'])
    with np.load(AQUI / 'resultados_globales_arbol/rasgos.npz', allow_pickle=False) as z:
        original = z['x']
    with np.load(AQUI / 'resultados_coherencia_ciclica/rasgos.npz', allow_pickle=False) as z:
        coherencia = z['x']
    ep = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(ep.read_text(encoding='utf-8'))}
    comprobar(next(q for q in alternativas.values() if q))
    evidencia = np.zeros((133, 84))
    for i, r in enumerate(d.itertuples()):
        if r.suficiente and r.detector_auditoria != 'N':
            evidencia[i] = extraer(alternativas[r.archivo])
    assert evidencia.shape == (133, 84) and np.isfinite(evidencia).all()
    x = np.concatenate([original, np.repeat(coherencia[:, None, :], 4, axis=1),
                        np.repeat(evidencia[:, None, :], 4, axis=1)], axis=2)
    assert x.shape == (133, 4, 2705) and np.isfinite(x[d.suficiente]).all()
    orden = pd.read_csv(AQUI / 'resultados_pd_doble/comparacion_133.csv')
    assert orden.archivo.equals(d.archivo) and orden.sha256_senal.equals(d.sha256_senal)
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert dev.sum() == 52 and not any('@' in q for q in g)
    protocolo = dict(codigo=firma(Path(__file__)), fuente=firma(fuente),
        crudos=firma(AQUI / 'resultados_globales_arbol/rasgos.npz'),
        coherencia=firma(AQUI / 'resultados_coherencia_ciclica/rasgos.npz'),
        evidencia_fuente=firma(ep), extractor_juez=firma(AQUI / 'clasificar_evidencia_seleccion.py'),
        modelo=firma(AQUI / 'modelo_por_dominio.py'), puntuador=firma(AQUI / 'puntuar_banco57.py'),
        receta='LDA shrinkage auto, 2021 crudos +600 ACP +84 evidencias de selección',
        evidencia='tres familias, potencias relativas, significancia, márgenes y persistencia de primera mitad',
        tipos='solo clasificador; no se sobrescriben mediante controles D/PD',
        grado='mismo juez D/PD/ND y dos ramas banco57', detector='mismo detector estricto',
        entrenamiento_fallas=52, exclusiones='tipo+tamaño completos, todas las posiciones OR juntas',
        ajuste='solo etiqueta de tipo en entrenamiento; nunca grado del paper',
        exploratorio=True, validacion_externa=False)
    rp = SALIDA / 'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(SALIDA / 'rasgos_juez84.npz', x=evidencia)
    modelos, pliegues, pred = {}, [], []
    with threadpool_limits(limits=1):
        for i, r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria == 'N':
                t = 'N' if r.suficiente else 'sin_diagnostico'
            else:
                if r.grupo_auditoria not in modelos:
                    tr = dev & (g != r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr, 'archivo'])
                    modelos[r.grupo_auditoria] = crear().fit(x[tr].reshape(-1, 2705), np.repeat(y[tr], 4))
                    pliegues.append(dict(grupo_exterior=r.grupo_auditoria,
                        train=d.loc[tr, 'archivo'].tolist(), grupos_train=sorted(set(g[tr]))))
                    print('Juez aprendido, fuera:', r.grupo_auditoria, flush=True)
                t = predecir_tipo(modelos[r.grupo_auditoria], x[i])
            pred.append(t)
    d['prediccion'] = pred
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA / 'tipos_133.csv', index=False)
    (SALIDA / 'pliegues.json').write_text(json.dumps(pliegues, indent=2), encoding='utf-8')
    evaluar(d, SALIDA)


if __name__ == '__main__':
    main()
