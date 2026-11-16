"""Ablación aditiva: LDA y controles históricos, añade ACP600 a los 2021 rasgos."""
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
SALIDA = AQUI / 'resultados_crudo_coherencia'


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_grupos_por_defecto/tipos_133.csv'
    d = pd.read_csv(fuente)
    cs = AQUI / 'resultados_coherencia_ciclica'
    cm = pd.read_csv(cs / 'muestra.csv')
    assert cm.archivo.equals(d.archivo) and cm.sha256_senal.equals(d.sha256_senal)
    assert cm.sensor.equals(d.sensor) and cm.fs.equals(d.fs)
    d['catalogo'], d['registro'] = cm.catalogo, cm.registro
    with np.load(AQUI / 'resultados_globales_arbol/rasgos.npz', allow_pickle=False) as z:
        original = z['x']
    with np.load(cs / 'rasgos.npz', allow_pickle=False) as z:
        coherencia = z['x']
    orden = pd.read_csv(AQUI / 'resultados_pd_doble/comparacion_133.csv')
    assert orden.archivo.equals(d.archivo) and orden.sha256_senal.equals(d.sha256_senal)
    assert original.shape == (133, 4, 2021) and coherencia.shape == (133, 600)
    x = np.concatenate([original, np.repeat(coherencia[:, None, :], 4, axis=1)], axis=2)
    assert x.shape == (133, 4, 2621) and np.isfinite(x[d.suficiente]).all()
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert dev.sum() == 52 and not any('@' in q for q in g)
    ep = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(ep.read_text(encoding='utf-8'))}
    protocolo = dict(codigo=firma(Path(__file__)), fuente=firma(fuente),
        rasgos_crudos=firma(AQUI / 'resultados_globales_arbol/rasgos.npz'),
        rasgos_coherencia=firma(cs / 'rasgos.npz'), preparador=firma(AQUI / 'coherencia_ciclica.py'),
        modelo=firma(AQUI / 'modelo_por_dominio.py'), controlador=firma(AQUI / 'evaluar_retirada_control_pd.py'),
        puntuador=firma(AQUI / 'puntuar_banco57.py'), alternativas=firma(ep),
        receta='LDA shrinkage auto sobre 2021+600, cuatro ventanas por registro; ACP común del prefijo 2.5 s',
        controles='D sin alias de giro y PD maxmin, sin cambios',
        entrenamiento_fallas=52, exclusiones='tipo+tamaño completos, todas las posiciones OR juntas',
        detector='mismo detector estricto', grado='banco57, mismo juez y dos ramas fijadas antes de verificar',
        seleccion_de_configuracion='receta única fijada antes de abrir grados futuros',
        exploratorio=True, validacion_externa=False)
    rp = SALIDA / 'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    modelos, pliegues, pred, tipos_base = {}, [], [], []
    with threadpool_limits(limits=1):
        for i, r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria == 'N':
                tb = t = 'N' if r.suficiente else 'sin_diagnostico'
            else:
                if r.grupo_auditoria not in modelos:
                    tr = dev & (g != r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr, 'archivo'])
                    modelos[r.grupo_auditoria] = crear().fit(x[tr].reshape(-1, 2621), np.repeat(y[tr], 4))
                    pliegues.append(dict(grupo_exterior=r.grupo_auditoria,
                        train=d.loc[tr, 'archivo'].tolist(), grupos_train=sorted(set(g[tr]))))
                    print('Crudo+ACP, fuera:', r.grupo_auditoria, flush=True)
                tb = predecir_tipo(modelos[r.grupo_auditoria], x[i])
                t = controlar(tb, r.sensor, alternativas[r.archivo], usar_d=True, usar_pd=True)
            pred.append(t)
            tipos_base.append(tb)
    d['prediccion'], d['tipo_base_crudo_acp'] = pred, tipos_base
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA / 'tipos_133.csv', index=False)
    (SALIDA / 'pliegues.json').write_text(json.dumps(pliegues, indent=2), encoding='utf-8')
    evaluar(d, SALIDA)


if __name__ == '__main__':
    main()
