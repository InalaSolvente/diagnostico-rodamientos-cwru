"""Selección anidada de clasificador y control; nunca consulta etiquetas exteriores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from evaluar_retirada_control_pd import controlar
from puntuar_tipos_fijos import evaluar

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_selector_controles_anidado'
CONTROLES = [('D_PD', True, True), ('D', True, False), ('ninguno', False, False)]


def modelos():
    return dict(lda=crear,
        svm1=lambda: make_pipeline(StandardScaler(), SVC(C=1., gamma='scale', class_weight='balanced')),
        svm10=lambda: make_pipeline(StandardScaler(), SVC(C=10., gamma='scale', class_weight='balanced')))


def elegir(predicciones, etiquetas):
    """Las únicas etiquetas permitidas son las de validación interna."""
    assert set(etiquetas) == {'B', 'IR', 'OR'}
    scores = {k: dict(equilibrada=float(np.mean([np.mean(p[etiquetas == c] == c)
                    for c in ('B', 'IR', 'OR')])), exactitud=float(np.mean(p == etiquetas)))
              for k, p in predicciones.items()}
    mejor = max(scores, key=lambda k: (scores[k]['equilibrada'], scores[k]['exactitud']))
    return mejor, scores


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_grupos_por_defecto/tipos_133.csv'
    d = pd.read_csv(fuente)
    catalogos = AQUI / 'resultados_plantillas_coherentes/tipos_133.csv'
    c = pd.read_csv(catalogos).set_index('archivo').loc[d.archivo]
    assert c.sha256_senal.tolist() == d.sha256_senal.tolist()
    assert c.fs.tolist() == d.fs.tolist() and c.sensor.tolist() == d.sensor.tolist()
    d['catalogo'] = c.catalogo.tolist()
    muestra = AQUI / 'resultados_dominios_desarrollo/muestra.csv'
    meta = pd.read_csv(muestra)
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    assert int(dev.sum()) == 52
    rasgos = AQUI / 'resultados_globales_arbol/rasgos.npz'
    with np.load(rasgos, allow_pickle=False) as z:
        x = z['x']
    origen_rasgos = pd.read_csv(AQUI / 'resultados_pd_doble/comparacion_133.csv')
    assert origen_rasgos.archivo.equals(d.archivo)
    assert origen_rasgos.sha256_senal.equals(d.sha256_senal)
    assert x.shape == (133, 4, 2021), x.shape
    suficientes = d.suficiente.to_numpy()
    assert np.isfinite(x[suficientes]).all() and np.all(suficientes[dev])
    ep = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(ep.read_text(encoding='utf-8'))}
    g = d.grupo_auditoria.to_numpy()
    y = d.real.to_numpy()
    assert not any('@' in q for q in g)
    protocolo = dict(codigo=firma(Path(__file__)), fuente=firma(fuente), catalogos=firma(catalogos),
        muestra=firma(muestra), rasgos=firma(rasgos), evidencia_seleccion=firma(ep),
        controlador=firma(AQUI / 'evaluar_retirada_control_pd.py'),
        puntuador=firma(AQUI / 'puntuar_tipos_fijos.py'),
        candidatos='LDA shrinkage, SVM RBF C=1 o C=10; cada uno sin control, D o D+PD',
        criterio_interno='exactitud equilibrada de tipo; desempate por exactitud y orden fijo',
        particion='exterior e interior: tipo+tamaño completos, todas las posiciones OR juntas',
        detector='mismo detector de grupos completos ya auditado',
        entrenamiento_fallas=52, verificacion_para_elegir=False,
        limitacion='exploracion reiterada del mismo dataset; no validacion externa')
    rp = SALIDA / 'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    cache = {}
    auditoria = []
    pred = np.full(len(d), '', dtype=object)
    base = np.full(len(d), '', dtype=object)
    configuracion = np.full(len(d), '', dtype=object)

    def predecir(excluidos, nombre, indices):
        key = (tuple(sorted(excluidos)), nombre)
        tr = dev & ~np.isin(g, list(excluidos))
        assert set(g[tr]).isdisjoint(excluidos)
        assert not np.any(tr[indices]) and set(y[tr]) == {'B', 'IR', 'OR'}
        if key not in cache:
            clf = modelos()[nombre]().fit(x[tr].reshape(-1, 2021), np.repeat(y[tr], 4))
            # Guarda solo predicciones; cada ajuste pertenece a un conjunto exacto de grupos.
            cache[key] = np.full(len(d), 'sin_diagnostico', dtype=object)
            for i in np.flatnonzero(suficientes):
                cache[key][i] = predecir_tipo(clf, x[i])
        return cache[key][indices]

    def aplicar(bases, indices, usar_d, usar_pd):
        return np.array([controlar(t, d.iloc[i].sensor, alternativas[d.iloc[i].archivo],
                         usar_d=usar_d, usar_pd=usar_pd) for t, i in zip(bases, indices)], dtype=object)

    with threadpool_limits(limits=1):
        for exterior in sorted(set(g)):
            te = np.flatnonzero(g == exterior)
            entrenamiento = np.flatnonzero(dev & (g != exterior))
            candidatos = {f'{m}:{control}': np.full(len(d), '', dtype=object)
                          for m in modelos() for control, _, _ in CONTROLES}
            trazas = []
            for interior in sorted(set(g[entrenamiento])):
                va = entrenamiento[g[entrenamiento] == interior]
                tr = dev & ~np.isin(g, [exterior, interior])
                assert set(d.loc[tr, 'archivo']).isdisjoint(d.loc[te, 'archivo'])
                assert set(d.loc[tr, 'archivo']).isdisjoint(d.loc[va, 'archivo'])
                for nombre in modelos():
                    p = predecir({exterior, interior}, nombre, va)
                    for control, usar_d, usar_pd in CONTROLES:
                        candidatos[f'{nombre}:{control}'][va] = aplicar(p, va, usar_d, usar_pd)
                trazas.append(dict(grupo_interior=interior, train=d.loc[tr, 'archivo'].tolist(),
                                   validacion=d.loc[va, 'archivo'].tolist()))
            reducidos = {k: p[entrenamiento] for k, p in candidatos.items()}
            assert all(np.all(p != '') for p in reducidos.values())
            mejor, scores = elegir(reducidos, y[entrenamiento])
            # Mutar las etiquetas exteriores no puede cambiar esta decisión.
            yy = y.copy()
            yy[te] = 'etiqueta_exterior_inaccesible'
            assert elegir(reducidos, yy[entrenamiento]) == (mejor, scores)
            nombre, control = mejor.split(':')
            _, usar_d, usar_pd = next(q for q in CONTROLES if q[0] == control)
            utilizables = te[d.loc[te, 'suficiente'].to_numpy() & d.loc[te, 'detector_auditoria'].ne('N').to_numpy()]
            if len(utilizables):
                p = predecir({exterior}, nombre, utilizables)
                base[utilizables] = p
                pred[utilizables] = aplicar(p, utilizables, usar_d, usar_pd)
            for i in te:
                if not d.iloc[i].suficiente:
                    pred[i] = base[i] = 'sin_diagnostico'
                elif d.iloc[i].detector_auditoria == 'N':
                    pred[i] = base[i] = 'N'
                configuracion[i] = mejor
            auditoria.append(dict(grupo_exterior=exterior, train=d.loc[entrenamiento, 'archivo'].tolist(),
                test=d.loc[te, 'archivo'].tolist(), configuracion=mejor, scores=scores, internos=trazas))
            print('Selector anidado:', exterior, mejor, scores[mejor], flush=True)
    assert np.all(pred != '')
    d['prediccion'] = pred
    d['tipo_base_selector'] = base
    d['configuracion_selector'] = configuracion
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA / 'tipos_133.csv', index=False)
    (SALIDA / 'auditoria_anidada.json').write_text(json.dumps(auditoria, indent=2), encoding='utf-8')
    evaluar(d, SALIDA)


if __name__ == '__main__':
    main()
