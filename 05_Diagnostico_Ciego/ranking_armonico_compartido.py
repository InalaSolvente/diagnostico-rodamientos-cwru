"""Ranker compartido entre familias: evidencia continua y armónicas exclusivas.

No recibe el tipo esperado ni parámetros específicos para una familia. Excluye
la condición completa al ajustar y conserva detector, filtro y juez históricos.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import firma

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_ranking_armonico_compartido'
TIPOS = np.array(['IR', 'OR', 'B'])


def representacion(arbol, catalogo):
    """Devuelve una misma descripción para cada candidato, sin su identidad."""
    arbol = np.asarray(arbol, dtype=float)
    if arbol.ndim != 2 or arbol.shape[1] != 1953 or not np.isfinite(arbol).all():
        raise ValueError('Se requieren rasgos de selección arbol L4 finitos')
    f = np.array([catalogo['BPFI'], catalogo['BPFO'], catalogo['BSF'], 2*catalogo['BSF']])
    if not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError('Catálogo completo positivo requerido')
    pertenencia = (0, 1, 2, 2)
    objetivos = f[:, None]*np.arange(1, 7)
    # Fracción de la ventana teórica que no coincide con otra clase.
    # BSF y 2BSF son representaciones del mismo tipo y no compiten entre sí.
    exclusividad = np.ones((4, 6))
    for i in range(4):
        otros = objetivos[[j for j in range(4) if pertenencia[j] != pertenencia[i]]].ravel()
        for h, centro in enumerate(objetivos[i]):
            a, b = .98*centro, 1.02*centro
            intervalos = sorted((max(a, .98*v), min(b, 1.02*v)) for v in otros
                                if min(b, 1.02*v) > max(a, .98*v))
            cubierto = 0.; final = a
            for lo, hi in intervalos:
                cubierto += max(0., hi-max(lo, final)); final = max(final, hi)
            exclusividad[i,h] = max(0., 1-cubierto/(b-a))
    candidatos = []
    for ventana in arbol:
        armonicos = ventana[:1860].reshape(31,5,6,2)
        estados = ventana[1860:].reshape(31,3)
        fila = []
        for indices in ((0,), (1,), (2,3)):
            opciones = []
            for i in indices:
                evidencia = armonicos[:,i]
                exclusiva = np.maximum(evidencia[:,:,0],0)*exclusividad[i]
                # Dos armónicas de la misma banda, sin exigir la primera:
                # PD puede apoyarse en una armónica superior.
                soporte = np.sort(exclusiva,axis=1)[:,-2:].mean(axis=1)
                opciones.append((float(soporte.max()), i, soporte, exclusiva))
            _, i, soporte, exclusiva = max(opciones, key=lambda q:q[0])
            v = np.column_stack([armonicos[:,i].reshape(31,12), exclusiva])
            mejores = np.argsort(-soporte,kind='stable')[:3]
            fila.append(np.r_[v[mejores].ravel(), v.max(axis=0), np.median(v,axis=0),
                v.mean(axis=0), estados[mejores,1:].ravel()])
        candidatos.append(fila)
    z = np.asarray(candidatos)
    # La misma resta simétrica para cada candidato; ninguna columna de tipo.
    relativos = np.asarray([z[:,i]-np.max(z[:,[j for j in range(3) if j != i]],axis=1)
                            for i in range(3)]).transpose(1,0,2)
    salida = np.concatenate([z, relativos],axis=2)
    assert salida.shape == (len(arbol),3,228) and np.isfinite(salida).all()
    return salida


def ajustar(x, y):
    diferencias = []
    for caso, real in zip(x,y):
        i = int(np.flatnonzero(TIPOS == real)[0])
        diferencias.extend(caso[i]-caso[j] for j in range(3) if j != i)
    d = np.asarray(diferencias)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.,fit_intercept=False,max_iter=2000))
    clf.fit(np.r_[d,-d],np.r_[np.ones(len(d)),np.zeros(len(d))])
    return clf


def puntuar(clf, casos):
    return clf.decision_function(casos.reshape(-1,casos.shape[-1])).reshape(-1,3)


def comprobar():
    rng = np.random.default_rng(914)
    x = rng.normal(size=(4,1953)); c = dict(BPFI=162.,BPFO=107.,BSF=71.)
    z = representacion(x,c)
    permutado = x.copy()
    h = permutado[:,:1860].reshape(4,31,5,6,2)
    h[:,:,0],h[:,:,1] = h[:,:,1].copy(),h[:,:,0].copy()
    otra = dict(c,BPFI=c['BPFO'],BPFO=c['BPFI'])
    assert np.allclose(z[:,[1,0,2]],representacion(permutado,otra))
    assert np.allclose(z,representacion(x,{k:v*10 for k,v in c.items()}))
    assert np.isfinite(representacion(np.zeros((1,1953)),c)).all()


def main():
    comprobar(); SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_dominios_desarrollo'
    meta = pd.read_csv(fuente/'muestra.csv')
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:
        arbol, origen = z['x'], z['origen']
    x = np.asarray([representacion(arbol[origen == i],json.loads(r.catalogo)).mean(axis=0)
                    for i,r in enumerate(meta.itertuples())])
    assert x.shape == (56,3,228)
    protocolo = dict(version='ranking_armonico_compartido_v1',
        hipotesis='aprender soporte armónico compartido, descontando coincidencias entre clases',
        modelo='ranker logístico por diferencias, C=1 fijo; sin identidad del candidato',
        entrada='árbol de selección y catálogo completo; no frecuencia verdadera ni grado del paper',
        seleccion='una variante fija; sin búsqueda de hiperparámetros',
        validacion='condición completa excluida, entrenamiento solo desarrollo histórico',
        limitacion='datos y desarrollo ya inspeccionados; no validación externa',
        codigo=firma(Path(__file__)), fuente=firma(fuente/'rasgos.npz'), muestra=firma(fuente/'muestra.csv'),
        evaluacion=firma(AQUI/'resultados_total_condicion_excluida/comparacion_133.csv'))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(SALIDA/'rasgos_desarrollo.npz',x=x)
    datos = AQUI/'resultados_total_condicion_excluida'
    d = pd.read_csv(datos/'comparacion_133.csv')
    with np.load(datos/'rasgos_evaluacion.npz',allow_pickle=False) as z:
        todos = z['x']
    assert todos.shape == (133,4,1953)
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy(); modelos = {}; folds = []
    pred = []; scores = []
    with threadpool_limits(limits=1):
        for i,r in enumerate(d.itertuples()):
            if not r.suficiente:
                pred.append('sin_diagnostico'); scores.append([None]*3); continue
            if r.grupo not in modelos:
                train = (grupos != r.grupo) & (y != 'N')
                assert r.grupo not in set(grupos[train])
                assert r.archivo not in set(meta.loc[train,'archivo'])
                modelos[r.grupo] = ajustar(x[train],y[train])
                folds.append(dict(grupo_excluido=r.grupo,train=meta.loc[train,'vista_uid'].tolist()))
            # Solo las proporciones del catálogo afectan la exclusividad;
            # las posiciones espectrales ya usaron el RPM medido al extraer.
            z = representacion(todos[i],F.MULTIPLICADORES[r.sensor]).mean(axis=0)
            s = puntuar(modelos[r.grupo],z[None])[0]
            pred.append('N' if r.detector == 'N' else str(TIPOS[s.argmax()]))
            scores.append(s.tolist())
    salida = d.drop(columns=['prediccion','grado','tipo_correcto']).assign(prediccion=pred)
    salida['tipo_correcto'] = salida.real.eq(salida.prediccion)
    salida.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    (SALIDA/'scores.json').write_text(json.dumps(dict(tipos=TIPOS.tolist(),scores=scores)),encoding='utf-8')
    referencia = salida[salida.paper.notna()]
    dev = referencia[referencia.archivo.isin(meta.archivo)]
    resumen = dict(tipo_correcto_105=int(referencia.tipo_correcto.sum()),total_referencia=105,
        tipo_correcto_52=int(dev.tipo_correcto.sum()),desarrollo=52,
        normales_correctos=int(salida.loc[salida.real.eq('N'),'tipo_correcto'].sum()),
        calidad_pendiente=True,desplegado=False)
    assert len(referencia) == 105 and len(dev) == 52
    (SALIDA/'resumen_tipo.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__ == '__main__':
    main()
