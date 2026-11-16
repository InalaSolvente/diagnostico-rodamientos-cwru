"""Control necesario: LDA original, con los mismos prefijos causales y datos igualados.

Receta única, sin seleccionar hiperparámetros ni mirar la referencia del paper.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_tipo_lda_completo_desarrollo'


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_tipo_seleccion_desarrollo'
    meta = pd.read_csv(fuente/'predicciones_oof.csv')
    with np.load(fuente/'rasgos.npz', allow_pickle=False) as datos:
        x = datos['x']
    protocolo = dict(version='lda_completo_prefijo_v1', clasificador='LDA lsqr shrinkage auto, arbol 1953 rasgos',
        entrada='prefijo 2 s, misma fs/duración y bandas físicas; sin tramo verificador',
        seleccion='receta única; condición completa fuera; cada normal único fuera individualmente',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rasgos_sha256=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    ruta = SALIDA/'protocolo.json'
    if ruta.exists():
        assert json.loads(ruta.read_text(encoding='utf-8')) == protocolo
    else:
        ruta.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    pred = np.empty(len(y), dtype=object); auditoria = []
    # Limita la sobreasignación de hilos BLAS; no cambia la receta estadística.
    with threadpool_limits(limits=1):
        for grupo in sorted(set(grupos)):
            test = grupos == grupo; train = ~test
            assert grupo not in set(grupos[train]) and set(y[train]) == set(y)
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
            clf.fit(x[train], y[train]); pred[test] = clf.predict(x[test])
            auditoria.append(dict(grupo_exterior=grupo, train=meta.loc[train,'archivo'].tolist(),
                test=meta.loc[test,'archivo'].tolist(), validaciones_internas=[], modelo_elegido='lda_completo_fijo'))
            print(f'LDA completo: {grupo[:20]}', flush=True)
    d = meta.drop(columns=['prediccion','acierto']).assign(prediccion=pred, modelo='lda_completo_fijo', acierto=pred == y)
    d.to_csv(SALIDA/'predicciones_oof.csv', index=False)
    (SALIDA/'auditoria_pliegues.json').write_text(json.dumps(auditoria, indent=2), encoding='utf-8')
    print(d.groupby(d.real.eq('N').map({True:'normales',False:'fallas'})).acierto.agg(['sum','count']).to_string())
