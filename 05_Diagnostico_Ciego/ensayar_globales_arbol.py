"""Ablación fija: árbol armónico más estadísticas y PSD globales existentes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_espectral import extraer
from ventanas_verificadas import prefijo
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from control_ordenes_giro import elegir

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_globales_arbol'


def globales(x, fs, catalogo):
    z = prefijo(x,fs)
    return np.asarray([extraer(z[i:i+12000],12000,catalogo)['globales'] for i in (0,6000,12000,18000)])


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_total_condicion_excluida'
    base = pd.read_csv(fuente/'comparacion_133.csv')
    with np.load(fuente/'rasgos_evaluacion.npz',allow_pickle=False) as z:
        arbol = z['x']
    evidencia = json.loads((AQUI/'resultados_correccion_d_persistente/evidencia.json').read_text(encoding='utf-8'))
    alternativas = {q['archivo']:q['alternativas_seleccion'] for q in evidencia}
    meta = pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    meta = meta[meta.real.ne('N')].copy()
    protocolo = dict(version='arbol_mas_globales_2021_v1',
        representacion='1953 rasgos de árbol existentes + 68 estadísticas y PSD globales del mismo extractor',
        modelo='StandardScaler + LDA lsqr shrinkage auto, parámetros fijos; cuatro ventanas por registro',
        control='mismo detector y control D sin alias de giro',
        entradas='primeros 2.5 s, fs y catálogo completo',
        validacion='52 fallas de desarrollo, condición completa excluida en ambos sensores y muestreos',
        limitacion='experimento sobre datos ya inspeccionados; sin validación externa',
        codigo=firma(Path(__file__)),base=firma(fuente/'comparacion_133.csv'),
        arbol=firma(fuente/'rasgos_evaluacion.npz'),extractor=firma(AQUI/'modelo_espectral.py'),
        control_giro=firma(AQUI/'control_ordenes_giro.py'))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    bloques = []
    for i,r in enumerate(base.itertuples()):
        if r.suficiente:
            fila = idx.loc[r.archivo]; mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            senal = np.ravel(mat[F.clave_canal(mat,r.sensor,int(fila.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            rpm = F.rpm_registro(mat,int(fila.registro),nominal=float(fila.rpm_nominal))
            c = {k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor].items()}
            g = globales(senal,int(r.fs),c)
            assert g.shape == (4,68) and np.isfinite(g).all()
            bloques.append(np.column_stack([arbol[i],g]))
        else:
            bloques.append(np.full((4,2021),np.nan))
        print(f'Globales {i+1}/133',flush=True)
    x = np.asarray(bloques)
    np.savez_compressed(SALIDA/'rasgos.npz',x=x)
    posiciones = base.reset_index().set_index('archivo')['index'].reindex(meta.archivo).to_numpy()
    assert len(posiciones) == 52 and not pd.isna(posiciones).any()
    xd = x[posiciones]; y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    ajustes = {}; folds = []; pred = []; pred_base = []
    with threadpool_limits(limits=1):
        for i,r in enumerate(base.itertuples()):
            if not r.suficiente or r.prediccion == 'N':
                t = r.prediccion; tb = t
            else:
                if r.grupo not in ajustes:
                    tr = grupos != r.grupo
                    assert r.archivo not in set(meta.loc[tr,'archivo'])
                    assert r.grupo not in set(grupos[tr])
                    ft = xd[tr].reshape(-1,2021); fy = np.repeat(y[tr],4)
                    ajustes[r.grupo] = crear().fit(ft,fy)
                    folds.append(dict(grupo_excluido=r.grupo,train=meta.loc[tr,'archivo'].tolist()))
                tb = predecir_tipo(ajustes[r.grupo],x[i])
                t,_ = elegir(tb,alternativas[r.archivo],F.MULTIPLICADORES[r.sensor])
            pred.append(t); pred_base.append(tb)
    d = base.drop(columns=['prediccion','grado','tipo_correcto']).assign(tipo_sin_control=pred_base,prediccion=pred)
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    f = d[d.paper.notna()]; dev = f[f.archivo.isin(meta.archivo)]
    resumen = dict(total=105,tipo_correcto=int(f.tipo_correcto.sum()),tipo_correcto_desarrollo=int(dev.tipo_correcto.sum()),
        total_desarrollo=52,normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        calidad_pendiente=True,desplegado=False)
    assert len(f) == 105 and len(dev) == 52
    (SALIDA/'resumen_tipo.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__ == '__main__':
    main()
