"""Agrega al tipo evidencia de envolvente cuadrada, sin cambiar el juez.

Se fija una variante antes de ejecutarla: arbol original y 180 agregados nuevos.
Solo usa el prefijo de seleccion; nunca el segmento reservado a verificacion.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_espectral import evidencia_armonica, FAMILIAS
from ventanas_verificadas import prefijo
from bandas_estables import envolvente, espectro
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo

AQUI = Path(__file__).parent
FUENTE = AQUI/'resultados_dominios_desarrollo'
SALIDA = AQUI/'resultados_tipo_envolvente_cuadrada'


def rasgos_cuadrados(x, fs, catalogo):
    z = prefijo(x, fs)
    frecuencias = [2*catalogo['BSF'] if k == '2BSF' else catalogo[k] for k in FAMILIAS]
    salida = []
    for inicio in (0, 6000, 12000, 18000):
        ventana = z[inicio:inicio+12000]
        valores = []
        for l in range(5):
            for m in range(2**l):
                a = envolvente(ventana, 12000, (l, m))
                f, s = espectro(a*a, 12000)
                valores.append(evidencia_armonica(f, s, frecuencias))
        v = np.asarray(valores)
        salida.append(np.r_[v.max(axis=0), np.median(v, axis=0), v.mean(axis=0)])
    resultado = np.asarray(salida)
    assert resultado.shape == (4, 180) and np.isfinite(resultado).all()
    return resultado


def comprobar_extractor():
    fs = 12000; t = np.arange(6*fs)/fs
    c = dict(BPFI=100., BPFO=137., BSF=83., FTF=17.)
    x = (1+.4*np.cos(2*np.pi*100*t)+.2*np.cos(2*np.pi*200*t))*np.cos(2*np.pi*2500*t)
    a = rasgos_cuadrados(x, fs, c)
    # Cambiar toda la segunda mitad no puede afectar los rasgos del tipo.
    otra = x.copy(); otra[len(x)//2:] = np.random.default_rng(52).normal(size=len(x)//2)
    assert np.array_equal(a, rasgos_cuadrados(otra, fs, c))
    assert np.allclose(a, rasgos_cuadrados(3*x, fs, c), atol=1e-8)


def main():
    comprobar_extractor()
    SALIDA.mkdir(exist_ok=True)
    meta = pd.read_csv(FUENTE/'muestra.csv')
    with np.load(FUENTE/'rasgos.npz', allow_pickle=False) as z:
        base, origen = z['x'], z['origen']
    assert base.shape == (224, 1953) and meta.archivo.nunique() == 54
    dependencias = ('modelo_espectral.py', 'ventanas_verificadas.py', 'bandas_estables.py',
        'modelo_por_dominio.py', 'ensayar_tipo_por_sensor.py')
    protocolo = dict(version='tipo_cuadrada_agregados_v1',
        hipotesis='incluir en el tipo la envolvente cuadrada ya disponible en el selector',
        representacion='arbol original + maximo, mediana y media de 31 bandas de envolvente cuadrada',
        modelo='LDA lsqr shrinkage auto; cuatro ventanas por registro; solo B/IR/OR',
        detector='OOF historico sin modificar',
        exterior='condicion completa fuera en ambos sensores y todas las ventanas',
        seleccion='una variante fija; sin ajuste de umbrales ni eleccion con el total',
        limitacion='desarrollo reutilizado; evidencia exploratoria, sin validacion externa',
        codigo=firma(Path(__file__)), fuente=firma(FUENTE/'rasgos.npz'),
        muestra=firma(FUENTE/'muestra.csv'),
        dependencias={n:firma(AQUI/n) for n in dependencias})
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    bloques = []
    for i, r in enumerate(meta.itertuples()):
        mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
        senal = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
        import hashlib
        assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        bloques.append(rasgos_cuadrados(senal, int(r.fs_original), json.loads(r.catalogo)))
        print(f'Envolvente cuadrada {i+1}/56', flush=True)
    x = np.column_stack([base, np.concatenate(bloques)])
    assert x.shape == (224, 2133)
    np.savez_compressed(SALIDA/'rasgos.npz', x=x, origen=origen)
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy()
    gate = pd.read_csv(AQUI/'resultados_por_dominio/tipos_oof.csv').set_index('vista_uid')
    gate = gate.prediccion.reindex(meta.vista_uid).to_numpy()
    assert not pd.isna(gate).any()
    pred = np.full(len(meta), '', dtype=object); folds = []
    with threadpool_limits(limits=1):
        for grupo in sorted(set(grupos)):
            train = (grupos != grupo) & (y != 'N'); test = np.flatnonzero(grupos == grupo)
            assert set(grupos[train]).isdisjoint(grupos[test])
            assert set(meta.loc[train, 'archivo']).isdisjoint(meta.loc[test, 'archivo'])
            ft = train[origen]
            clf = crear().fit(x[ft], y[origen[ft]])
            for i in test:
                pred[i] = 'N' if gate[i] == 'N' else predecir_tipo(clf, x[origen == i])
            folds.append(dict(grupo=grupo, train=meta.loc[train, 'vista_uid'].tolist(),
                test=meta.loc[test, 'vista_uid'].tolist()))
            print('OOF cuadrada:', grupo, flush=True)
    assert all(pred != '')
    d = meta[['archivo', 'vista_uid', 'sensor', 'real', 'grupo']].assign(prediccion=pred)
    d.to_csv(SALIDA/'tipos_oof.csv', index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    f = d[d.real.ne('N')]
    resumen = dict(total=len(f), tipo_correcto=int(f.real.eq(f.prediccion).sum()),
        tipo_incorrecto=int(f.real.ne(f.prediccion).sum()),
        normales_vistas_correctas=int(d.loc[d.real.eq('N')].prediccion.eq('N').sum()),
        evaluacion='solo tipo; falta evaluar grados con el juez fijo', desplegado=False)
    assert len(f) == 52
    (SALIDA/'resumen_tipo.json').write_text(json.dumps(resumen, indent=2), encoding='utf-8')
    print(json.dumps(resumen), flush=True)


if __name__ == '__main__':
    main()
