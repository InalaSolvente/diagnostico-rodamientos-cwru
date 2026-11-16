"""Combina tipo OOF anidado y banda estable sin mirar la referencia al predecir."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from bandas_estables import seleccionar, verificar
from criterio_usuario import ORDEN
from modelo_ciego import F

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_candidato_causal_desarrollo'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compacto', action='store_true')
    parser.add_argument('--lda-completo', action='store_true')
    args = parser.parse_args()
    if args.compacto and args.lda_completo:
        parser.error('Elija una sola variante')
    if args.compacto:
        SALIDA = AQUI/'resultados_candidato_compacto_desarrollo'
    if args.lda_completo:
        SALIDA = AQUI/'resultados_candidato_lda_completo_desarrollo'
    SALIDA.mkdir(exist_ok=True)
    ruta_tipos = AQUI/('resultados_tipo_compacto_desarrollo' if args.compacto else 'resultados_tipo_anidado_desarrollo')/'predicciones_oof.csv'
    if args.lda_completo:
        ruta_tipos = AQUI/'resultados_tipo_lda_completo_desarrollo/predicciones_oof.csv'
    tipos = pd.read_csv(ruta_tipos).set_index('archivo')
    split = pd.read_csv(AQUI/'particion_desarrollo.csv')
    dev = set(split.loc[split.particion.eq('desarrollo'), 'archivo'])
    assert len(tipos) == 54 and set(tipos.index) == dev
    protocolo = dict(nombre='tipo_compacto_prefijo_mas_banda_estable_v1' if args.compacto else 'tipo_anidado_prefijo_mas_banda_estable_v1',
        tipo='OOF anidado usando exclusivamente primeros 2 s, fs/duración igualadas',
        banda='selección en primera mitad, misma banda/familia/transformación en segunda',
        tipos_sha256=hashlib.sha256(ruta_tipos.read_bytes()).hexdigest(),
        bandas_sha256=hashlib.sha256((AQUI/'bandas_estables.py').read_bytes()).hexdigest(),
        prohibiciones='sin referencia ni tipo real al predecir; sin elección por resultados de reserva')
    if args.lda_completo:
        protocolo['nombre'] = 'tipo_lda_completo_prefijo_mas_banda_estable_v1'
        protocolo['tipo'] = 'OOF con LDA completo fijo, primeros 2 s; sin selección de modelos'
    ruta_protocolo = SALIDA/'protocolo.json'
    if ruta_protocolo.exists():
        assert json.loads(ruta_protocolo.read_text(encoding='utf-8')) == protocolo
    else:
        ruta_protocolo.write_text(json.dumps(protocolo, indent=2, ensure_ascii=False), encoding='utf-8')
    antiguo = AQUI/'resultados_bandas_estables_desarrollo'
    config = json.loads((antiguo/'protocolo.json').read_text(encoding='utf-8'))
    assert config['codigo_sha256'] == protocolo['bandas_sha256']
    cache = {r['archivo']: r for r in json.loads((antiguo/'predicciones.json').read_text(encoding='utf-8'))}
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo', drop=False)
    filas = []
    for i, archivo in enumerate(sorted(dev)):
        r = idx.loc[archivo]
        mat = loadmat(AQUI.parent/'data2/CWRU'/archivo)
        clave = next(k for k in mat if k.endswith(f'_{r.sensor_falla}_time'))
        x = np.ravel(mat[clave]); mitad = len(x)//2
        krpm = [k for k in mat if k.endswith('RPM')]
        rpm = float(np.ravel(mat[krpm[0]])[0]) if krpm else float(r.rpm_nominal)
        catalogo = {k: float(v*rpm/60) for k, v in F.MULTIPLICADORES[r.sensor_falla].items()}
        tipo = tipos.loc[archivo, 'prediccion']
        huella = hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        guardado = cache.get(archivo)
        if guardado and guardado['tipo_predicho'] == tipo:
            assert guardado['sha256_senal'] == huella
            q, ver = guardado['seleccion'], guardado['verificacion']
        else:
            q = seleccionar(x[:mitad], float(r.fs), catalogo, tipo)
            ver = verificar(x[mitad:], float(r.fs), catalogo, q)
        filas.append(dict(archivo=archivo, tipo_predicho=tipo, grado=ver['grado'],
            seleccion=q, verificacion=ver, sha256_senal=huella))
        (SALIDA/'predicciones.json').write_text(json.dumps(filas, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'{i+1}/54 {archivo}: {tipo} {ver["grado"]}', flush=True)
    # Todas las predicciones ya están emitidas. Ahora se evalúan con las etiquetas.
    base = pd.read_csv(AQUI/'auditoria_modelos_actuales/persistencia_temporal_sin_copias.csv').set_index('archivo')
    d = pd.DataFrame([{k: r[k] for k in ('archivo','tipo_predicho','grado')} for r in filas]).set_index('archivo')
    d['tipo_real'] = idx.loc[d.index, 'zona']
    d['tipo_correcto'] = d.tipo_predicho.eq(d.tipo_real)
    d['paper'] = base.paper.reindex(d.index)
    d.to_csv(SALIDA/'comparacion.csv')
    fallas = d[d.paper.notna()]
    delta = fallas.grado.map(ORDEN)-fallas.paper.map(ORDEN)
    ok = fallas.tipo_correcto
    resumen = dict(total=len(fallas), mejoran=int((ok & delta.gt(0)).sum()),
        iguales=int((ok & delta.eq(0)).sum()), empeoran=int((ok & delta.lt(0)).sum()),
        tipo_incorrecto=int((~ok).sum()), sin_retroceso=int((ok & delta.ge(0)).sum()))
    assert len(fallas) == 52 and sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 52
    normales = d[d.tipo_real.eq('N')]
    resumen.update(normales=len(normales), normales_correctos=int(normales.tipo_correcto.sum()),
                   alarmas_con_evidencia_en_normales=int((normales.tipo_predicho.ne('N') & normales.grado.ne('ND')).sum()))
    (SALIDA/'resumen.json').write_text(json.dumps(resumen, indent=2), encoding='utf-8')
    print(json.dumps(resumen), flush=True)
