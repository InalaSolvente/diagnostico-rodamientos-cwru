"""Clasifica por evidencia de las tres familias, solo en la primera mitad.

Un LDA fijo aprende de indicadores continuos del mismo juez. No recibe los
resultados de verificacion, la referencia del paper ni la frecuencia verdadera.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import crear, firma
from bandas_estables import margenes, verificar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_evidencia_seleccion'
TIPOS = ('IR','OR','B')


def extraer(alternativas):
    """Ninguna entrada temporal posterior a la seleccion forma parte de esta API."""
    if set(alternativas) != set(TIPOS):
        raise ValueError('Las tres hipotesis de seleccion son obligatorias')
    salida = []
    eps = np.finfo(float).tiny
    for tipo in TIPOS:
        q = alternativas[tipo]; e = q['evidencia']; h = {v['h']:v for v in e['armonicas']}
        escala = max(e['max_izquierda'], e['potencia_primera'], eps)
        for i in range(1,7):
            a = h.get(i)
            if a is None:
                salida.extend([-12.,-12.,0.]); continue
            snr = a['potencia']/max(a['fondo']*a['umbral'],eps)
            relativa = a['potencia']/escala
            salida.extend([float(np.clip(np.log10(max(snr,eps)),-12,12)),
                float(np.clip(np.log10(max(relativa,eps)),-12,12)), float(a['significativa'])])
        salida.extend(q['margen_d_segmentos'])
        salida.extend(q['margen_p_segmentos'])
        salida.extend(margenes(e))
        salida.extend([float(ORDEN[q['grado']]),float(len(e['identificadas']))])
    z = np.asarray(salida)
    assert z.shape == (84,) and np.isfinite(z).all()
    return z


def comprobar(alternativas):
    import copy
    a = extraer(alternativas)
    b = copy.deepcopy(alternativas)
    # Metadatos ajenos a la evidencia de seleccion no pueden influir.
    for q in b.values():
        q['archivo'] = 'otra_identidad'; q['tipo_esperado'] = 'otro'
        q['verificacion'] = dict(grado='D',etiqueta='alterada')
        q['frecuencia_base'] = 9999.
    assert np.array_equal(a,extraer(b))


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_correccion_d_persistente'
    base_dir = AQUI/'resultados_total_condicion_excluida'
    base = pd.read_csv(base_dir/'comparacion_133.csv')
    evidencia = json.loads((fuente/'evidencia.json').read_text(encoding='utf-8'))
    assert [q['archivo'] for q in evidencia] == base.archivo.tolist()
    por_archivo = {q['archivo']:q for q in evidencia}
    primero = next(q['alternativas_seleccion'] for q in evidencia if q['alternativas_seleccion'])
    comprobar(primero)
    protocolo = dict(version='seleccion_84_lda_fijo_v1',
        hipotesis='clasificar con calidad de todas las familias durante seleccion, en lugar de corregir por un D aislado',
        modelo='StandardScaler + LDA lsqr shrinkage auto, parametros fijos',
        rasgos='84 medidas: SNR, potencia relativa, significancia, estabilidad y grado en seleccion para IR/OR/B',
        entradas='solo evidencia de primera mitad, sin resultados de verificacion ni identidad del archivo',
        entrenamiento='52 fallas de desarrollo; condicion completa excluida en DE/FE y todos los muestreos',
        detector='se conserva la decision del detector con condicion excluida',
        limitacion='datos ya inspeccionados; ensayo exploratorio, no validacion externa',
        codigo=firma(Path(__file__)),evidencia=firma(fuente/'evidencia.json'),
        base=firma(base_dir/'comparacion_133.csv'),bandas=firma(AQUI/'bandas_estables.py'),
        criterio=firma(AQUI/'criterio_usuario.py'))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    meta = pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    meta = meta[meta.real.ne('N')].copy()
    assert len(meta) == 52 and meta.archivo.is_unique
    x = np.asarray([extraer(por_archivo[r.archivo]['alternativas_seleccion']) for r in meta.itertuples()])
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy(); ajustes = {}; folds = []
    pred = []
    with threadpool_limits(limits=1):
        for r in base.itertuples():
            if not r.suficiente or r.prediccion == 'N':
                pred.append(r.prediccion); continue
            if r.grupo not in ajustes:
                train = grupos != r.grupo
                assert r.grupo not in set(grupos[train])
                assert r.archivo not in set(meta.loc[train,'archivo'])
                ajustes[r.grupo] = crear().fit(x[train],y[train])
                folds.append(dict(grupo_excluido=r.grupo,train=meta.loc[train,'archivo'].tolist()))
            z = extraer(por_archivo[r.archivo]['alternativas_seleccion'])
            pred.append(str(ajustes[r.grupo].predict(z[None])[0]))
    # Congela todas las decisiones antes de consultar la verificacion.
    d = base.drop(columns=['prediccion','grado','tipo_correcto']).assign(prediccion=pred)
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    (SALIDA/'pliegues.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    cache = {}
    anterior = AQUI/'comparacion_con_sin_esperada_todos'
    config = json.loads((anterior/'configuracion.json').read_text(encoding='utf-8'))
    for nombre in ('bandas_estables.py','criterio_usuario.py'):
        assert config['hashes'][nombre] == firma(AQUI/nombre)
    for fila in json.loads((anterior/'evidencia.json').read_text(encoding='utf-8')):
        for escenario in ('sin_esperada','con_esperada'):
            q = fila[escenario]
            if q and 'seleccion' in q:
                cache[(fila['archivo'],q['tipo'])] = q
    for fila in evidencia:
        if fila['seleccion'] is not None:
            cache[(fila['archivo'],fila['tipo'])] = fila
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    grados = []; resultados = []
    for r in d.itertuples():
        seleccion = None; v = None
        if not r.suficiente or r.prediccion == 'N':
            grado = 'ND'
        else:
            seleccion = por_archivo[r.archivo]['alternativas_seleccion'][r.prediccion]
            guardado = cache.get((r.archivo,r.prediccion))
            if guardado:
                assert guardado['seleccion'] == seleccion
                v = guardado['verificacion']; grado = guardado['grado']
            else:
                fila = idx.loc[r.archivo]; mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
                senal = np.ravel(mat[F.clave_canal(mat,r.sensor,int(fila.registro))])
                assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
                rpm = F.rpm_registro(mat,int(fila.registro),nominal=float(fila.rpm_nominal))
                c = {k:float(a*rpm/60) for k,a in F.MULTIPLICADORES[r.sensor].items()}
                v = verificar(senal[len(senal)//2:],int(r.fs),c,seleccion); grado = v['grado']
        grados.append(grado)
        resultados.append(dict(archivo=r.archivo,tipo=r.prediccion,grado=grado,seleccion=seleccion,verificacion=v))
    d['grado'] = grados; d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(resultados,ensure_ascii=False,indent=2),encoding='utf-8')
    f = d[d.paper.notna()]; ok = f.tipo_correcto; delta = f.grado.map(ORDEN)-f.paper.map(ORDEN)
    resumen = dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,desplegado=False)
    assert len(f) == 105 and sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__ == '__main__':
    main()
