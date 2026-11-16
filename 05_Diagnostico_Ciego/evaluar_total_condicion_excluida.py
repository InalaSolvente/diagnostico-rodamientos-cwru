"""Audita los 133 archivos con toda su condicion fuera del entrenamiento.

Entrena exclusivamente con la muestra de desarrollo historica. Para cada caso
retira sus cargas, sensores y frecuencias de muestreo de la misma condicion.
Es una reevaluacion interna, no una prueba externa ni una reserva nueva.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from modelo_ciego import F
from modelo_por_dominio import crear, calidad, firma
from ventanas_verificadas import rasgos
from ranking_catalogo import grupo_condicion
from ensayar_tipo_por_sensor import predecir_tipo
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
DATOS = AQUI.parent/'data2/CWRU'
FUENTE = AQUI/'resultados_dominios_desarrollo'
SALIDA = AQUI/'resultados_total_condicion_excluida'


def main():
    SALIDA.mkdir(exist_ok=True)
    meta = pd.read_csv(FUENTE/'muestra.csv')
    with np.load(FUENTE/'rasgos.npz', allow_pickle=False) as z:
        x, origen = z['x'], z['origen']
    idx = F.leer_indice(str(DATOS))
    assert len(idx) == 133 and idx.archivo.is_unique
    dependencias = ('modelo_espectral.py', 'ventanas_verificadas.py',
        'modelo_por_dominio.py', 'bandas_estables.py', 'criterio_usuario.py',
        'ranking_catalogo.py', 'ensayar_tipo_por_sensor.py')
    protocolo = dict(version='total_condicion_excluida_v1',
        entrenamiento='solo 52 fallas y dos archivos normales de desarrollo historico',
        exclusion='toda la condicion de falla en ambos sensores y muestreos; normal por archivo',
        tipo='LDA shrinkage auto agrupado DE/FE, identico al jerarquico historico',
        detector='LDA shrinkage auto por sensor, reentrenado excluyendo condicion',
        entrada='senal, fs y catalogo completo; etiquetas solo entrenamiento y evaluacion',
        seleccion_banda='primera mitad; verificar banda fija en segunda mitad',
        limitacion='dataset previamente inspeccionado; exploratorio, sin validacion externa',
        codigo=firma(Path(__file__)), fuente=firma(FUENTE/'rasgos.npz'),
        muestra=firma(FUENTE/'muestra.csv'),
        hashes={n: firma(AQUI/n) for n in dependencias})
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    previo = AQUI/'comparacion_con_sin_esperada_todos'
    config = json.loads((previo/'configuracion.json').read_text(encoding='utf-8'))
    for n in ('bandas_estables.py', 'criterio_usuario.py'):
        assert config['hashes'][n] == firma(AQUI/n)
    anteriores = pd.read_csv(previo/'comparacion_133.csv').set_index('archivo')
    cache = {}
    for fila in json.loads((previo/'evidencia.json').read_text(encoding='utf-8')):
        for escenario in ('sin_esperada', 'con_esperada'):
            q = fila[escenario]
            if q and 'seleccion' in q:
                cache[(fila['archivo'], q['tipo'])] = {
                    k:q[k] for k in ('tipo', 'grado', 'seleccion', 'verificacion')}
    y = meta.real.to_numpy(); grupos = meta.grupo.to_numpy(); dom = meta.sensor.to_numpy()
    ajustes = {}; pliegues = []; filas = []; evidencias = []; bloques = []
    with threadpool_limits(limits=1):
        for num, r in enumerate(idx.itertuples()):
            mat = loadmat(DATOS/r.archivo)
            clave = F.clave_canal(mat, r.sensor_falla, int(r.registro))
            senal = np.ravel(mat[clave])
            huella = hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest()
            assert huella == anteriores.loc[r.archivo, 'sha256_senal']
            rpm = F.rpm_registro(mat, int(r.registro), nominal=float(r.rpm_nominal))
            c = {k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
            # El grupo solo controla el evaluador: no se pasa a los predictores.
            grupo = 'normal:'+r.archivo if r.zona == 'N' else grupo_condicion(r.archivo)
            suficiente = len(senal)//2 >= 2.5*r.fs
            if not suficiente:
                tipo = 'sin_diagnostico'; gate = 'sin_diagnostico'
                q = dict(tipo=tipo, grado='ND', motivo='menos_de_5_segundos')
                z = np.full((4, x.shape[1]), np.nan)
            else:
                if grupo not in ajustes:
                    train = grupos != grupo
                    assert grupo not in set(grupos[train])
                    assert r.archivo not in set(meta.loc[train, 'archivo'])
                    ft = (train & (y != 'N'))[origen]
                    clasificador = crear().fit(x[ft], y[origen[ft]])
                    detectores = {}
                    for sensor in ('DE', 'FE'):
                        fd = (train & (dom == sensor))[origen]
                        assert set(y[origen[fd]]) == {'N', 'B', 'IR', 'OR'}
                        detectores[sensor] = crear().fit(x[fd], y[origen[fd]])
                    ajustes[grupo] = (clasificador, detectores)
                    pliegues.append(dict(grupo_excluido=grupo,
                        train=meta.loc[train, 'vista_uid'].tolist(),
                        grupos_train=sorted(set(grupos[train]))))
                clf, detectores = ajustes[grupo]
                conocido = meta.index[meta.archivo.eq(r.archivo) & meta.sensor.eq(r.sensor_falla)]
                if len(conocido):
                    j = int(conocido[0]); m = meta.loc[j]
                    assert m.sha256_senal == huella and int(m.fs_original) == int(r.fs)
                    assert json.loads(m.catalogo) == c
                    z = x[origen == j]
                else:
                    z = rasgos(senal, int(r.fs), c)
                gate = predecir_tipo(detectores[r.sensor_falla], z)
                tipo = 'N' if gate == 'N' else predecir_tipo(clf, z)
                q = cache.get((r.archivo, tipo))
                if q is None:
                    q = calidad(senal, int(r.fs), c, tipo)
            bloques.append(z)
            filas.append(dict(archivo=r.archivo, sensor=r.sensor_falla, fs=int(r.fs),
                real=r.zona, grupo=grupo, sha256_senal=huella, suficiente=suficiente,
                detector=gate, prediccion=tipo, grado=q['grado'],
                paper=anteriores.loc[r.archivo, 'paper'], tipo_correcto=tipo == r.zona))
            evidencias.append(dict(archivo=r.archivo, **q))
            pd.DataFrame(filas).to_csv(SALIDA/'comparacion_parcial.csv', index=False)
            print(f'{num+1}/133 condicion excluida: {r.archivo} {tipo}/{q["grado"]}', flush=True)
    d = pd.DataFrame(filas)
    assert d.sha256_senal.is_unique and len(d) == 133
    # El nuevo evaluador debe reproducir el OOF historico, caso por caso.
    oof = pd.read_csv(AQUI/'resultados_jerarquico/comparacion.csv')
    comparables = oof[oof.real.ne('N')].merge(d, on='archivo', suffixes=('_antes', '_ahora'), validate='one_to_one')
    assert len(comparables) == 52
    assert comparables.prediccion_antes.eq(comparables.prediccion_ahora).all()
    assert comparables.grado_antes.eq(comparables.grado_ahora).all()
    for fold in pliegues:
        assert fold['grupo_excluido'] not in fold['grupos_train']
    d.to_csv(SALIDA/'comparacion_133.csv', index=False)
    np.savez_compressed(SALIDA/'rasgos_evaluacion.npz', x=np.asarray(bloques))
    (SALIDA/'pliegues.json').write_text(json.dumps(pliegues, indent=2), encoding='utf-8')
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias, ensure_ascii=False, indent=2), encoding='utf-8')
    f = d[d.paper.notna()]; delta = f.grado.map(ORDEN)-f.paper.map(ORDEN); ok = f.tipo_correcto
    assert len(f) == 105
    resumen = dict(total=105, mejoran=int((ok & delta.gt(0)).sum()),
        iguales=int((ok & delta.eq(0)).sum()), empeoran=int((ok & delta.lt(0)).sum()),
        tipo_incorrecto=int((~ok).sum()), sin_retroceso=int((ok & delta.ge(0)).sum()),
        normales_correctos=int(d.loc[d.real.eq('N'), 'tipo_correcto'].sum()), normales=4,
        fallas_48k_tipo_correcto=int(d.loc[d.fs.eq(48000) & d.real.ne('N'), 'tipo_correcto'].sum()),
        fallas_48k=24, desarrollo_52_reproducido=True, validacion_externa=False)
    assert sum(resumen[k] for k in ('mejoran', 'iguales', 'empeoran', 'tipo_incorrecto')) == 105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen, indent=2), encoding='utf-8')
    print(json.dumps(resumen), flush=True)


if __name__ == '__main__':
    main()
