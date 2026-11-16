"""Piloto CONDICIONAL en desarrollo: no valida el clasificador ni abre la reserva."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from bandas_estables import seleccionar, verificar
from criterio_usuario import ORDEN
from modelo_ciego import F

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_bandas_estables_desarrollo'


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    protocolo = dict(nombre='bandas_estables_v1', alcance='solo 52 fallas de desarrollo 12k',
        selecciones='tipo OOF histórico congelado; banda/familia/representación en primera mitad',
        verificaciones='misma banda/familia/representación en segunda mitad, sin búsqueda adicional',
        ranking='grado de selección, cuartil inferior de márgenes en tres subsegmentos, armónicas',
        transformaciones=['amplitud Hann', 'envolvente cuadrada Hann'],
        p='.01 / (31 bandas x 5 familias x 6 armónicas x 2 representaciones), nominal',
        limitacion='El tipo histórico usó el registro completo; ensayo condicional, no objetivo alcanzado ni validación independiente',
        codigo_sha256=hashlib.sha256((AQUI/'bandas_estables.py').read_bytes()).hexdigest())
    ruta_protocolo = SALIDA/'protocolo.json'
    if ruta_protocolo.exists():
        assert json.loads(ruta_protocolo.read_text(encoding='utf-8')) == protocolo, 'Nuevo algoritmo requiere otro experimento, no sobrescribir'
    else:
        ruta_protocolo.write_text(json.dumps(protocolo, indent=2, ensure_ascii=False), encoding='utf-8')
    split = pd.read_csv(AQUI/'particion_desarrollo.csv')
    dev = set(split.loc[split.particion.eq('desarrollo') & split.carpeta.ne('Normal'), 'archivo'])
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo', drop=False)
    tipos = pd.read_csv(AQUI/'resultados_espectrales_desarrollo/verificacion_oof.csv').set_index('archivo')
    assert len(dev) == 52 and dev.issubset(set(tipos.index))
    filas = []
    for i, archivo in enumerate(sorted(dev)):
        r = idx.loc[archivo]
        mat = loadmat(AQUI.parent/'data2/CWRU'/archivo)
        clave = next(k for k in mat if k.endswith(f'_{r.sensor_falla}_time'))
        krpm = [k for k in mat if k.endswith('RPM')]
        rpm = float(np.ravel(mat[krpm[0]])[0]) if krpm else float(r.rpm_nominal)
        catalogo = {k: float(v*rpm/60) for k, v in F.MULTIPLICADORES[r.sensor_falla].items()}
        x = np.ravel(mat[clave]); mitad = len(x)//2
        tipo = tipos.loc[archivo, 'prediccion']
        q = seleccionar(x[:mitad], float(r.fs), catalogo, tipo)
        ver = verificar(x[mitad:], float(r.fs), catalogo, q)
        filas.append(dict(archivo=archivo, tipo_predicho=tipo, grado=ver['grado'],
            seleccion=q, verificacion=ver, sha256_senal=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()))
        print(f'{i+1}/52 {archivo}: {q["grado"]}/{ver["grado"]}', flush=True)
        (SALIDA/'predicciones.json').write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding='utf-8')
    # Referencia y verdad consultadas para evaluar DESPUÉS de fijar todas las salidas.
    pred = pd.DataFrame([{k: r[k] for k in ('archivo', 'tipo_predicho', 'grado')} for r in filas]).set_index('archivo')
    base = pd.read_csv(AQUI/'auditoria_modelos_actuales/persistencia_temporal_sin_copias.csv').set_index('archivo')
    d = pred.join(base.loc[sorted(dev), ['paper', 'grado_conservador', 'tipo_correcto']])
    d.to_csv(SALIDA/'comparacion.csv')
    resumen = []
    for nombre, columna in [('base_auditada', 'grado_conservador'), ('bandas_estables_v1', 'grado')]:
        delta = d[columna].map(ORDEN)-d.paper.map(ORDEN)
        ok = d.tipo_correcto.eq(True)
        resumen.append(dict(metodo=nombre, total=len(d), mejoran=int((ok & delta.gt(0)).sum()),
            iguales=int((ok & delta.eq(0)).sum()), empeoran=int((ok & delta.lt(0)).sum()),
            tipo_incorrecto=int((~ok).sum()), sin_retroceso=int((ok & delta.ge(0)).sum())))
    pd.DataFrame(resumen).to_csv(SALIDA/'resumen.csv', index=False)
    print(pd.DataFrame(resumen).to_string(index=False))
