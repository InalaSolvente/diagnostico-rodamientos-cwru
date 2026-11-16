"""Prueba ciega uniforme del banco57; tipos y detector estrictos sin cambios."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from banco_desplazado import seleccionar_57, comprobar
from doble_banda_fijada import verificar_dos
from modelo_ciego import F
from modelo_por_dominio import firma
from puntuar_tipos_fijos import contar

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_banco_desplazado'


def main():
    comprobar()
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_grupos_por_defecto/tipos_133.csv'
    d = pd.read_csv(fuente)
    catalogos = AQUI / 'resultados_plantillas_coherentes/tipos_133.csv'
    c = pd.read_csv(catalogos).set_index('archivo').loc[d.archivo]
    assert c.sha256_senal.tolist() == d.sha256_senal.tolist()
    assert c.fs.tolist() == d.fs.tolist() and c.sensor.tolist() == d.sensor.tolist()
    d['catalogo'] = c.catalogo.tolist()
    d['registro'] = c.registro.tolist()
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    protocolo = dict(codigo=firma(Path(__file__)), selector=firma(AQUI / 'banco_desplazado.py'),
        fuente=firma(fuente), catalogos=firma(catalogos), bandas=firma(AQUI / 'bandas_estables.py'),
        verificador=firma(AQUI / 'doble_banda_fijada.py'), criterio=firma(AQUI / 'criterio_usuario.py'),
        tipo_y_detector='sin cambios respecto a referencia con grupos completos',
        entrenamiento_fallas=52, banco='57 bandas fijas; mismo ancho fisico a 12 y 48 kHz',
        seleccion='primera mitad; dos bandas disjuntas; verificacion en la segunda mitad',
        p='.01/(57*5*6*2); misma regla D/PD/ND',
        excepciones_por_archivo=False, etiquetas_para_seleccionar=False,
        exploratorio=True, validacion_externa=False)
    rp = SALIDA / 'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    d.to_csv(SALIDA / 'tipos_133.csv', index=False)
    cp = SALIDA / 'progreso.json'
    evidencia = json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else []
    hechos = {q['archivo']: q for q in evidencia}
    grados = []
    for r in d.itertuples():
        if r.archivo in hechos:
            q = hechos[r.archivo]
            assert q['tipo'] == r.prediccion and q['sha256_senal'] == r.sha256_senal
            v = {k: q[k] for k in ('grado', 'ramas')}
        elif r.prediccion in ('N', 'sin_diagnostico'):
            v = dict(grado='ND', ramas=[])
        else:
            mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
            x = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            mitad = len(x) // 2
            catalogo = json.loads(r.catalogo)
            s = seleccionar_57(x[:mitad], int(r.fs), catalogo, r.prediccion)
            v = verificar_dos(x[mitad:], int(r.fs), catalogo, s)
            print('Banco57:', r.archivo, v['grado'], flush=True)
        grados.append(v['grado'])
        if r.archivo not in hechos:
            evidencia.append(dict(archivo=r.archivo, tipo=r.prediccion, sha256_senal=r.sha256_senal, **v))
            cp.write_text(json.dumps(evidencia, ensure_ascii=False), encoding='utf-8')
    assert len(evidencia) == 133 and len({q['archivo'] for q in evidencia}) == 133
    d['grado'] = grados
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d.to_csv(SALIDA / 'comparacion_133.csv', index=False)
    (SALIDA / 'evidencia.json').write_text(json.dumps(evidencia, ensure_ascii=False, indent=2), encoding='utf-8')
    resumen = contar(d)
    resumen.update(conteo_ND_emitido=contar(d, nd_emitido=True),
        normales_correctos=int(d.loc[d.real.eq('N'), 'tipo_correcto'].sum()),
        desplegado=False, validacion_externa=False)
    (SALIDA / 'resumen.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resumen), flush=True)


if __name__ == '__main__':
    main()
