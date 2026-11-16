"""Ensayo uniforme de TSA en el grado; congela los tipos del modelo temporal84."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tsa_controlado import aplicar, comprobar, FS
from modelo_ciego import F
from modelo_por_dominio import firma
from banco_desplazado import seleccionar_57
from doble_banda_fijada import verificar_dos
from puntuar_tipos_fijos import contar

AQUI = Path(__file__).parent
SALIDA = AQUI / 'resultados_tsa_grados'


def main():
    comprobar()
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_temporales_acp/tipos_133.csv'
    d = pd.read_csv(fuente).drop(columns=['tipo_diagnosticado'])
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    protocolo = dict(codigo=firma(Path(__file__)), filtro=firma(AQUI / 'tsa_controlado.py'),
        fuente=firma(fuente), selector=firma(AQUI / 'banco_desplazado.py'),
        verificador=firma(AQUI / 'doble_banda_fijada.py'), criterio=firma(AQUI / 'criterio_usuario.py'),
        receta='TSA ajustado en selección; protege todas las familias H1..6 +/-2% y resolución',
        giro='inferido del catálogo completo, no de una etiqueta de falla',
        remuestreo='mitades separadas a 12k; origen de fase de verificación conservado',
        clasificador='tipos temporales84 congelados, sin cambios ni reentrenamiento',
        grado='mismo banco57, dos ramas y regla D/PD/ND; se filtran ambas mitades',
        eleccion_del_metodo_por_etiqueta=False, etiquetas_para_filtrar=False,
        limitacion='ensayo exploratorio del grado; no puede corregir tipos incorrectos')
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
    for i, r in enumerate(d.itertuples()):
        if r.archivo in hechos:
            q = hechos[r.archivo]
            assert q['tipo'] == r.prediccion and q['sha256_senal'] == r.sha256_senal
            v = {k: q[k] for k in ('grado', 'ramas')}
            meta = q['tsa']
        elif r.prediccion in ('N', 'sin_diagnostico'):
            v, meta = dict(grado='ND', ramas=[]), None
        else:
            mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
            x = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            c = json.loads(r.catalogo)
            seleccion, verificacion, meta = aplicar(x, int(r.fs), c)
            s = seleccionar_57(seleccion, FS, c, r.prediccion)
            v = verificar_dos(verificacion, FS, c, s)
        grados.append(v['grado'])
        if r.archivo not in hechos:
            evidencia.append(dict(archivo=r.archivo, tipo=r.prediccion, sha256_senal=r.sha256_senal, tsa=meta, **v))
            cp.write_text(json.dumps(evidencia, ensure_ascii=False), encoding='utf-8')
        if (i+1) % 10 == 0 or i == len(d)-1:
            print('TSA, grado:', i+1, '/133', flush=True)
    assert len(evidencia) == 133
    d['grado'] = grados
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d['tipo_diagnosticado'] = d.prediccion.where(d.grado.ne('ND'))
    d.to_csv(SALIDA / 'comparacion_133.csv', index=False)
    (SALIDA / 'evidencia.json').write_text(json.dumps(evidencia, ensure_ascii=False, indent=2), encoding='utf-8')
    r = contar(d)
    r.update(conteo_ND_emitido=contar(d, nd_emitido=True),
        normales_correctos=int(d.loc[d.real.eq('N'), 'tipo_correcto'].sum()),
        validacion_externa=False, desplegado=False)
    (SALIDA / 'resumen.json').write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
