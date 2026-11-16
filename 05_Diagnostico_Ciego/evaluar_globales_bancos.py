"""Evalúa dos bancos fijados previamente con las mismas predicciones de tipo."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from doble_banda_fijada import seleccionar_dos,verificar_dos
from banco_63 import seleccionar_63,comprobar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_globales_bancos'


def main():
    comprobar(); SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_globales_arbol'
    tipos = pd.read_csv(fuente/'tipos_133.csv').set_index('archivo',drop=False)
    padre = AQUI/'resultados_doble_banda_fijada'
    previo = pd.read_csv(padre/'comparacion_133.csv').set_index('archivo',drop=False)
    evidencia_previa = {q['archivo']:q for q in json.loads((padre/'evidencia.json').read_text(encoding='utf-8'))}
    protocolo = dict(version='globales_dos_bancos_fijos_v1',
        tipo='mismas decisiones del clasificador arbol+globales con condicion excluida y control de giro',
        variantes='31 bandas o 63 bandas; maximo dos disjuntas fijadas antes de verificar',
        umbral='p=.01/(numero_bandas*5*6*2), mas estricto al ampliar banco',
        seleccion='primera mitad; tipo y ambas bandas fijos antes de usar segunda mitad',
        limitacion='comparacion exploratoria; dataset ya inspeccionado; p nominal sin calibracion empirica',
        codigo=firma(Path(__file__)),tipos=firma(fuente/'tipos_133.csv'),
        banco31=firma(AQUI/'doble_banda_fijada.py'),banco63=firma(AQUI/'banco_63.py'),
        bandas=firma(AQUI/'bandas_estables.py'),criterio=firma(AQUI/'criterio_usuario.py'))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU'))
    filas = []; evidencias = []
    for i,r in enumerate(idx.itertuples()):
        t = tipos.loc[r.archivo]; b = previo.loc[r.archivo]
        ramas = {}
        if not t.suficiente or t.prediccion == 'N':
            ramas = {k:dict(grado='ND',ramas=[]) for k in ('banco31','banco63')}
        else:
            mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            x = np.ravel(mat[F.clave_canal(mat,r.sensor_falla,int(r.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == t.sha256_senal
            rpm = F.rpm_registro(mat,int(r.registro),nominal=float(r.rpm_nominal))
            c = {k:float(a*rpm/60) for k,a in F.MULTIPLICADORES[r.sensor_falla].items()}
            mitad = len(x)//2
            if t.prediccion == b.prediccion:
                ramas['banco31'] = {k:evidencia_previa[r.archivo][k] for k in ('grado','ramas')}
            else:
                selecciones = seleccionar_dos(x[:mitad],int(r.fs),c,t.prediccion)
                ramas['banco31'] = verificar_dos(x[mitad:],int(r.fs),c,selecciones)
            # Seleccionar sin mirar ninguna verificación, también en el banco ampliado.
            selecciones63 = seleccionar_63(x[:mitad],int(r.fs),c,t.prediccion)
            ramas['banco63'] = verificar_dos(x[mitad:],int(r.fs),c,selecciones63)
        filas.append(dict(t,grado31=ramas['banco31']['grado'],grado63=ramas['banco63']['grado']))
        evidencias.append(dict(archivo=r.archivo,tipo=t.prediccion,**ramas))
        pd.DataFrame(filas).to_csv(SALIDA/'comparacion_parcial.csv',index=False)
        print(f'{i+1}/133 globales: {r.archivo} {t.prediccion} 31={ramas["banco31"]["grado"]} 63={ramas["banco63"]["grado"]}',flush=True)
    d = pd.DataFrame(filas)
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    resumen = []
    for nombre,col in [('globales_banco31','grado31'),('globales_banco63','grado63')]:
        f = d[d.paper.notna()]; delta = f[col].map(ORDEN)-f.paper.map(ORDEN); ok = f.tipo_correcto
        q = dict(variante=nombre,total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
            empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
            sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
            validacion_externa=False,desplegado=False)
        assert len(f) == 105 and sum(q[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 105
        resumen.append(q)
    pd.DataFrame(resumen).to_csv(SALIDA/'resumen.csv',index=False)
    print(pd.DataFrame(resumen).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
