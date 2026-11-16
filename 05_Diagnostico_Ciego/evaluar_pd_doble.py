"""Puntúa PD persistente sin volver a buscar bandas en verificación."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from doble_banda_fijada import seleccionar_dos,verificar_dos
from banco_63 import seleccionar_63
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_pd_doble'


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_pd_persistente'
    tipos = pd.read_csv(fuente/'tipos_133.csv')
    padre = AQUI/'resultados_globales_bancos'
    base = pd.read_csv(padre/'comparacion_133.csv').set_index('archivo')
    evidencia = {q['archivo']:q for q in json.loads((padre/'evidencia.json').read_text(encoding='utf-8'))}
    protocolo = dict(version='PD_persistente_dos_bancos_v1',
        regla='tipos fijados antes de verificar; mismo banco31 o banco63 y mismos umbrales',
        fuente=firma(fuente/'tipos_133.csv'),base=firma(padre/'comparacion_133.csv'),
        politica=firma(AQUI/'ensayar_pd_persistente.py'),codigo=firma(Path(__file__)),
        banco31=firma(AQUI/'doble_banda_fijada.py'),banco63=firma(AQUI/'banco_63.py'),
        limitacion='ensayo exploratorio sobre datos ya inspeccionados; sin validación externa')
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    filas = []; evidencias = []
    for r in tipos.itertuples():
        b = base.loc[r.archivo]
        if r.prediccion == b.prediccion:
            ramas = {k:evidencia[r.archivo][k] for k in ('banco31','banco63')}
        else:
            a = idx.loc[r.archivo]; mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            x = np.ravel(mat[F.clave_canal(mat,r.sensor,int(a.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
            rpm = F.rpm_registro(mat,int(a.registro),nominal=float(a.rpm_nominal))
            c = {k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor].items()}
            mitad = len(x)//2
            s31 = seleccionar_dos(x[:mitad],int(r.fs),c,r.prediccion)
            s63 = seleccionar_63(x[:mitad],int(r.fs),c,r.prediccion)
            # Ambas elecciones se fijan antes de evaluar el segmento reservado.
            ramas = dict(banco31=verificar_dos(x[mitad:],int(r.fs),c,s31),
                banco63=verificar_dos(x[mitad:],int(r.fs),c,s63))
            print('Tipo cambiado, bandas recalculadas:',r.archivo,r.prediccion,flush=True)
        fila = r._asdict(); fila.pop('Index',None)
        fila.update(grado31=ramas['banco31']['grado'],grado63=ramas['banco63']['grado'])
        filas.append(fila); evidencias.append(dict(archivo=r.archivo,tipo=r.prediccion,**ramas))
    d = pd.DataFrame(filas)
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    resumen = []
    for banco,col in [(31,'grado31'),(63,'grado63')]:
        f = d[d.paper.notna()]; delta = f[col].map(ORDEN)-f.paper.map(ORDEN); ok = f.tipo_correcto
        q = dict(banco=banco,total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
            empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
            sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
            validacion_externa=False,desplegado=False)
        assert len(f) == 105 and sum(q[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 105
        resumen.append(q)
    pd.DataFrame(resumen).to_csv(SALIDA/'resumen.csv',index=False)
    print(pd.DataFrame(resumen).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
