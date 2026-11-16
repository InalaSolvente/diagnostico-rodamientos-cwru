"""Evalúa hipótesis congeladas con el mismo banco; sin elegir tipos por etiqueta."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from scipy.io import loadmat
from modelo_ciego import F
from doble_banda_fijada import seleccionar_dos,verificar_dos
from criterio_usuario import ORDEN,comparar

AQUI=Path(__file__).parent


def contar(d,*,nd_emitido=False):
    c=dict(total=0,mejoran=0,iguales=0,empeoran=0,tipo_incorrecto=0,sin_retroceso=0)
    for r in d[d.paper.notna()].itertuples():
        c['total']+=1
        if nd_emitido:
            salida,ok=comparar(r.paper,r.grado,r.tipo_correcto)
        else:
            salida=r.grado if r.tipo_correcto else 'tipo_incorrecto'
            ok=r.tipo_correcto and ORDEN[r.grado]>=ORDEN[r.paper]
        if salida=='tipo_incorrecto':c['tipo_incorrecto']+=1
        else:
            delta=ORDEN[salida]-ORDEN[r.paper]
            c['mejoran' if delta>0 else 'iguales' if delta==0 else 'empeoran']+=1
        c['sin_retroceso']+=int(ok)
    assert sum(c[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto'))==c['total']
    return c


def evaluar(d,salida):
    salida=Path(salida)
    assert len(d)==133 and d.archivo.is_unique and d.sha256_senal.is_unique
    # El llamador guarda tipos_133.csv antes de abrir cualquier verificación.
    congelado=pd.read_csv(salida/'tipos_133.csv')
    assert congelado.archivo.equals(d.archivo) and congelado.prediccion.equals(d.prediccion)
    cache={}
    permitidos=[('resultados_grupos_por_defecto',False),('resultados_pd_doble',True),
        ('resultados_globales_bancos',True),('resultados_doble_banda_fijada',False),
        ('resultados_plantillas_coherentes',False)]
    for carpeta,banco in permitidos:
        # Solo resultados sobre señal cruda: nunca reutilizar los del filtro AR.
        p=AQUI/carpeta
        datos=pd.read_csv(p/'comparacion_133.csv').set_index('archivo')
        for q in json.loads((p/'evidencia.json').read_text(encoding='utf-8')):
            if q['archivo'] in set(d.archivo):
                assert datos.loc[q['archivo'],'sha256_senal']==d.set_index('archivo').loc[q['archivo'],'sha256_senal']
            cache[(q['archivo'],q['tipo'])]=q['banco31'] if banco else {k:q[k] for k in ('grado','ramas')}
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    grados=[];evidencias=[]
    for r in d.itertuples():
        if r.prediccion in ('N','sin_diagnostico'):v=dict(grado='ND',ramas=[])
        elif (r.archivo,r.prediccion) in cache:v=cache[(r.archivo,r.prediccion)]
        else:
            a=idx.loc[r.archivo];mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            x=np.ravel(mat[F.clave_canal(mat,r.sensor,int(a.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
            mitad=len(x)//2;c=json.loads(r.catalogo)
            s=seleccionar_dos(x[:mitad],int(r.fs),c,r.prediccion)
            v=verificar_dos(x[mitad:],int(r.fs),c,s)
            print('Bandas nuevas:',r.archivo,r.prediccion,flush=True)
        grados.append(v['grado']);evidencias.append(dict(archivo=r.archivo,tipo=r.prediccion,**v))
    d=d.copy();d['grado']=grados;d['tipo_correcto']=d.real.eq(d.prediccion)
    # ND es una falta de diagnóstico; la hipótesis interna se conserva aparte.
    d['tipo_diagnosticado']=d.prediccion.where(d.grado.ne('ND'))
    d.to_csv(salida/'comparacion_133.csv',index=False)
    (salida/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    r=contar(d)
    r.update(conteo_ND_emitido=contar(d,nd_emitido=True),
        criterio_principal='tipo exigido en ND, como matrices recientes; consulta al usuario pendiente',
        normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,desplegado=False)
    assert r['total']==105
    (salida/'resumen.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False),flush=True)
    return d,r
