"""Datos por sensor: normales DE y FE de los mismos dos archivos de desarrollo.

No abre normales de reserva ni etiqueta como sano el sensor opuesto de un archivo
con falla. Las dos vistas de un normal mantienen el mismo grupo de adquisición.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from ventanas_verificadas import rasgos
from ranking_catalogo import grupo_condicion

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_dominios_desarrollo'


if __name__=='__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente=AQUI/'resultados_ventanas_verificadas'
    meta=pd.read_csv(fuente/'muestra.csv')
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:x,origen=z['x'],z['origen']
    protocolo=dict(version='dominios_de_fe_v1', archivos=54, fallas=52, normales_archivos=2, normales_vistas=4,
        fs_normal_FE='misma frecuencia declarada del registro; se exige igual longitud DE/FE, sin elegir fs por rendimiento',
        exclusiones='ambas vistas de un normal fuera simultáneamente; condiciones de falla fuera en ambos sensores',
        reserva_abierta=False, codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fuente_sha256=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    rp=SALIDA/'protocolo.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    filas=[];bloques=[]
    for i,r in enumerate(meta.itertuples()):
        sensor='FE' if r.clave_canal.endswith('_FE_time') else 'DE'
        d=r._asdict();d.pop('Index',None)
        d.update(sensor=sensor,vista_uid=r.archivo+'|'+sensor,
                 grupo=('normal:'+r.archivo) if r.real=='N' else grupo_condicion(r.archivo))
        filas.append(d);bloques.append(x[origen==i])
        if r.real=='N':
            mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            de=np.ravel(mat[F.clave_canal(mat,'DE',int(r.registro))])
            fe=np.ravel(mat[F.clave_canal(mat,'FE',int(r.registro))])
            assert len(de)==len(fe), 'No asumir mismo fs si los canales normales tienen distinta duración'
            rpm=F.rpm_registro(mat,int(r.registro),nominal=r.rpm)
            catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES['FE'].items()}
            q=dict(d);q.update(sensor='FE',vista_uid=r.archivo+'|FE',clave_canal=F.clave_canal(mat,'FE',int(r.registro)),
                catalogo=json.dumps(catalogo,sort_keys=True),sha256_senal=hashlib.sha256(fe.astype('<f8').tobytes()).hexdigest())
            filas.append(q);bloques.append(rasgos(fe,int(r.fs_original),catalogo))
            print(f'Normal pareado {r.archivo}: DE/FE {len(de)} muestras, fs declarado {r.fs_original}',flush=True)
    datos=pd.DataFrame(filas);z=np.concatenate(bloques);o=np.repeat(np.arange(len(datos)),4)
    assert len(datos)==56 and datos.archivo.nunique()==54 and datos.vista_uid.is_unique
    assert datos.real.ne('N').sum()==52 and datos.real.eq('N').sum()==4 and z.shape==(224,1953)
    assert datos[datos.real.eq('N')].groupby('archivo').sensor.nunique().eq(2).all()
    datos.to_csv(SALIDA/'muestra.csv',index=False)
    np.savez_compressed(SALIDA/'rasgos.npz',x=z,origen=o)
    print('54 archivos: 52 fallas y dos normales. 56 vistas, 224 ventanas; sin aumentar el número de casos evaluables.')
