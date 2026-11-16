"""Combina curvas coherentes con la modulación ya implementada, sin oráculo."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from bandas_laterales import rasgos_laterales

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_plantillas_modulacion'


def preparar():
    SALIDA.mkdir(exist_ok=True)
    padre=AQUI/'resultados_plantillas_coherentes'
    d=pd.read_csv(padre/'muestra.csv')
    with np.load(padre/'rasgos.npz',allow_pickle=False) as z:coherentes=z['x']
    assert coherentes.shape==(133,724) and d.archivo.is_unique
    protocolo=dict(version='curvas724_laterales240_v1',
        hipotesis='añadir modulación por giro y FTF, además de alineación del pico principal',
        rasgos='724 coherentes + media de cuatro ventanas de 240 rasgos laterales existentes',
        periodo='primeros 2.5 s; todas las familias, nunca tipo esperado o verificación',
        dimensiones=964,codigo=firma(Path(__file__)),laterales=firma(AQUI/'bandas_laterales.py'),
        curvas=firma(padre/'rasgos.npz'),muestra=firma(padre/'muestra.csv'))
    rp=SALIDA/'protocolo_preparacion.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    cache={}
    anterior=AQUI/'resultados_laterales_desarrollo'
    p=json.loads((anterior/'protocolo.json').read_text(encoding='utf-8'))
    if p['codigo']==firma(AQUI/'bandas_laterales.py'):
        meta=pd.read_csv(anterior/'muestra.csv')
        with np.load(anterior/'rasgos.npz',allow_pickle=False) as z:
            x,origen=z['x'][:,-240:],z['origen']
        assert x.shape==(224,240)
        for i,r in enumerate(meta.itertuples()):
            clave=(r.archivo,r.sensor,r.sha256_senal,int(r.fs_original),json.dumps(json.loads(r.catalogo),sort_keys=True))
            cache[clave]=x[origen==i].mean(axis=0)
    bloques=[];reusados=0
    for i,r in enumerate(d.itertuples()):
        if not r.suficiente:v=np.full(240,np.nan)
        else:
            clave=(r.archivo,r.sensor,r.sha256_senal,int(r.fs),json.dumps(json.loads(r.catalogo),sort_keys=True))
            if clave in cache:v=cache[clave];reusados+=1
            else:
                mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
                x=np.ravel(mat[F.clave_canal(mat,r.sensor,int(r.registro))])
                assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
                v=rasgos_laterales(x,int(r.fs),json.loads(r.catalogo)).mean(axis=0)
        bloques.append(np.r_[coherentes[i],v])
        if (i+1)%10==0 or i==len(d)-1:print(f'Modulación {i+1}/133',flush=True)
    x=np.asarray(bloques)
    assert x.shape==(133,964)
    assert np.isfinite(x[d.suficiente]).all()
    d.to_csv(SALIDA/'muestra.csv',index=False)
    np.savez_compressed(SALIDA/'rasgos.npz',x=x)
    print('OK: 133 registros; rasgos existentes reutilizados con firma de entrada completa:',reusados)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparar',action='store_true',required=True)
    parser.parse_args();preparar()
