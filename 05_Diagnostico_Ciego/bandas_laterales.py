"""Mide modulación de todas las familias; no recibe defecto/frecuencia real.

Mide líneas a ±giro y ±FTF de tres armónicas de cada familia. No inserta,
elimina ni realza líneas en el espectro usado para asignar ND/PD/D.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from bandas_estables import envolvente,espectro
from alinear_bandas import geometria
from ventanas_verificadas import prefijo
from rasgos_fisicos import giro_catalogo

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_laterales_desarrollo'


def intervalo(f,centro,radio):
    a,b=np.searchsorted(f,[centro-radio,centro+radio],side='left')
    a=max(int(a),1);b=min(int(b)+1,len(f))
    return a,b


def tripleta(f,s,objetivo,separacion):
    df=f[1]-f[0];eps=max(float(s.max())*1e-15,np.finfo(float).tiny)
    lo,hi=intervalo(f,objetivo,max(.02*objetivo,df))
    if hi<=lo:return np.full(4,-60.)
    i=lo+int(np.argmax(s[lo:hi]));centro=float(f[i])
    radio=min(max(.02*objetivo,df),.2*separacion)
    if centro-separacion-radio<=0:return np.full(4,-60.)
    a,b=intervalo(f,centro-1.5*separacion,1.5*separacion)
    # Fondo alrededor de la tripleta, excluyendo sus tres ventanas.
    a,b=intervalo(f,centro,1.5*separacion)
    soporte=f[a:b];fondo=np.ones(len(soporte),dtype=bool)
    potencias=[]
    for c in (centro-separacion,centro,centro+separacion):
        u,v=intervalo(f,c,radio)
        if v<=u:return np.full(4,-60.)
        potencias.append(float(s[u:v].max()))
        fondo &= abs(soporte-c)>radio
    media=float(s[a:b][fondo].mean()) if fondo.any() else float(s[1:].mean())
    c=np.clip(10*np.log10((np.asarray(potencias)+eps)/(media+eps)),-60,60)
    relativo=float(np.clip(10*np.log10((potencias[0]+potencias[2]+eps)/(2*potencias[1]+eps)),-60,60))
    return np.array([min(c[0],c[2]),min(c),relativo,-abs(c[0]-c[2])])


def ventana(x,catalogo):
    fr=giro_catalogo(catalogo)
    familias=[catalogo[k] for k in ('BPFI','BPFO','BSF')]+[2*catalogo['BSF'],catalogo['FTF']]
    nodos,_=geometria(12000);matriz=[]
    for nodo in nodos:
        f,s=espectro(envolvente(x,12000,nodo),12000)
        fila=[]
        for base in familias:
            for h in (1,2,3):
                for separacion in (fr,catalogo['FTF']):fila.append(tripleta(f,s,h*base,separacion))
        matriz.append(fila)
    matriz=np.asarray(matriz)  # 31 bandas × 30 hipótesis × 4 medidas
    salida=[]
    for i in range(30):
        mejores=np.argsort(-matriz[:,i,1],kind='stable')[:3]
        # Las cuatro medidas provienen juntas de la misma banda, no de máximos incompatibles.
        salida.extend(matriz[mejores[0],i]);salida.extend(np.median(matriz[mejores,i],axis=0))
    z=np.asarray(salida)
    assert z.shape==(240,) and np.isfinite(z).all()
    return z


def rasgos_laterales(x,fs,catalogo):
    z=prefijo(x,fs)
    return np.asarray([ventana(z[i:i+12000],catalogo) for i in (0,6000,12000,18000)])


if __name__=='__main__':
    SALIDA.mkdir(exist_ok=True)
    fuente=AQUI/'resultados_dominios_desarrollo';meta=pd.read_csv(fuente/'muestra.csv')
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:x,origen=z['x'],z['origen']
    protocolo=dict(version='laterales_h3_v1',dimensiones_laterales=240,
        frecuencias='todas BPFI/BPFO/BSF/2BSF/FTF; separaciones giro y FTF, inferidas del catálogo completo',
        ventanas='primeros 2,5 s, cuatro ventanas de 1 s, sin acceso a verificación',
        radio='máximo un quinto de separación para no contar la central como lateral',
        calificacion_modificada=False,codigo=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fuente=hashlib.sha256((fuente/'rasgos.npz').read_bytes()).hexdigest())
    rp=SALIDA/'protocolo.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    nuevos=np.empty((len(x),240))
    for i,r in enumerate(meta.itertuples()):
        m=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
        senal=np.ravel(m[r.clave_canal])
        assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
        nuevos[origen==i]=rasgos_laterales(senal,int(r.fs_original),json.loads(r.catalogo))
        print(f'Líneas laterales {i+1}/56',flush=True)
    ampliados=np.column_stack([x,nuevos]);assert ampliados.shape==(224,2193)
    np.savez_compressed(SALIDA/'rasgos.npz',x=ampliados,origen=origen)
    meta.to_csv(SALIDA/'muestra.csv',index=False)
    print('Modulación medida sin usar etiquetas ni referencia del paper.',flush=True)
