"""Preblanqueo causal AR(32), ajustado solo con los primeros 2,5 segundos.

La API del filtro no recibe catálogo, tipo, nombre de archivo ni etiquetas.
El remuestreo de las mitades se separa para impedir contaminación desde futuro.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import resample_poly,correlate,lfilter
from scipy.linalg import solve_toeplitz
from scipy.io import loadmat
from modelo_ciego import F
from modelo_espectral import extraer
from modelo_por_dominio import firma
from ranking_catalogo import grupo_condicion

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_preblanqueo_ar'
FS=12000
ORDEN=32
AJUSTE=30000


def preblanquear(x,fs):
    if np.iscomplexobj(x):raise ValueError('Se requiere aceleración real')
    x=np.asarray(x,dtype=float)
    if fs not in (12000,48000) or x.ndim!=1 or not np.isfinite(x).all() or len(x)//2<2.5*fs:
        raise ValueError('Se requieren señal finita, fs 12/48 kHz y al menos 5 segundos')
    mitad=len(x)//2
    if fs==48000:
        seleccion=resample_poly(x[:mitad],1,4,padtype='line')
        futuro=resample_poly(x[mitad:],1,4,padtype='line')
        z=np.r_[seleccion,futuro]
    else:
        seleccion=x[:mitad];z=x.copy()
    media=float(z[:AJUSTE].mean());z=z-media;prefijo=z[:AJUSTE]
    r=correlate(prefijo,prefijo,mode='full',method='fft')[AJUSTE-1:AJUSTE+ORDEN]/AJUSTE
    coef=np.zeros(ORDEN+1);coef[0]=1.
    if r[0]>np.finfo(float).tiny:
        col=r[:ORDEN].copy();col[0]*=1+1e-6
        coef[1:]=-solve_toeplitz(col,r[1:ORDEN+1],check_finite=True)
    # Calentamiento por reflexión usando únicamente el comienzo de selección.
    anterior=z[1:ORDEN+1][::-1]
    y=lfilter(coef,[1.],np.r_[anterior,z])[ORDEN:]
    assert np.isfinite(y).all() and np.isfinite(coef).all()
    assert len(y)//2==len(seleccion)
    info=dict(fs_procesado=FS,orden=ORDEN,ajuste_muestras=AJUSTE,media=media,
        coef=coef.tolist(),muestras_seleccion=len(seleccion),
        energia_relativa=float(np.mean(y[:AJUSTE]**2)/max(float(np.mean(prefijo**2)),np.finfo(float).tiny)))
    return y,info


def rasgos_blancos(y,catalogo):
    filas=[]
    for inicio in (0,6000,12000,18000):
        q=extraer(y[inicio:inicio+FS],FS,catalogo)
        filas.append(np.r_[q['arbol'],q['globales']])
    x=np.asarray(filas)
    assert x.shape==(4,2021) and np.isfinite(x).all()
    return x


def comprobar():
    rng=np.random.default_rng(6205)
    for fs in (12000,48000):
        x=rng.normal(size=6*fs+1)
        a,ia=preblanquear(x,fs)
        otro=x.copy();otro[len(x)//2:]=rng.normal(scale=3,size=len(x)-len(x)//2)
        b,ib=preblanquear(otro,fs)
        assert ia['coef']==ib['coef'] and ia['media']==ib['media']
        assert np.array_equal(a[:len(a)//2],b[:len(b)//2])
        assert not np.array_equal(a[len(a)//2:],b[len(b)//2:])
        c,ic=preblanquear(3*x,fs)
        assert np.allclose(ia['coef'],ic['coef'],atol=1e-10)
        assert np.allclose(3*a,c,atol=1e-10)
    y,info=preblanquear(np.zeros(60000),12000)
    assert not y.any() and info['coef'][0]==1
    print('OK: filtro independiente del futuro a 12/48 kHz, escala lineal y entrada constante.')


def preparar():
    SALIDA.mkdir(exist_ok=True)
    original=AQUI/'resultados_dominios_desarrollo/muestra.csv'
    muestra=pd.read_csv(original)
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU'))
    anteriores=pd.read_csv(AQUI/'resultados_grupos_por_defecto/comparacion_133.csv').set_index('archivo')
    protocolo=dict(version='AR32_prefijo_25_canonico_v1',orden=ORDEN,fs=FS,ajuste_muestras=AJUSTE,
        algoritmo='Yule-Walker con ridge relativo 1e-6, FIR causal; mitades remuestreadas por separado',
        entradas_filtro='solo señal y fs; ningún catálogo o etiqueta',
        seleccion='primeros 2.5 s para ajustar filtro y extraer rasgos; nunca segunda mitad',
        parametros='una configuración fija, sin búsqueda de orden ni ajuste a resultados',
        codigo=firma(Path(__file__)),extractor=firma(AQUI/'modelo_espectral.py'),muestra=firma(original))
    rp=SALIDA/'protocolo_preparacion.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==protocolo
    else:rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    solicitudes=[]
    for r in idx.itertuples():
        solicitudes.append(dict(archivo=r.archivo,registro=int(r.registro),sensor=r.sensor_falla,
            fs_original=int(r.fs),real=r.zona,rpm_nominal=float(r.rpm_nominal),primaria=True))
    for r in muestra[muestra.real.eq('N') & muestra.sensor.eq('FE')].itertuples():
        solicitudes.append(dict(archivo=r.archivo,registro=int(r.registro),sensor='FE',
            fs_original=int(r.fs_original),real='N',rpm_nominal=float(r.rpm),primaria=False))
    assert len(solicitudes)==135
    filas=[];bloques=[]
    for i,r in enumerate(solicitudes):
        mat=loadmat(AQUI.parent/'data2/CWRU'/r['archivo'])
        clave=F.clave_canal(mat,r['sensor'],r['registro']);x=np.ravel(mat[clave])
        huella=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        if r['primaria']:assert huella==anteriores.loc[r['archivo'],'sha256_senal']
        else:assert huella==muestra.loc[muestra.archivo.eq(r['archivo']) & muestra.sensor.eq('FE'),'sha256_senal'].iloc[0]
        rpm=F.rpm_registro(mat,r['registro'],nominal=r['rpm_nominal'])
        c={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r['sensor']].items()}
        suficiente=len(x)//2>=2.5*r['fs_original']
        info={};procesada=None
        if suficiente:
            y,info=preblanquear(x,r['fs_original']);z=rasgos_blancos(y,c)
            procesada=hashlib.sha256(y.astype('<f8').tobytes()).hexdigest()
        else:z=np.full((4,2021),np.nan)
        bloques.append(z)
        grupo='normal:'+r['archivo'] if r['real']=='N' else grupo_condicion(r['archivo']).split('@')[0]
        filas.append(dict(r,vista_uid=r['archivo']+'|'+r['sensor'],grupo=grupo,
            clave_canal=clave,catalogo=json.dumps(c,sort_keys=True),suficiente=suficiente,
            sha256_senal=huella,sha256_procesada=procesada,coef=json.dumps(info.get('coef')),
            media=info.get('media'),energia_relativa=info.get('energia_relativa')))
        print(f'Preblanqueo {i+1}/135: {r["archivo"]}|{r["sensor"]}',flush=True)
    d=pd.DataFrame(filas);z=np.asarray(bloques)
    assert d.vista_uid.is_unique and d.primaria.sum()==133 and z.shape==(135,4,2021)
    assert set(muestra.vista_uid).issubset(set(d.vista_uid))
    d.to_csv(SALIDA/'vistas.csv',index=False)
    np.savez_compressed(SALIDA/'rasgos.npz',x=z)
    print('OK: 133 registros más dos vistas FE normales de entrenamiento; sin aumentar los casos evaluados.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparar',action='store_true')
    args=parser.parse_args();comprobar()
    if args.preparar:preparar()
