"""Curvas de soporte armónico con desplazamiento común y control de giro.

Solo genera rasgos para identificar el tipo. No filtra ni altera el espectro
con que se decide ND/PD/D. No recibe tipo esperado, archivo ni etiquetas.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from ventanas_verificadas import prefijo
from modelo_espectral import extraer,FAMILIAS
from bandas_estables import envolvente,transformar,espectro,TRANSFORMACIONES
from modelo_ciego import F,validar_catalogo
from modelo_por_dominio import firma
from rasgos_fisicos import giro_catalogo

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_plantillas_coherentes'
DELTA=np.linspace(-.02,.02,81)
FS=12000


def soportes(e):
    return np.stack([e.max(axis=0),np.sort(e,axis=0)[-2:].mean(axis=0),
        np.minimum(e[0],e[1:].max(axis=0))])


def resumir(curvas):
    if curvas.shape!=(31,2,5,2,3,81) or not np.isfinite(curvas).all():
        raise ValueError('Banco de curvas incoherente')
    c=np.stack([curvas.max(axis=0),curvas.mean(axis=0)]).reshape(-1,81)
    peso=np.exp(c-c.max(axis=1,keepdims=True));peso/=peso.sum(axis=1,keepdims=True)
    mu=(peso*DELTA).sum(axis=1)
    ancho=np.sqrt((peso*(DELTA-mu[:,None])**2).sum(axis=1))/.02
    pico=np.where(np.ptp(c,axis=1)>1e-12,DELTA[c.argmax(axis=1)]/.02,0.)
    z=np.column_stack([c[:,40],c.max(axis=1),c.mean(axis=1),pico,mu/.02,ancho]).ravel()
    assert z.shape==(720,) and np.isfinite(z).all()
    return z


def medir(x,fs,catalogo):
    validar_catalogo(catalogo,FS)
    z=prefijo(x,fs);giro=giro_catalogo(catalogo)
    bases=[2*catalogo['BSF'] if k=='2BSF' else catalogo[k] for k in FAMILIAS]
    curvas=np.empty((31,2,5,2,3,81));nodo_i=0
    for l in range(5):
        for m in range(2**l):
            a=envolvente(z,FS,(l,m))
            for ti,transformacion in enumerate(TRANSFORMACIONES):
                f,s=espectro(transformar(a,transformacion),FS)
                s=np.convolve(s,[.25,.5,.25],mode='same')
                eps=max(float(s.max())*1e-15,np.finfo(float).tiny);df=f[1]-f[0]
                for fi,base in enumerate(bases):
                    frecuencias=base*np.arange(1,7)[:,None]*(1+DELTA)
                    potencia=np.interp(frecuencias.ravel(),f,s).reshape(6,81)
                    fondos=[]
                    for h in range(1,7):
                        objetivo=h*base
                        lo,hi=np.searchsorted(f,[objetivo*.98,objetivo*1.02])
                        inicio,fin=np.searchsorted(f,[max(0,objetivo-.3*base),objetivo+.3*base])
                        fondo=np.r_[s[inicio:lo],s[hi:fin]]
                        fondos.append(max(float(fondo.mean()) if len(fondo) else float(s.mean()),eps))
                    evidencia=np.log1p(potencia/np.asarray(fondos)[:,None])
                    distancia=abs(frecuencias-np.rint(frecuencias/giro)*giro)
                    especificidad=-np.expm1(-.5*(distancia/df)**2)
                    curvas[nodo_i,ti,fi,0]=soportes(evidencia)
                    curvas[nodo_i,ti,fi,1]=soportes(evidencia*especificidad)
            nodo_i+=1
    momentos=extraer(z,FS,catalogo)['globales'][:4]
    salida=np.r_[momentos,resumir(curvas)]
    assert salida.shape==(724,) and np.isfinite(salida).all()
    return salida


def comprobar():
    rng=np.random.default_rng(6203)
    c=rng.uniform(size=(31,2,5,2,3,81))
    assert np.allclose(resumir(c),resumir(c[rng.permutation(31)]))
    catalogo={k:30*v for k,v in F.MULTIPLICADORES['DE'].items()}
    for fs in (12000,48000):
        x=rng.normal(size=6*fs)
        a=medir(x,fs,catalogo)
        otra=x.copy();otra[len(x)//2:]=rng.normal(scale=4,size=len(x)//2)
        assert np.array_equal(a,medir(otra,fs,catalogo))
    print('OK: rasgos invariantes al orden de portadoras e independientes de futuro a 12/48 kHz.')


def preparar():
    SALIDA.mkdir(exist_ok=True)
    base_dir=AQUI/'resultados_grupos_por_defecto'
    base=pd.read_csv(base_dir/'comparacion_133.csv')
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    protocolo=dict(version='curvas_81_724_v1',
        rasgos='momentos4 + curvas de soporte simple/doble/fundamental, con y sin especificidad de giro, maximo/media sobre portadoras',
        frecuencia='mismo desplazamiento relativo para todas las armonicas de una familia; 81 puntos en +/-2%',
        resolucion='prefijo fijo 2.5 s a 12 kHz, periodograma Hann y suavizado fijo de un bin',
        giro='rasgo continuo por distancia a orden de giro en unidades de resolucion; no elimina picos del juez',
        bandas='31; envolvente amplitud y cuadrada; seis armonicas para TODAS las cinco familias',
        entrada='solo señal, fs y catalogo completo; ninguna etiqueta o identidad como rasgo',
        codigo=firma(Path(__file__)),base=firma(base_dir/'comparacion_133.csv'),
        extractor=firma(AQUI/'modelo_espectral.py'),ventanas=firma(AQUI/'ventanas_verificadas.py'),
        bandas_fuente=firma(AQUI/'bandas_estables.py'),giro_fuente=firma(AQUI/'rasgos_fisicos.py'))
    p=SALIDA/'protocolo_preparacion.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocolo
    else:p.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    filas=[];bloques=[]
    for i,r in enumerate(base.itertuples()):
        a=idx.loc[r.archivo];mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
        x=np.ravel(mat[F.clave_canal(mat,r.sensor,int(a.registro))])
        assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()==r.sha256_senal
        rpm=F.rpm_registro(mat,int(a.registro),nominal=float(a.rpm_nominal))
        c={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor].items()}
        bloques.append(medir(x,int(r.fs),c) if r.suficiente else np.full(724,np.nan))
        d=r._asdict();d.pop('Index',None);d.update(catalogo=json.dumps(c,sort_keys=True),registro=int(a.registro))
        filas.append(d)
        if (i+1)%10==0 or i==len(base)-1:print(f'Curvas {i+1}/133',flush=True)
    d=pd.DataFrame(filas);z=np.asarray(bloques)
    assert len(d)==133 and d.archivo.is_unique and z.shape==(133,724)
    d.to_csv(SALIDA/'muestra.csv',index=False)
    np.savez_compressed(SALIDA/'rasgos.npz',x=z)
    print('OK: 133 registros, 724 rasgos; sin identidad de portadora ni información de verificación.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparar',action='store_true')
    args=parser.parse_args();comprobar()
    if args.preparar:preparar()
