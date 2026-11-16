"""Subárbol físicamente equivalente a 12 kHz/L4, sin remuestrear ni reentrenar."""
from pathlib import Path
import hashlib
import numpy as np
from modelo_espectral import evidencia_armonica,FAMILIAS
from modelo_ciego import validar_catalogo
from predictor_espectral import MODELO


def geometria(fs):
    if not np.isfinite(fs) or fs<12000:
        raise ValueError("Se requiere fs >= 12000 Hz")
    desplazamiento=int(round(np.log2(fs/12000)))
    if not np.isclose(fs,12000*2**desplazamiento):
        raise ValueError("Se requiere una razón potencia de dos respecto a 12 kHz")
    nodos=[(l+desplazamiento,m) for l in range(5) for m in range(2**l)]
    return nodos,4+desplazamiento


def rasgos_alineados(x,fs,catalogo):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)<1024 or not np.isfinite(x).all():
        raise ValueError("Señal inválida")
    validar_catalogo(catalogo,fs)
    nodos,L=geometria(fs)
    x=x-x.mean(); rms=np.sqrt(np.mean(x*x)); x=x/max(rms,np.finfo(float).tiny)
    n=len(x); X=np.fft.fft(x); f_abs=abs(np.fft.fftfreq(n,1/fs)); f=np.fft.rfftfreq(n,1/fs)
    hilbert=np.zeros(n);hilbert[0]=1;hilbert[1:(n+1)//2]=2
    if n%2==0:hilbert[n//2]=1
    frecuencias=[catalogo[k] if k!="2BSF" else 2*catalogo["BSF"] for k in FAMILIAS]
    rasgos=[];estadisticas=[]
    for l,m in nodos:
        ancho=fs/2**(l+1)
        # En 12 kHz incluir Nyquist como en el extractor original. En 48 kHz
        # 6 kHz es un borde interior: no se incluye en dos bandas contiguas.
        ultimo=m==2**l-1
        mascara=(f_abs>=m*ancho)&((f_abs<=(m+1)*ancho) if ultimo else (f_abs<(m+1)*ancho))
        z=np.fft.ifft(X*mascara*hilbert)
        e=abs(z-z.mean())**2;energia=float(e.mean())
        k=float(np.mean(e**2)/energia**2-2) if energia>1e-20 else 0.
        a=abs(z);s=abs(np.fft.rfft(a-a.mean()))**2/n**2
        p=s[1:]/max(float(s[1:].sum()),np.finfo(float).tiny);pos=p[p>0]
        h=-float(np.sum(pos*np.log(pos)))/np.log(max(len(p),2)) if len(pos) else 1.
        rasgos.append(evidencia_armonica(f,s,frecuencias))
        estadisticas.extend([np.log10(energia+1e-15),np.log1p(max(k,0)),h])
    resultado=np.r_[np.asarray(rasgos).ravel(),estadisticas]
    if not np.isfinite(resultado).all(): raise ValueError("Rasgos no finitos")
    return resultado,nodos,L


def predecir_alineado(x,fs,catalogo,*,modelo=MODELO):
    rasgos,nodos,L=rasgos_alineados(x,fs,catalogo)
    with np.load(modelo,allow_pickle=False) as m:
        firma=hashlib.sha256((Path(__file__).parent/"modelo_espectral.py").read_bytes()).hexdigest()
        if str(m["extractor_sha256"])!=firma or str(m["banco"])!="arbol":
            raise ValueError("Modelo incompatible con el experimento de alineación")
        if rasgos.shape!=m["media"].shape: raise ValueError("Dimensión incompatible")
        scores=((rasgos-m["media"])/m["escala"])@m["coef"].T+m["intercepto"]
        tipo=str(m["clases"][int(np.argmax(scores))])
    return dict(tipo=tipo,nodos=nodos,L=L,ancho_final=fs/2**(L+1),rasgos=rasgos)
