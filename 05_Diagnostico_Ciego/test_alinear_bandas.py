import numpy as np
from alinear_bandas import geometria,rasgos_alineados
from modelo_espectral import extraer


if __name__=="__main__":
    n12,L12=geometria(12000);n48,L48=geometria(48000)
    assert L12==4 and L48==6 and len(n12)==len(n48)==31
    for (l,m),(ll,mm) in zip(n12,n48):
        assert 12000/2**(l+1)==48000/2**(ll+1)
        assert m==mm and (m+1)*12000/2**(l+1)<=6000
    assert 48000/2**(L48+1)==375
    c=dict(BPFI=160.,BPFO=106.,BSF=70.,FTF=12.)
    x=np.random.default_rng(6).normal(size=4096)
    a,_,_=rasgos_alineados(x,12000,c)
    assert np.allclose(a,extraer(x,12000,c)["arbol"])
    # Con muestras coherentes, el tono de 3 kHz cae en el mismo intervalo físico.
    for fs in (12000,48000):
        nodos,L=geometria(fs)
        bandas=[(m*fs/2**(l+1),(m+1)*fs/2**(l+1)) for l,m in nodos if l==L]
        assert bandas==[(i*375,(i+1)*375) for i in range(16)]
    try:geometria(44100)
    except ValueError:pass
    else:raise AssertionError("Aceptó una alineación diádica no exacta")
    print("OK: L48=6, 375 Hz, intervalos físicos iguales y rasgos 12k idénticos")
