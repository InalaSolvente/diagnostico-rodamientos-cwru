import inspect
import numpy as np
from modelo_espectral import extraer,evidencia_armonica


if __name__=="__main__":
    assert set(inspect.signature(extraer).parameters)=={"x","fs","frecuencias_fabricante","L"}
    rng=np.random.default_rng(3)
    x=rng.normal(size=4096)
    c=dict(BPFI=160.,BPFO=106.,BSF=70.,FTF=12.)
    a=extraer(x,12000,c,L=2)
    b=extraer(x,12000,dict(reversed(list(c.items()))),L=2)
    assert all(np.allclose(a[k],b[k]) for k in a)
    assert all(np.isfinite(v).all() for v in extraer(np.ones(4096),12000,c,L=2).values())
    f=np.arange(2001.); s=np.ones_like(f); s[100]=100
    r=evidencia_armonica(f,s,[100])
    assert len(r)==12 and r[0]>15 and abs(r[2])<1e-10
    print("OK: contrato sin fd, catálogo independiente de orden, constante finita y armónicas")
