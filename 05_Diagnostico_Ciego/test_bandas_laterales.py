"""Impide confundir la central con sus laterales y comprobar acceso al futuro."""
import numpy as np
from bandas_laterales import tripleta,rasgos_laterales
from modelo_ciego import F


if __name__=='__main__':
    f=np.arange(0.,2001.);s=np.ones_like(f);s[100]=10000
    sin=tripleta(f,s,100.,30.)
    s[70]=s[130]=1000
    con=tripleta(f,s,100.,30.)
    assert con[0]>sin[0]+20 and con[1]>sin[1]+20
    s=np.ones_like(f);s[1000]=1e9
    assert tripleta(f,s,1000.,12.)[0]<1, 'La central contaminó la medición lateral'
    rng=np.random.default_rng(58);fs=12000
    x=rng.normal(size=6*fs);c={k:v*30 for k,v in F.MULTIPLICADORES['DE'].items()}
    a=rasgos_laterales(x,fs,c)
    x[int(2.5*fs):]*=1000
    b=rasgos_laterales(x,fs,c)
    assert a.shape==(4,240) and np.array_equal(a,b)
    print('OK: laterales reales detectadas, central no duplicada como lateral, futuro excluido, todas las familias medidas.')
