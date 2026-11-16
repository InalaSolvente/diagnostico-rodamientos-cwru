"""Un mismo puntaje aprendido para las familias IR/OR/BSF/2BSF.

El tipo B admite dos representaciones mediante log-media-exp. No hay pesos
específicos para la identidad IR/OR ni información de verificación.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

CLASES=np.array(['IR','OR','B'])


def candidatos(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=724 or not np.isfinite(x).all():
        raise ValueError('Se requieren 724 rasgos coherentes finitos')
    # Orden del preparador: pool, transformación, familia, especificidad,
    # soporte, resumen. Cada familia obtiene exactamente las mismas medidas.
    z=x[:,4:].reshape(-1,2,2,5,2,3,6).transpose(0,3,1,2,4,5,6).reshape(-1,5,144)
    return z[:,:4]


def logits(w,z):
    s=z@w
    return np.column_stack([s[:,0],s[:,1],logsumexp(s[:,2:],axis=1)-np.log(2.)])


def perdida_gradiente(w,z,y,peso):
    s=z@w
    sb=logsumexp(s[:,2:],axis=1)
    l=np.column_stack([s[:,0],s[:,1],sb-np.log(2.)])
    normalizador=logsumexp(l,axis=1)
    perdida=float(np.sum(peso*(normalizador-l[np.arange(len(y)),y]))+.5*(w@w))
    residual=np.exp(l-normalizador[:,None]);residual[np.arange(len(y)),y]-=1
    residual*=peso[:,None]
    mezcla=np.exp(s[:,2:]-sb[:,None])
    zb=(mezcla[:,:,None]*z[:,2:]).sum(axis=1)
    gradiente=(residual[:,0,None]*z[:,0]+residual[:,1,None]*z[:,1]+residual[:,2,None]*zb).sum(axis=0)+w
    return perdida,gradiente


class RankerPlantillas:
    def fit(self,x,y):
        z=candidatos(x);y=np.asarray(y)
        if len(z)!=len(y) or set(y)!=set(CLASES):
            raise ValueError('Entrenamiento con IR/OR/B requerido')
        self.media_=z.mean(axis=(0,1));self.escala_=z.std(axis=(0,1))
        self.escala_[self.escala_<1e-12]=1.
        z=(z-self.media_)/self.escala_
        etiquetas=np.array([int(np.flatnonzero(CLASES==t)[0]) for t in y])
        cuentas=np.bincount(etiquetas,minlength=3)
        pesos=len(y)/(3*cuentas[etiquetas])
        inicial=np.zeros(z.shape[-1])
        r=minimize(perdida_gradiente,inicial,args=(z,etiquetas,pesos),jac=True,
            method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-10,gtol=1e-6))
        if not r.success:raise RuntimeError('Optimizador no convergió: '+str(r.message))
        assert np.isfinite(r.x).all() and r.fun<=perdida_gradiente(inicial,z,etiquetas,pesos)[0]+1e-9
        self.coef_=r.x;self.classes_=CLASES.copy();self.iteraciones_=int(r.nit)
        return self

    def decision_function(self,x):
        z=(candidatos(x)-self.media_)/self.escala_
        return logits(self.coef_,z)

    def predict(self,x):
        return self.classes_[self.decision_function(x).argmax(axis=1)]


def comprobar():
    rng=np.random.default_rng(100)
    z=rng.normal(size=(6,4,5));w=rng.normal(scale=.2,size=5)
    y=np.array([0,1,2,0,1,2]);peso=np.ones(6)
    valor,g=perdida_gradiente(w,z,y,peso)
    numerico=[]
    for i in range(len(w)):
        d=np.zeros(len(w));d[i]=1e-6
        numerico.append((perdida_gradiente(w+d,z,y,peso)[0]-perdida_gradiente(w-d,z,y,peso)[0])/2e-6)
    assert np.allclose(g,numerico,atol=1e-6,rtol=1e-5)
    cambiado=z[:,[1,0,2,3]];etiquetas=np.array([1,0,2,1,0,2])
    otro,og=perdida_gradiente(w,cambiado,etiquetas,peso)
    assert np.allclose(valor,otro) and np.allclose(g,og)
    assert np.allclose(logits(w,z)[:,[1,0,2]],logits(w,cambiado))
    x=rng.normal(size=(9,724));m=RankerPlantillas().fit(x,np.tile(CLASES,3))
    assert m.predict(x).shape==(9,) and np.isfinite(m.decision_function(x)).all()
    print('OK: gradiente numérico, simetría IR/OR y ajuste reproducible del puntaje compartido.')


if __name__=='__main__':comprobar()
