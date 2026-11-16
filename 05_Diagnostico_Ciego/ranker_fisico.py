"""Puntaje compartido con monotonía física: soporte positivo, desajuste negativo."""
import numpy as np
from scipy.optimize import minimize
from ranker_plantillas import candidatos,logits,perdida_gradiente,CLASES


def rasgos(x):
    z=candidatos(x).copy().reshape(-1,4,24,6)
    z[:,:,:,3:5]=abs(z[:,:,:,3:5])
    return z.reshape(-1,4,144)


class RankerFisico:
    def fit(self,x,y):
        z=rasgos(x);y=np.asarray(y)
        if len(z)!=len(y) or set(y)!=set(CLASES):raise ValueError('IR/OR/B requeridos')
        self.media_=z.mean(axis=(0,1));self.escala_=z.std(axis=(0,1))
        self.escala_[self.escala_<1e-12]=1.
        z=(z-self.media_)/self.escala_
        etiquetas=np.array([int(np.flatnonzero(CLASES==t)[0]) for t in y])
        cuentas=np.bincount(etiquetas,minlength=3);peso=len(y)/(3*cuentas[etiquetas])
        bounds=[(0,None) if i%6<3 else (None,0) for i in range(144)]
        inicio=np.zeros(144)
        r=minimize(perdida_gradiente,inicio,args=(z,etiquetas,peso),jac=True,bounds=bounds,
            method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-10,gtol=1e-6))
        if not r.success:raise RuntimeError('Optimizador no convergió: '+str(r.message))
        self.coef_=r.x;self.classes_=CLASES.copy();self.iteraciones_=int(r.nit)
        assert np.isfinite(self.coef_).all()
        c=self.coef_.reshape(24,6)
        assert (c[:,:3]>=0).all() and (c[:,3:]<=0).all()
        assert r.fun<=perdida_gradiente(inicio,z,etiquetas,peso)[0]+1e-9
        return self

    def decision_function(self,x):
        return logits(self.coef_,(rasgos(x)-self.media_)/self.escala_)

    def predict(self,x):
        return self.classes_[self.decision_function(x).argmax(axis=1)]


def comprobar():
    rng=np.random.default_rng(105)
    x=rng.normal(size=(12,724));y=np.tile(CLASES,4)
    m=RankerFisico().fit(x,y);z=(rasgos(x)-m.media_)/m.escala_
    original=logits(m.coef_,z)
    fuerza=z.copy();fuerza[:,0,0]+=1
    assert (logits(m.coef_,fuerza)[:,0]>=original[:,0]-1e-12).all()
    desajuste=z.copy();desajuste[:,0,3]+=1
    assert (logits(m.coef_,desajuste)[:,0]<=original[:,0]+1e-12).all()
    assert np.allclose(logits(m.coef_,z[:,[1,0,2,3]]),original[:,[1,0,2]])
    assert np.isfinite(m.decision_function(x)).all()
    print('OK: monotonía de soporte/desajuste, simetría IR/OR y convergencia verificadas.')


if __name__=='__main__':comprobar()
