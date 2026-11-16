import numpy as np
from criterio_usuario import evaluar,comparar


if __name__=="__main__":
    f=np.arange(0.,701.)
    def espectro(picos):
        s=np.ones_like(f)
        for frecuencia,potencia in picos.items(): s[frecuencia]=potencia
        return s
    assert evaluar(f,espectro({100:1000,200:500}),100)["grado"]=="D"
    assert evaluar(f,espectro({30:2000,100:1000,200:500}),100)["grado"]=="PD"
    assert evaluar(f,espectro({100:1000}),100)["grado"]=="ND"
    assert evaluar(f,espectro({100:5,200:5}),100)["grado"]=="ND"
    assert evaluar(f,espectro({0:100000,100:1000,200:500}),100)["grado"]=="D"
    assert evaluar(f,espectro({30:1000,100:1000,200:500}),100)["grado"]=="D"
    derecha=espectro({100:1000,200:2000})
    assert evaluar(f,derecha,100)["grado"]=="D"
    falta=espectro({30:2000,200:1000,300:500})
    assert evaluar(f,falta,100)["grado"]=="PD"
    assert evaluar(f,espectro({30:2000,100:1000}),100)["grado"]=="PD"
    assert evaluar(f,espectro({30:2000,200:1000}),100)["grado"]=="PD"
    assert evaluar(f,espectro({200:1000}),100)["grado"]=="ND"
    assert evaluar(f,espectro({30:10,200:1000}),100)["grado"]=="ND"
    assert evaluar(f,espectro({150:2000,200:1000}),100)["grado"]=="PD"
    assert evaluar(f,espectro({102:1000,204:500}),100)["grado"]=="D"
    assert comparar("ND","PD",False)==("tipo_incorrecto",False)
    assert comparar("ND","ND",False)==("ND",True)
    assert comparar("D","PD",True)==("PD",False)
    assert comparar("PD","D",True)==("D",True)
    from evaluar_no_retroceso import calificar
    t=np.arange(12000)/12000
    x=(1+.4*np.cos(2*np.pi*100*t)+.2*np.cos(2*np.pi*200*t))*np.cos(2*np.pi*2500*t)
    calidad=calificar(x,12000,dict(BPFI=100.,BPFO=137.,BSF=83.,FTF=17.),"IR",criterio="usuario")
    assert calidad["grado"]=="D" and calidad["evidencia"]["primera_identificada"]
    try:
        evaluar(f,espectro({100:1000}),100,H=1)
    except ValueError: pass
    else: raise AssertionError("H inválido aceptado")
    print("OK: dos armonicas, primera, izquierda/derecha, DC, empates, tolerancia y salida sin oraculo")
