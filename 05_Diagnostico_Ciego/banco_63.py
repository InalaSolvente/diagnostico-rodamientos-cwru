"""Un nivel físico adicional; mismo juez y corrección por las 63 bandas."""
import numpy as np
from bandas_estables import (validar_senal,envolvente,transformar,espectro,margenes,TRANSFORMACIONES)
from alinear_bandas import geometria
from doble_banda_fijada import disjuntas
from criterio_usuario import evaluar, ORDEN


def nodos_63(fs):
    nodos,profundidad = geometria(fs)
    return nodos+[(profundidad+1,m) for m in range(32)]


def seleccionar_63(x, fs, catalogo, tipo):
    x = validar_senal(x,fs,catalogo); nodos = nodos_63(fs)
    familias = {'IR':[('BPFI',catalogo['BPFI'])], 'OR':[('BPFO',catalogo['BPFO'])],
        'B':[('BSF',catalogo['BSF']),('2BSF',2*catalogo['BSF'])]}
    if tipo not in familias:
        raise ValueError('Se requiere una clase de falla predicha')
    p = .01/(len(nodos)*5*6*len(TRANSFORMACIONES))
    candidatos = []
    for nodo in nodos:
        a = envolvente(x,fs,nodo)
        for transformacion in TRANSFORMACIONES:
            b = transformar(a,transformacion); f,s = espectro(b,fs)
            segmentos = [espectro(v,fs) for v in np.array_split(b,3)]
            for familia,base in familias[tipo]:
                q = evaluar(f,s,base,H=6,p=p)
                partes = [evaluar(fp,sp,base,H=6,p=p) for fp,sp in segmentos]
                md = [margenes(v)[0] for v in partes]; mp = [margenes(v)[1] for v in partes]
                robusto = float(np.quantile(md if q['grado'] == 'D' else mp,.25))
                clave = (ORDEN[q['grado']],robusto,len(q['identificadas']),margenes(q)[0])
                seleccion = dict(grado=q['grado'],nodo=list(nodo),familia=familia,frecuencia_base=float(base),
                    transformacion=transformacion,p_prueba=p,margen_d_segmentos=md,
                    margen_p_segmentos=mp,evidencia=q)
                candidatos.append((clave,seleccion))
    candidatos.sort(key=lambda v:v[0],reverse=True)
    primera = candidatos[0][1]
    segunda = next((s for _,s in candidatos if disjuntas(primera['nodo'],s['nodo'],fs)),None)
    return [primera]+([segunda] if segunda is not None else [])


def comprobar():
    from doble_banda_fijada import intervalo
    a = nodos_63(12000); b = nodos_63(48000)
    assert len(a) == len(b) == 63 and len(set(a)) == 63
    assert [intervalo(n,12000) for n in a] == [intervalo(n,48000) for n in b]
    assert intervalo(a[-1],12000) == (5812.5,6000.)
    t = np.arange(30000)/12000
    x = (1+.3*np.cos(2*np.pi*70*t))*np.cos(2*np.pi*2500*t)
    s = seleccionar_63(x,12000,dict(BPFI=162.,BPFO=107.,BSF=70.,FTF=12.),'B')
    assert 1 <= len(s) <= 2 and s[0]['familia'] in ('BSF','2BSF')
    assert all(q['p_prueba'] == .01/(63*5*6*2) for q in s)


if __name__ == '__main__':
    comprobar()
    print('OK: 63 bandas únicas; mismo ancho físico a 12 y 48 kHz.')
