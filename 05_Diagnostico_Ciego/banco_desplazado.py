"""Banco físico de 57 bandas: diádicas y desplazadas medio ancho, sin etiquetas."""
import numpy as np
from alinear_bandas import geometria
from bandas_estables import validar_senal, envolvente, transformar, espectro, margenes, TRANSFORMACIONES
from doble_banda_fijada import disjuntas, intervalo
from criterio_usuario import evaluar, ORDEN


def nodos_57(fs):
    nodos, profundidad = geometria(fs)
    inicio = profundidad - 4
    desplazados = [(inicio + nivel, m + .5)
                   for nivel in range(1, 5) for m in range(2**nivel - 1)]
    return nodos + desplazados


def seleccionar_57(x, fs, catalogo, tipo):
    x = validar_senal(x, fs, catalogo)
    nodos = nodos_57(fs)
    familias = {'IR': [('BPFI', catalogo['BPFI'])], 'OR': [('BPFO', catalogo['BPFO'])],
                'B': [('BSF', catalogo['BSF']), ('2BSF', 2*catalogo['BSF'])]}
    if tipo not in familias:
        raise ValueError('Se requiere una hipótesis de falla')
    p = .01 / (len(nodos) * 5 * 6 * len(TRANSFORMACIONES))
    candidatos = []
    for nodo in nodos:
        a = envolvente(x, fs, nodo)
        for transformacion in TRANSFORMACIONES:
            b = transformar(a, transformacion)
            f, s = espectro(b, fs)
            segmentos = [espectro(v, fs) for v in np.array_split(b, 3)]
            for familia, base in familias[tipo]:
                q = evaluar(f, s, base, H=6, p=p)
                partes = [evaluar(fp, sp, base, H=6, p=p) for fp, sp in segmentos]
                md = [margenes(v)[0] for v in partes]
                mp = [margenes(v)[1] for v in partes]
                robusto = float(np.quantile(md if q['grado'] == 'D' else mp, .25))
                clave = (ORDEN[q['grado']], robusto, len(q['identificadas']), margenes(q)[0])
                candidatos.append((clave, dict(grado=q['grado'], nodo=list(nodo), familia=familia,
                    frecuencia_base=float(base), transformacion=transformacion, p_prueba=p,
                    margen_d_segmentos=md, margen_p_segmentos=mp, evidencia=q)))
    candidatos.sort(key=lambda v: v[0], reverse=True)
    primera = candidatos[0][1]
    segunda = next((s for _, s in candidatos if disjuntas(primera['nodo'], s['nodo'], fs)), None)
    return [primera] + ([segunda] if segunda is not None else [])


def comprobar():
    a, b = nodos_57(12000), nodos_57(48000)
    assert len(a) == len(b) == len(set(a)) == 57
    assert [intervalo(n, 12000) for n in a] == [intervalo(n, 48000) for n in b]
    assert all(0 <= intervalo(n, 12000)[0] < intervalo(n, 12000)[1] <= 6000 for n in a)
    t = np.arange(30000) / 12000
    x = (1 + .4*np.cos(2*np.pi*100*t) + .2*np.cos(2*np.pi*200*t))*np.cos(2*np.pi*2500*t)
    s = seleccionar_57(x, 12000, dict(BPFI=100., BPFO=137., BSF=83., FTF=17.), 'IR')
    assert s[0]['grado'] == 'D' and all(q['p_prueba'] == .01/(57*5*6*2) for q in s)
    assert len(s) == 1 or disjuntas(s[0]['nodo'], s[1]['nodo'], 12000)


if __name__ == '__main__':
    comprobar()
    print('OK: 57 bandas, misma geometria fisica 12/48 kHz, correccion multiplicidad y dos ramas disjuntas')
