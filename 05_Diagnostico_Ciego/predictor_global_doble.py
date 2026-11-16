"""Inferencia experimental: señal, fs y catálogo; pesos entrenados solo en desarrollo.

Exportar no valida el modelo: las métricas proceden de los pliegues por condición,
nunca de volver a predecir el entrenamiento con estos pesos congelados.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ventanas_verificadas import rasgos
from ensayar_globales_arbol import globales
from modelo_por_dominio import crear,dominio_catalogo,firma
from modelo_jerarquico import aplicar
from bandas_estables import validar_senal
from control_ordenes_giro import elegir
from doble_banda_fijada import seleccionar_dos,verificar_dos
from banco_63 import seleccionar_63
from ensayar_pd_persistente import elegir as elegir_pd

AQUI = Path(__file__).parent
MODELO = AQUI/'resultados_globales_arbol/modelo_tipo.npz'
DEPENDENCIAS = ('modelo_espectral.py','ventanas_verificadas.py','ensayar_globales_arbol.py',
    'control_ordenes_giro.py','doble_banda_fijada.py','banco_63.py','ensayar_pd_persistente.py')


def exportar():
    fuente = AQUI/'resultados_globales_arbol'
    casos = pd.read_csv(fuente/'tipos_133.csv')
    meta = pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    meta = meta[meta.real.ne('N')]
    posiciones = casos.reset_index().set_index('archivo')['index'].reindex(meta.archivo).to_numpy()
    assert len(posiciones) == 52 and not pd.isna(posiciones).any()
    with np.load(fuente/'rasgos.npz',allow_pickle=False) as z:
        x = z['x'][posiciones].reshape(-1,2021)
    y = np.repeat(meta.real.to_numpy(),4)
    with threadpool_limits(limits=1):
        clf = crear().fit(x,y)
    escala,lda = clf.steps[0][1],clf.steps[1][1]
    hashes = {n:firma(AQUI/n) for n in DEPENDENCIAS}
    np.savez_compressed(MODELO,media=escala.mean_,escala=escala.scale_,coef=lda.coef_,
        intercepto=lda.intercept_,clases=lda.classes_,archivos=np.array(list(hashes)),
        hashes=np.array(list(hashes.values())),muestra=firma(AQUI/'resultados_dominios_desarrollo/muestra.csv'))
    with np.load(MODELO,allow_pickle=False) as m:
        reconstruido=((x-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
        assert np.allclose(reconstruido,clf.decision_function(x))
        assert np.array_equal(m['clases'][reconstruido.argmax(axis=1)],clf.predict(x))
    print('OK: pesos congelados de 52 fallas de desarrollo; serialización verificada. No son métricas OOF.')


def predecir(x, fs, catalogo, *, banco=31):
    if banco not in (31,63):
        raise ValueError('Banco fijo permitido: 31 o 63')
    x = validar_senal(x,fs,catalogo)
    if len(x)//2 < 2.5*fs:
        raise ValueError('Se requieren al menos 5 segundos')
    base = rasgos(x,fs,catalogo)
    dominio = dominio_catalogo(catalogo)
    gate = aplicar(base.mean(axis=0),AQUI/'resultados_por_dominio'/f'modelo_{dominio}.npz')
    if gate == 'N':
        return dict(tipo='N',grado='ND',tipo_base='N',selecciones=[],alternativas_seleccion={},ramas=[])
    with np.load(MODELO,allow_pickle=False) as m:
        for archivo,h in zip(m['archivos'],m['hashes']):
            if firma(AQUI/str(archivo)) != str(h):
                raise ValueError('Cambió una dependencia; exportar y reevaluar el modelo')
        z = np.column_stack([base,globales(x,fs,catalogo)]).mean(axis=0)
        scores=((z-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
        tipo_base=str(m['clases'][scores.argmax()])
    mitad=len(x)//2
    pares={t:seleccionar_dos(x[:mitad],fs,catalogo,t) for t in ('IR','OR','B')}
    alternativas={t:s[0] for t,s in pares.items()}
    tipo_d,motivo=elegir(tipo_base,alternativas,catalogo)
    tipo=elegir_pd(tipo_d,alternativas)
    selecciones=pares[tipo] if banco == 31 else seleccionar_63(x[:mitad],fs,catalogo,tipo)
    v=verificar_dos(x[mitad:],fs,catalogo,selecciones)
    return dict(tipo=tipo,tipo_base=tipo_base,tipo_control_d=tipo_d,selecciones=selecciones,
        alternativas_seleccion=alternativas,motivo=motivo,**v)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exportar',action='store_true',required=True)
    parser.parse_args();exportar()
