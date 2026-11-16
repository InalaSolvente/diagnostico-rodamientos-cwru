"""Predictor experimental con señal, fs y catálogo completo como únicas entradas.

Los pesos históricos congelados se usan en inferencia. Las cifras OOF del
experimento provienen del evaluador por condición, no de aplicar estos pesos
sobre los registros con que se entrenaron.
"""
from pathlib import Path
import numpy as np
from ventanas_verificadas import rasgos
from modelo_por_dominio import dominio_catalogo
from modelo_jerarquico import aplicar
from bandas_estables import validar_senal, seleccionar, verificar
from correccion_d_persistente import elegir

AQUI = Path(__file__).parent


def predecir(x, fs, catalogo):
    x = validar_senal(x, fs, catalogo)
    if len(x)//2 < 2.5*fs:
        raise ValueError('Se requieren al menos 5 segundos')
    z = rasgos(x, fs, catalogo).mean(axis=0)
    dominio = dominio_catalogo(catalogo)
    gate = aplicar(z,AQUI/'resultados_por_dominio'/f'modelo_{dominio}.npz')
    if gate == 'N':
        return dict(tipo='N',grado='ND',tipo_base='N',seleccion=None,
            alternativas_seleccion={},verificacion=None,motivo='detector_normal')
    base = aplicar(z,AQUI/'resultados_jerarquico/modelo_tipo.npz')
    mitad = len(x)//2
    alternativas = {t:seleccionar(x[:mitad],fs,catalogo,t) for t in ('IR','OR','B')}
    tipo,motivo = elegir(base,alternativas)
    seleccion = alternativas[tipo]
    v = verificar(x[mitad:],fs,catalogo,seleccion)
    return dict(tipo=tipo,grado=v['grado'],tipo_base=base,seleccion=seleccion,
        alternativas_seleccion=alternativas,verificacion=v,motivo=motivo)
