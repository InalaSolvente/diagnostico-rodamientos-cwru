"""Ejecutar: python 01_MFD_Kurtogram_Tarea1/test_entropia.py"""
import numpy as np
import pandas as pd
import Fuentes as F


def comprobar():
    assert F.entropia_envolvente(np.zeros(1024)) == 1
    assert F.entropia_envolvente(np.ones(1024)) == 1
    t = np.arange(1024)
    z = (1 + .5 * np.cos(2 * np.pi * 16 * t / t.size)).astype(complex)
    assert F.entropia_envolvente(z) < 1e-10
    assert np.isclose(F.entropia_envolvente(z), F.entropia_envolvente(z * 100))
    tabla = pd.DataFrame(dict(l=[1, 1], m=[0, 1], K=[0., 10.], H=[0., 1.]))
    assert F.seleccionar_banda(tabla) == (1, 1)
    assert F.seleccionar_banda(tabla, .7) == (1, 1)
    assert F.seleccionar_banda(tabla, .5) == (1, 0)
    assert F.seleccionar_banda(tabla, .3) == (1, 0)
    tabla.K = 2.
    assert F.seleccionar_banda(tabla, .7) == (1, 0)
    for peso in (-.1, 1.1, np.nan):
        try:
            F.seleccionar_banda(tabla, peso)
        except ValueError:
            pass
        else:
            raise AssertionError("Peso inválido aceptado")
    x = np.random.default_rng(7).normal(size=1024)
    original = F.kurtogram(x, 12000, 2)
    combinado = F.kurtogram(x, 12000, 2, con_entropia=True)
    assert np.array_equal(original.K, combinado.K)
    assert combinado.H.between(0, 1).all()
    assert F.seleccionar_banda(original) == F.seleccionar_banda(combinado)
    print("OK: entropía, pesos, empates y compatibilidad")


if __name__ == "__main__":
    comprobar()
