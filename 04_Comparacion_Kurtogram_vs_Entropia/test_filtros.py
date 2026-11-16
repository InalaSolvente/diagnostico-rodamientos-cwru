"""Ejecutar con Python: comprueba filtrado y selección posterior por K."""
import numpy as np
import pandas as pd
import compara_entropia as E


if __name__ == "__main__":
    t = pd.DataFrame(dict(l=[2]*4, m=[0, 1, 2, 3], H=[.1, .2, .3, .4], K=[1., 3., 2., 9.]))
    for fraccion, esperado, cantidad in ((.25, (2, 0), 1), (.5, (2, 1), 2),
                                        (.75, (2, 1), 3), (1., (2, 3), 4)):
        candidatos = E.candidatos_entropia(t, fraccion)
        assert len(candidatos) == cantidad
        assert E.F.seleccionar_banda(candidatos) == esperado
    assert len(E.candidatos_entropia(t.iloc[:3], .5)) == 2
    t.H = .5
    assert E.candidatos_entropia(t, .5).index.tolist() == [0, 1]
    t.K = 1.
    assert E.F.seleccionar_banda(E.candidatos_entropia(t, 1.)) == (2, 0)
    for fraccion in (0., -.1, 1.1, np.nan):
        try:
            E.candidatos_entropia(t, fraccion)
        except ValueError:
            pass
        else:
            raise AssertionError("Fracción inválida aceptada")
    antes = pd.Series(["N", "P", "N", "D", "P", "D"])
    ahora = pd.Series(["P", "D", "D", "N", "P", "P"])
    matriz = E.matriz_transiciones(antes, ahora)
    assert matriz.to_numpy().sum() == 6
    assert matriz.loc["N", "P"] == 1 and matriz.loc["N", "D"] == 1
    assert matriz.loc["P", "D"] == 1 and matriz.loc["D", "N"] == 1
    assert matriz.loc["D", "P"] == 1 and matriz.loc["P", "P"] == 1
    assert matriz.loc["N", "N"] == 0
    assert matriz.sum(axis=1).to_dict() == antes.value_counts().to_dict()
    assert matriz.sum(axis=0).to_dict() == ahora.value_counts().to_dict()
    print("OK: filtros, redondeo, empates, selección y transiciones")
