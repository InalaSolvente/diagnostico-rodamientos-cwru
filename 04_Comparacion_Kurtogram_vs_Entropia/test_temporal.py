"""Prueba ejecutable: medianas, resistencia a una ventana atípica y validación."""
import numpy as np
import compara_entropia as E


if __name__ == "__main__":
    rng = np.random.default_rng(8)
    x = rng.normal(size=1024)
    base = E.F.kurtogram(x, 12000, 3, con_entropia=True)
    repetida = np.tile(x, 5)
    t = E.kurtogram_temporal(repetida, 12000, 3)
    assert np.allclose(t[["K", "H"]], base[["K", "H"]])
    assert np.allclose(t[["K_MAD", "H_MAD"]], 0)
    repetida[:1024] = 0
    repetida[512] = 1000
    robusta = E.kurtogram_temporal(repetida, 12000, 3)
    assert np.allclose(robusta[["K", "H"]], base[["K", "H"]])
    assert E.F.seleccionar_banda(E.candidatos_entropia(robusta, .75)) == E.F.seleccionar_banda(E.candidatos_entropia(base, .75))
    assert np.allclose(E.kurtogram_temporal(np.r_[repetida, [99, 99]], 12000, 3)[["K", "H"]], robusta[["K", "H"]])
    for senal, ventanas in ((x, 2), (x[:10], 5), (np.array([np.nan]*1024), 5)):
        try:
            E.kurtogram_temporal(senal, 12000, 3, ventanas)
        except ValueError:
            pass
        else:
            raise AssertionError("Entrada inválida aceptada")
    print("OK: medianas temporales, ventana atípica, restos y validación")
