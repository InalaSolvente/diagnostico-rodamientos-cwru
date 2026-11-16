"""Verifica que la corrección rechaza un pico marginal y no penaliza una sola banda."""
import numpy as np
import doble_banda as B


if __name__ == "__main__":
    f = np.linspace(0, 40, 401)
    s = np.ones_like(f)
    M = int((np.abs(f - 10) <= .02 * 10).sum())
    umbral = lambda p: -np.log(1 - (1-p)**(1/M))
    s[np.argmin(abs(f-10))] = (umbral(.01) + umbral(.005))/2
    una = {(1, 0): (f, s)}
    dos = {**una, (1, 1): (f, np.ones_like(s))}
    assert B.evaluar_bandas(una, 10, 1)[0] == "D"
    assert B.evaluar_bandas(una, 10, 1)[2] == .01
    assert B.evaluar_bandas(dos, 10, 1, control=False)[0] == "D"
    assert B.evaluar_bandas(dos, 10, 1)[0] == "N"
    assert B.evaluar_bandas(dos, 10, 1)[2] == .005
    s[np.argmin(abs(f-10))] = 100
    assert B.evaluar_bandas(dos, 10, 1)[0] == "D"
    assert B.evaluar_bandas(dict(reversed(list(dos.items()))), 10, 1)[0] == "D"
    for espectros, p in (({}, .01), (una, 0), (una, np.nan)):
        try:
            B.evaluar_bandas(espectros, 10, 1, p=p)
        except ValueError:
            pass
        else:
            raise AssertionError("Entrada inválida aceptada")
    print("OK: unión, corrección, pico marginal/fuerte, banda única y validación")
