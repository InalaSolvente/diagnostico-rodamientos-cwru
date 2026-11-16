"""Prueba reproducible del contrato sin frecuencia objetivo y de la búsqueda armónica."""
import inspect
import numpy as np
from modelo_ciego import inferir, buscar_periodicidad, buscar_catalogo, clasificar_evidencia, F
from evaluar_catalogo import dividir_desarrollo
import pandas as pd


if __name__ == "__main__":
    assert set(inspect.signature(inferir).parameters) == {"x", "fs", "frecuencias_fabricante", "L", "fmin", "fmax", "p", "estrategia"}
    f = np.arange(0., 4001.)
    s = np.ones_like(f)
    s[[100, 200, 300]] = [10000, 8000, 6000]
    r = buscar_periodicidad({(0, 0):(f, s)})
    assert r and abs(r[0]["frecuencia_hz"]-100) < 1
    catalogo = dict(BPFI=100., BPFO=137., BSF=83., FTF=17.)
    c = buscar_catalogo({(0, 0):(f, s)}, catalogo)
    assert c and c[0]["tipo"] == "IR"
    assert c == buscar_catalogo({(0, 0):(f, s)}, dict(reversed(list(catalogo.items()))))
    s[200] = s[300] = 1
    assert buscar_periodicidad({(0, 0):(f, s)}) == []
    assert buscar_catalogo({(0, 0):(f, s)}, catalogo, estrategia="conservador") == []
    assert buscar_catalogo({(0, 0):(f, s)}, catalogo, estrategia="dominante")[0]["tipo"] == "IR"
    evidencia = [dict(tipo="IR", familia="BPFI", armonicas=[1], score=100., categoria="P", control=True),
                 dict(tipo="OR", familia="BPFO", armonicas=[1], score=10., categoria="D", control=True)]
    assert clasificar_evidencia(evidencia, "sensible")[0]["tipo"] == "IR"
    assert clasificar_evidencia(evidencia, "dominante")[0]["tipo"] == "OR"
    assert clasificar_evidencia(evidencia, "sin_control") == []
    assert inferir(np.ones(4096), 12000)["estado"] == "sin_evidencia"
    t = np.arange(16384)/12000
    x = (1+.4*np.cos(2*np.pi*100*t)+.3*np.cos(2*np.pi*200*t)+.2*np.cos(2*np.pi*300*t))*np.cos(2*np.pi*2500*t)
    def prohibido(*args, **kwargs):
        raise AssertionError("Inferencia consultó frecuencia o diagnóstico objetivo")
    anterior = F.frecuencia_falla
    F.frecuencia_falla = prohibido
    try:
        a = inferir(x, 12000, L=3)
        b = inferir(x, 12000, L=3)
        con_catalogo = inferir(x, 12000, L=3, frecuencias_fabricante=catalogo)
    finally:
        F.frecuencia_falla = anterior
    assert a == b and a["candidatos"]
    assert abs(a["candidatos"][0]["frecuencia_hz"]-100) < 2
    assert con_catalogo["candidatos"][0]["tipo"] == "IR"
    idx = pd.DataFrame(dict(archivo=[f"{i}.mat" for i in range(12)], registro=range(12), carpeta=["12k_DE"]*6+["12k_FE"]*4+["Normal"]*2))
    particion = dividir_desarrollo(idx)
    assert len(particion[particion.particion == "desarrollo"]) == 6
    assert particion.equals(dividir_desarrollo(idx.sample(frac=1, random_state=3)))
    assert particion.registro.is_unique
    assert set(particion.particion) == {"desarrollo", "reserva"}
    idx.loc[11, "registro"] = idx.loc[10, "registro"]
    assert dividir_desarrollo(idx).archivo.is_unique
    try:
        inferir(np.ones(4096), 12000, frecuencias_fabricante={"BPFI":100})
    except ValueError:
        pass
    else:
        raise AssertionError("Aceptó catálogo con una sola hipótesis")
    for x, fs in ((np.ones(10), 12000), (np.array([np.nan]*4096), 12000), (np.ones(4096), 0)):
        try:
            inferir(x, fs)
        except ValueError:
            pass
        else:
            raise AssertionError("Entrada inválida aceptada")
    print("OK: inferencia sin fd/RPM/etiquetas, familia armónica, pico aislado y validación")
