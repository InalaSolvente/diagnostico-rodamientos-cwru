"""Verifica denominadores y diferencias de proporciones del resumen compartido."""
import tempfile
import pandas as pd
import compara_entropia as E


if __name__ == "__main__":
    df = pd.DataFrame({"bench": ["Y", "N"], "prueba|fd": ["D", "P"],
                       "prueba|senuelo0": ["D", "N"], "prueba|senuelo1": ["N", "N"]})
    with tempfile.TemporaryDirectory() as carpeta:
        r = E.C.resumir(df, METODOS=["prueba"], RES=carpeta).iloc[0]
    assert r["D (%)"] == 50 and r["P (%)"] == 50 and r["N (%)"] == 0
    assert r["D en señuelos (%)"] == 25
    assert r["J_D = TPR_D - FPR_D"] == .25
    assert r["D+P (%)"] == 100 and r["J_DP"] == .75
    assert r["D en cat. S&R Y (%)"] == 100
    assert r["D en cat. S&R N (%)"] == 0
    assert pd.isna(r["D en cat. S&R P (%)"])
    print("OK: porcentajes, señuelos y J del resumen")
