"""Compara curtosis y curtosis/entropía sin recalcular el árbol por peso."""
import argparse
from pathlib import Path

import pandas as pd
import Fuentes as F


def correr(registros, datos, salida):
    filas = []
    for numero in registros:
        reg = F.cargar_registro(numero, ruta_datos=str(datos))
        x, fs, fd = reg["x"], reg["fs"], reg["fd"]
        if fd is None:
            raise ValueError(f"Registro {numero}: se requiere frecuencia de falla para evaluar")
        tabla = F.kurtogram(x, fs, F.nivel_maximo(fs, fd), con_entropia=True)
        for peso in (1.0, 0.7, 0.5, 0.3):
            scores = F.puntuacion_bandas(tabla, peso)
            r = tabla.loc[scores.idxmax()]
            l, m = int(r.l), int(r.m)
            xf = F.filtrar(x, fs, l, m)
            diag = F.evaluar_diagnostico(*F.envolvente_espectral(xf, fs), fd, reg["fr"])
            filas.append(dict(registro=numero, peso_K=peso, peso_H=round(1-peso, 1),
                              l=l, m=m, fL=r.fL, fU=r.fU, K=r.K, H=r.H,
                              score=float(scores.loc[r.name]), categoria=diag["categoria"],
                              ratio_db_fd=diag["ratio_db_fd"]))
        print(numero, [(f["peso_K"], f["fL"], f["fU"], f["categoria"]) for f in filas[-4:]], flush=True)
    salida.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(filas)
    df.to_csv(salida / "comparacion_entropia.csv", index=False)
    resumen = pd.crosstab(df.peso_K, df.categoria).reindex(columns=["D", "P", "N"], fill_value=0)
    resumen.to_csv(salida / "resumen_entropia.csv")
    print(resumen.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registros", nargs="*", type=int)
    parser.add_argument("--datos", type=Path, default=Path(F.RAIZ_PROYECTO) / "data2" / "CWRU")
    parser.add_argument("--salida", type=Path, default=Path(__file__).parent / "resultados" / "entropia")
    args = parser.parse_args()
    registros = args.registros
    if not registros:
        idx = F.leer_indice(str(args.datos))
        registros = idx.loc[idx.carpeta.isin(["12k_DE", "12k_FE"]), "registro"].astype(int).tolist()
    correr(registros, args.datos, args.salida)
