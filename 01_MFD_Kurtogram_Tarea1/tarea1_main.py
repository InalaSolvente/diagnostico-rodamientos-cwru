"""
tarea1_main.py — Ejecuta la Tarea 1 completa.

    python tarea1_main.py                 # todos los registros 12 kHz (DE y FE) + figuras de los casos elegidos
    python tarea1_main.py 282 105         # sólo figuras y fichas de esos registros

Salidas (dentro de esta carpeta):
    resultados/tabla_registros_12k.csv  diagnóstico de cada registro (señal cruda y filtrada)
    resultados/fichas.md                fichas de entregable (Parcialmente / No diagnosticable)
    figuras/<registro>/                 las 6 figuras de la tarea + kurtogram
"""
import os
import sys
import json
import pandas as pd
import Fuentes as F

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "resultados")
FIG = os.path.join(AQUI, "figuras")


def correr_todos() -> pd.DataFrame:
    idx = F.leer_indice()
    idx = idx[idx.carpeta.isin(["12k_DE", "12k_FE"])]
    filas = []
    for _, r in idx.iterrows():
        reg = F.cargar_registro(int(r.registro))
        res = F.diagnosticar(reg)
        fi = F.ficha(reg, res)
        fi.update({"carpeta": r.carpeta, "archivo": r.nombre, "zona": reg["zona"],
                   "cat_cruda": res["diag_cruda"]["categoria"], "cat_filtrada": res["diag_filtrada"]["categoria"],
                   "L": res["L"], "l": res["banda"][0], "m": res["banda"][1], "nota": r.nota})
        filas.append(fi)
        print(f"{r.registro:>4} {r.nombre:<12} banda {fi['Banda MFD (Hz)']:>11}  cruda={fi['cat_cruda']}  filtrada={fi['cat_filtrada']}")
    tabla = pd.DataFrame(filas)
    os.makedirs(RES, exist_ok=True)
    tabla.to_csv(os.path.join(RES, "tabla_registros_12k.csv"), index=False)
    return tabla


def figuras_y_fichas(registros):
    os.makedirs(RES, exist_ok=True)
    fichas = []
    for n in registros:
        reg = F.cargar_registro(int(n))
        res = F.diagnosticar(reg)
        F.figuras_registro(reg, res, os.path.join(FIG, str(n)))
        fi = F.ficha(reg, res)
        fichas.append(fi)
        with open(os.path.join(FIG, str(n), "diagnostico.json"), "w", encoding="utf-8") as fh:
            json.dump({"ficha": fi, "diag_cruda": res["diag_cruda"], "diag_filtrada": res["diag_filtrada"]},
                      fh, ensure_ascii=False, indent=2, default=float)
    return fichas


def escribir_fichas(fichas, titulo_por_registro):
    lineas = ["# Fichas de entregable — Tarea 1 (MFD + Envolvente Espectral)\n"]
    for fi in fichas:
        lineas.append(f"## {titulo_por_registro.get(fi['Nro. File'], 'Registro')} — archivo {fi['Nro. File']}\n")
        for k, v in fi.items():
            lineas.append(f"- **{k}:** {v}")
        lineas.append(f"\nFiguras: `figuras/{fi['Nro. File']}/`\n")
    with open(os.path.join(RES, "fichas.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        figuras_y_fichas([int(a) for a in sys.argv[1:]])
    else:
        tabla = correr_todos()
        print(tabla.groupby("cat_filtrada").size())
