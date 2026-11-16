"""Kurtogram vs. kurtogram + entropía, con el criterio y resumen de 03."""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "03_Comparacion_Kurtogram_vs_Sinergia"))
import compara as C

F = C.F
METODOS = ["Kurtogram (Tarea 1)", "Kurtogram + entropía 70/30",
           "Kurtogram + entropía 50/50", "Kurtogram + entropía 30/70"]
FILTROS = ["Kurtogram (Tarea 1)", "Entropygram: conservar 25%",
           "Entropygram: conservar 50%", "Entropygram: conservar 75%"]
TEMPORAL = [FILTROS[0], FILTROS[3], "Filtro 75% + mediana temporal"]


def candidatos_entropia(tabla, fraccion):
    """Conserva ceil(fracción*N) nodos; empates H respetan el orden del árbol."""
    if not math.isfinite(fraccion) or not 0 < fraccion <= 1:
        raise ValueError("fraccion debe estar en (0, 1]")
    if tabla.empty or not tabla.H.between(0, 1).all():
        raise ValueError("Se requiere una tabla no vacía con H en [0, 1]")
    return tabla.sort_values("H", kind="stable").head(math.ceil(fraccion * len(tabla))).sort_index()


def kurtogram_temporal(x, fs, L, ventanas=5):
    """Medianas por nodo en ventanas iguales no solapadas; descarta <ventanas muestras finales."""
    x = np.asarray(x, dtype=float)
    if not isinstance(ventanas, int) or ventanas < 3:
        raise ValueError("Se requieren al menos tres ventanas")
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Se requiere una señal unidimensional finita")
    n = x.size // ventanas
    if n < max(32, 4 * 2 ** L):
        raise ValueError("Ventanas demasiado cortas para el árbol")
    tablas = [F.kurtogram(x[i*n:(i+1)*n], fs, L, con_entropia=True)
              for i in range(ventanas)]
    tabla = tablas[0].copy()
    for columna in ("K", "H"):
        valores = np.array([t[columna].to_numpy() for t in tablas])
        tabla[columna] = np.median(valores, axis=0)
        tabla[columna + "_MAD"] = np.median(np.abs(valores - tabla[columna].to_numpy()), axis=0)
    return tabla


def correr(datos, salida, registros, filtros=False, temporal=False, ventanas=5):
    metodos = TEMPORAL if temporal else FILTROS if filtros else METODOS
    idx = F.leer_indice(str(datos))
    idx = idx[idx.carpeta.isin(["12k_DE", "12k_FE"])]
    if registros:
        faltantes = set(registros) - set(idx.registro)
        if faltantes:
            raise ValueError(f"Registros fuera del conjunto de 12 kHz: {faltantes}")
        idx = idx[idx.registro.isin(registros)]
    filas = []
    for _, r in idx.iterrows():
        reg = F.cargar_registro(int(r.registro), ruta_datos=str(datos))
        x, fs, fd, fr = (reg[k] for k in ("x", "fs", "fd", "fr"))
        L = F.nivel_maximo(fs, fd)
        tabla = F.kurtogram(x, fs, L, con_entropia=True)
        tabla_temporal = kurtogram_temporal(x, fs, L, ventanas) if temporal else None
        freqs = {"fd": fd, **{f"senuelo{j}": c * fr
                 for j, c in enumerate(C.senuelos(reg["sensor_falla"]))}}
        fila = dict(registro=int(r.registro), archivo=r.nombre,
                    sensor=reg["sensor_falla"], zona=reg["zona"], L=L,
                    bench=C.BENCH.get(int(r.registro), "?"), **freqs)
        espectros = {}
        parametros = (1., .75, .75) if temporal else (1., .25, .5, .75) if filtros else (1., .7, .5, .3)
        for metodo, parametro in zip(metodos, parametros):
            base = tabla_temporal if temporal and metodo == TEMPORAL[-1] else tabla
            candidatos = candidatos_entropia(base, parametro) if filtros or temporal else base
            scores = F.puntuacion_bandas(candidatos, 1. if filtros or temporal else parametro)
            nodo = candidatos.loc[scores.idxmax()]
            l, m = int(nodo.l), int(nodo.m)
            if (l, m) not in espectros:
                espectros[l, m] = F.envolvente_espectral(F.filtrar(x, fs, l, m), fs)
            for tag, f in freqs.items():
                fila[f"{metodo}|{tag}"] = F.evaluar_diagnostico(
                    *espectros[l, m], f, fr)["categoria"]
            fila.update({f"{metodo}|banda": f"{nodo.fL:.0f}-{nodo.fU:.0f}",
                         f"{metodo}|l": l, f"{metodo}|m": m,
                         f"{metodo}|K": nodo.K, f"{metodo}|H": nodo.H,
                         f"{metodo}|candidatos": len(candidatos),
                         f"{metodo}|score": float(scores.loc[nodo.name])})
            if temporal and metodo == TEMPORAL[-1]:
                fila.update({f"{metodo}|K_MAD": nodo.K_MAD, f"{metodo}|H_MAD": nodo.H_MAD,
                             "ventanas": ventanas, "muestras_ventana": len(x) // ventanas,
                             "muestras_descartadas": len(x) % ventanas})
        filas.append(fila)
        print(f"{len(filas)}/{len(idx)} registro {r.registro}: " +
              str([fila[f"{met}|fd"] for met in metodos]), flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(salida / "comparacion_12k.csv", index=False)
    return df


def matriz_transiciones(antes, ahora):
    """Filas=antes, columnas=ahora; N se muestra como ND en la figura."""
    categorias = ["N", "P", "D"]
    if not antes.isin(categorias).all() or not ahora.isin(categorias).all():
        raise ValueError("Las clasificaciones deben ser N, P o D")
    if not antes.index.equals(ahora.index):
        raise ValueError("Se requieren los mismos registros en el mismo orden")
    return pd.crosstab(antes, ahora).reindex(index=categorias, columns=categorias, fill_value=0)


def graficar_transiciones(df, salida, metodos):
    import matplotlib.pyplot as plt
    import numpy as np

    antes = df[f"{metodos[0]}|fd"]
    matrices = [matriz_transiciones(antes, df[f"{met}|fd"]) for met in metodos[1:]]
    fig, axs = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 4.4),
                            constrained_layout=True, squeeze=False)
    vmax = max(int(m.to_numpy().max()) for m in matrices)
    filas = []
    for ax, metodo, matriz in zip(axs.flat, metodos[1:], matrices):
        valores = matriz.to_numpy()
        ax.imshow(valores, cmap="Blues", vmin=0, vmax=max(vmax, 1))
        for i, previa in enumerate(matriz.index):
            for j, actual in enumerate(matriz.columns):
                n = int(valores[i, j])
                ax.text(j, i, str(n), ha="center", va="center", fontsize=16,
                        color="white" if n > vmax / 2 else "#111111")
                filas.append(dict(metodo=metodo, antes=previa, ahora=actual, cantidad=n))
        mejoran = int(np.triu(valores, 1).sum())
        empeoran = int(np.tril(valores, -1).sum())
        iguales = int(np.trace(valores))
        ax.set_xticks(range(3), ["ND", "P", "D"])
        ax.set_yticks(range(3), ["ND", "P", "D"])
        ax.set_xlabel("Ahora: clasificación del método")
        ax.set_ylabel("Antes: kurtogram original")
        ax.set_title(metodo.replace("Entropygram: ", ""), fontsize=11)
        ax.text(.5, -.24, f"Mejoran: {mejoran} · Empeoran: {empeoran}\nSin cambio: {iguales}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
    fig.suptitle(f"Cambios de clasificación en fd · {len(df)} registros\n"
                 "ND: no diagnosticable · P: parcialmente diagnosticable · D: diagnosticable",
                 fontsize=12)
    fig.savefig(salida / "transiciones_clasificacion.png", dpi=180)
    plt.close(fig)
    transiciones = pd.DataFrame(filas)
    transiciones.to_csv(salida / "transiciones_clasificacion.csv", index=False)
    return transiciones


def resumir(df, salida, filtros=False, temporal=False):
    metodos = TEMPORAL if temporal else FILTROS if filtros else METODOS
    res = C.resumir(df, METODOS=metodos, RES=str(salida))
    C.graficar(res, df, METODOS=metodos, RES=str(salida))
    graficar_transiciones(df, salida, metodos)
    # Markdown sin dependencia adicional de tabulate.
    lineas = ["# Kurtogram vs. kurtogram + entropía", "",
              f"{len(df)} registros; dos frecuencias señuelo por registro.", "",
              (f"Temporal: mediana de K y H en {int(df.ventanas.iloc[0])} ventanas; conservar 75% por H y maximizar K."
               if temporal else "Filtros: conservar 25%, 50% y 75% de menor H y maximizar K entre ellos."
               if filtros else "Pesos en orden curtosis/entropía."),
              "D: diagnosticable; P: parcial; N: no diagnosticable.",
              "J_D = proporción D en fd − proporción D en señuelos. Mayor es mejor.", "",
              "| " + " | ".join(res.columns) + " |",
              "| " + " | ".join(["---"] * len(res.columns)) + " |"]
    lineas += ["| " + " | ".join(str(v) for v in fila) + " |"
               for fila in res.itertuples(index=False, name=None)]
    (salida / "resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registros", nargs="*", type=int)
    parser.add_argument("--datos", type=Path, default=RAIZ / "data2" / "CWRU")
    parser.add_argument("--salida", type=Path, default=Path(__file__).parent / "resultados")
    parser.add_argument("--solo-resumen", action="store_true")
    parser.add_argument("--filtros", action="store_true", help="Filtrar candidatos por entropía antes de maximizar K")
    parser.add_argument("--temporal", action="store_true", help="Comparar el filtro 75% con medianas temporales")
    parser.add_argument("--ventanas", type=int, default=5)
    args = parser.parse_args()
    if args.temporal:
        args.salida = args.salida / "temporal"
    elif args.filtros:
        args.salida = args.salida / "filtros"
    args.salida.mkdir(parents=True, exist_ok=True)
    df = (pd.read_csv(args.salida / "comparacion_12k.csv") if args.solo_resumen
          else correr(args.datos, args.salida, args.registros, args.filtros, args.temporal, args.ventanas))
    resumir(df, args.salida, args.filtros, args.temporal)
