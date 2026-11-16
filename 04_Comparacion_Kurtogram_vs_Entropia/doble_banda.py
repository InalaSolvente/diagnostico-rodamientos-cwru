"""Unión kurtogram/filtro 25%, con y sin corrección por bandas distintas."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import compara_entropia as E

F = E.F
METODOS = [E.FILTROS[0], E.FILTROS[1], "Doble banda sin control", "Doble banda con control"]
ORDEN = {"N": 0, "P": 1, "D": 2}


def evaluar_bandas(espectros, frecuencia, fr, control=True, p=.01):
    """La unión conserva la mejor categoría; Bonferroni solo entre bandas únicas."""
    if not 1 <= len(espectros) <= 2 or not np.isfinite(p) or not 0 < p < 1:
        raise ValueError("Se requieren una o dos bandas y p en (0, 1)")
    p_banda = p / len(espectros) if control else p
    categorias = [F.evaluar_diagnostico(*s, frecuencia, fr, p=p_banda)["categoria"]
                  for s in espectros.values()]
    return max(categorias, key=ORDEN.get), categorias, p_banda


def correr(datos, bandas, salida, registros):
    origen = pd.read_csv(bandas)
    if origen.registro.duplicated().any() or origen.empty:
        raise ValueError("El CSV de bandas debe contener registros únicos")
    if registros:
        if set(registros) - set(origen.registro):
            raise ValueError("Hay registros ausentes del CSV de bandas")
        origen = origen[origen.registro.isin(registros)]
    filas = []
    for _, r in origen.iterrows():
        reg = F.cargar_registro(int(r.registro), ruta_datos=str(datos))
        x, fs, fd, fr = (reg[k] for k in ("x", "fs", "fd", "fr"))
        if fd is None or not np.isclose(fd, r.fd) or F.nivel_maximo(fs, fd) != r.L:
            raise ValueError(f"Registro {r.registro}: metadatos incompatibles con las bandas")
        nodos = []
        for metodo in METODOS[:2]:
            l, m = r[f"{metodo}|l"], r[f"{metodo}|m"]
            if l != int(l) or m != int(m) or not 0 <= l <= r.L or not 0 <= m < 2**int(l):
                raise ValueError("Nodo inválido en el CSV de bandas")
            nodos.append((int(l), int(m)))
        espectros = {n: F.envolvente_espectral(F.filtrar(x, fs, *n), fs) for n in dict.fromkeys(nodos)}
        fila = {k: r[k] for k in ("registro", "archivo", "sensor", "zona", "L", "bench")}
        fila["bandas_distintas"] = len(espectros)
        fila["p_por_banda_control"] = .01 / len(espectros)
        for j, metodo in enumerate(METODOS[:2]):
            fila[f"{metodo}|banda"] = r[f"{metodo}|banda"]
            fila[f"{metodo}|l"], fila[f"{metodo}|m"] = nodos[j]
        for tag, frecuencia in {"fd": fd, **{f"senuelo{j}": c * fr
                               for j, c in enumerate(E.C.senuelos(reg["sensor_falla"]))}}.items():
            fila[tag] = frecuencia
            for metodo, nodo in zip(METODOS[:2], nodos):
                cat = evaluar_bandas({nodo: espectros[nodo]}, frecuencia, fr)[0]
                if cat != r[f"{metodo}|{tag}"]:
                    raise ValueError(f"Registro {r.registro}: referencia no reproducible ({metodo}, {tag})")
                fila[f"{metodo}|{tag}"] = cat
            for metodo, control in zip(METODOS[2:], (False, True)):
                cat, por_banda, _ = evaluar_bandas(espectros, frecuencia, fr, control)
                fila[f"{metodo}|{tag}"] = cat
                fila[f"{metodo}|{tag}|por_banda"] = ";".join(por_banda)
        filas.append(fila)
        print(f"{len(filas)}/{len(origen)} registro {int(r.registro)}: " +
              str([fila[f"{met}|fd"] for met in METODOS]), flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(salida / "comparacion_12k.csv", index=False)
    (salida / "configuracion.json").write_text(json.dumps(dict(
        bandas=str(bandas.resolve()), sha256_bandas=hashlib.sha256(bandas.read_bytes()).hexdigest(),
        p=.01, armonicas=3, tolerancia=.02, correccion="p / número de bandas distintas",
        datos=str(datos.resolve()), registros=df.registro.astype(int).tolist()), indent=2), encoding="utf-8")
    return df


def resumir(df, salida):
    res = E.C.resumir(df, METODOS=METODOS, RES=str(salida))
    E.C.graficar(res, df, METODOS=METODOS, RES=str(salida))
    E.graficar_transiciones(df, salida, METODOS)
    conteos = []
    for metodo in METODOS:
        c = df[f"{metodo}|fd"]
        conteos.append(dict(metodo=metodo, total=len(df), D=int((c=="D").sum()),
                            P=int((c=="P").sum()), N=int((c=="N").sum()),
                            detectados=int(c.isin(["D", "P"]).sum())))
    pd.DataFrame(conteos).to_csv(salida / "conteos.csv", index=False)
    lineas = ["# Doble banda: kurtogram y filtro de entropía 25%", "",
              f"{len(df)} registros. Dos bandas distintas: {int((df.bandas_distintas == 2).sum())}.", "",
              "D+P cuenta detecciones compatibles con la falla esperada, no precisión validada.",
              "La corrección divide p=0,01 entre el número de bandas distintas; no corrige las tres armónicas ni la selección adaptativa de bandas.",
              "Los señuelos son controles fuera de frecuencias objetivo, no falsos positivos confirmados.", "",
              "| " + " | ".join(res.columns) + " |", "| " + " | ".join(["---"]*len(res.columns)) + " |"]
    lineas += ["| " + " | ".join(map(str, fila)) + " |" for fila in res.itertuples(index=False, name=None)]
    (salida / "resumen.md").write_text("\n".join(lineas)+"\n", encoding="utf-8")
    print(pd.DataFrame(conteos).to_string(index=False))
    print(res.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registros", nargs="*", type=int)
    parser.add_argument("--datos", type=Path, default=E.RAIZ / "data2" / "CWRU")
    parser.add_argument("--bandas", type=Path, default=Path(__file__).parent / "resultados/filtros/comparacion_12k.csv")
    parser.add_argument("--salida", type=Path, default=Path(__file__).parent / "resultados/doble_banda")
    parser.add_argument("--solo-resumen", action="store_true")
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    df = (pd.read_csv(args.salida / "comparacion_12k.csv") if args.solo_resumen
          else correr(args.datos, args.bandas, args.salida, args.registros))
    resumir(df, args.salida)
