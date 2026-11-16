"""Desarrollo en media muestra con catálogo completo; etiquetas solo a posteriori."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import loadmat
from modelo_ciego import inferir, F, ESTRATEGIAS


def dividir_desarrollo(idx, semilla=2026):
    """Mitad por carpeta de adquisición, sin usar el defecto real."""
    if idx.archivo.duplicated().any():
        raise ValueError("Archivos duplicados")
    rng = np.random.default_rng(semilla)
    filas = []
    for carpeta, grupo in idx.groupby("carpeta", sort=True):
        grupo = grupo.sort_values("archivo")
        ids = grupo.archivo.to_numpy()
        elegidos = set(rng.permutation(ids)[:len(ids)//2])
        filas.extend(dict(archivo=r.archivo, registro=int(r.registro), carpeta=carpeta,
                          particion="desarrollo" if r.archivo in elegidos else "reserva")
                     for r in grupo.itertuples())
    return pd.DataFrame(filas).sort_values("archivo").reset_index(drop=True)


def correr(datos, salida, registros, estrategia="dominante"):
    idx = F.leer_indice(str(datos))
    idx = idx[idx.carpeta.isin(["12k_DE", "12k_FE", "Normal"])]
    particion = dividir_desarrollo(idx)
    ruta = Path(__file__).parent/"particion_desarrollo.csv"
    if ruta.exists():
        if not pd.read_csv(ruta).equals(particion):
            raise ValueError("El índice no coincide con la partición fijada")
    else:
        particion.to_csv(ruta, index=False)
    desarrollo = set(particion.loc[particion.particion == "desarrollo", "archivo"])
    idx = idx[idx.archivo.isin(desarrollo)]
    if registros and set(registros)-set(idx.registro):
        raise ValueError("Solo se permiten registros de desarrollo, no de reserva")
    if registros:
        idx = idx[idx.registro.isin(registros)]
    predicciones = []
    for _, r in idx.iterrows():
        d = loadmat(datos/r.archivo)
        claves = [k for k in d if k.endswith(f"_{r.sensor_falla}_time")]
        if not claves:
            raise ValueError("Canal de adquisición ausente")
        claves_rpm = [k for k in d if k.endswith("RPM")]
        rpm = float(np.ravel(d[claves_rpm[0]])[0]) if claves_rpm else float(r.rpm_nominal)
        catalogo = {k: float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
        pred = inferir(np.ravel(d[claves[0]]), float(r.fs), frecuencias_fabricante=catalogo,
                       estrategia=estrategia)
        predicciones.append(dict(archivo=r.archivo, registro=int(r.registro), catalogo=catalogo, **pred))
        (salida/"predicciones.json").write_text(json.dumps(predicciones, indent=2), encoding="utf-8")
        print(f"{len(predicciones)}/{len(idx)} {int(r.registro)}: " + str([c["tipo"] for c in pred["candidatos"]]), flush=True)
    # Se consulta la etiqueta después de fijar todas las predicciones; no se calcula fd real.
    etiquetas = idx.set_index("archivo").zona.to_dict()
    filas = []
    for pred in predicciones:
        real = etiquetas[pred["archivo"]]
        tipos = [c["tipo"] for c in pred["candidatos"]]
        filas.append(dict(archivo=pred["archivo"], registro=pred["registro"], real=real, estado=pred["estado"],
                          tipo_top1=tipos[0] if tipos else "sin_evidencia",
                          acierto_top1=(bool(tipos) and tipos[0] == real) if real != "N" else None,
                          acierto_top3=real in tipos if real != "N" else None,
                          alerta_en_sano=bool(tipos) if real == "N" else None))
    df = pd.DataFrame(filas)
    df.to_csv(salida/"evaluacion.csv", index=False)
    pd.crosstab(df.real, df.tipo_top1).to_csv(salida/"matriz_tipos.csv")
    fallas, sanos = df[df.real != "N"], df[df.real == "N"]
    lineas = ["# Catálogo completo: mitad de desarrollo", "",
              "L=4 fijo; filtro 25% + kurtogram; catálogo BPFI/BPFO/BSF/FTF y 2BSF.",
              f"Estrategia: {estrategia}. Conservador exige dos armónicas; las demás admiten una.",
              "Corrección nominal por cinco hipótesis, bandas y tres armónicas, salvo estrategia sin_control.",
              "Inferencia sin frecuencia ni tipo de defecto real.", "",
              f"Desarrollo ejecutado: {len(df)}; reserva no ejecutada: {int((particion.particion == 'reserva').sum())}.",
              f"Con falla: {len(fallas)}. Tipo correcto top1: {int(fallas.acierto_top1.sum())}/{len(fallas)}.",
              f"Tipo correcto entre tres hipótesis: {int(fallas.acierto_top3.sum())}/{len(fallas)}.",
              f"Sanos: {len(sanos)}. Con alerta: {int(sanos.alerta_en_sano.sum())}/{len(sanos)}.",
              "Sin evidencia no confirma salud. Top3 no equivale a diagnóstico único correcto.",
              "La reserva se observó en experimentos anteriores; no es validación independiente nueva.",
              "Factores disponibles: referencia del proyecto; no ficha de fabricante verificada."]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n", encoding="utf-8")
    (salida/"configuracion.json").write_text(json.dumps(dict(semilla=2026, L=4, p=.01, estrategia=estrategia,
        fraccion_entropia=.25, datos=str(datos.resolve()), particion=str(ruta.resolve()),
        archivos=df.archivo.tolist()), indent=2), encoding="utf-8")
    print("\n".join(lineas))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("registros", nargs="*", type=int)
    p.add_argument("--datos", type=Path, default=Path(F.RAIZ_PROYECTO)/"data2/CWRU")
    p.add_argument("--salida", type=Path)
    p.add_argument("--estrategia", choices=ESTRATEGIAS, default="dominante")
    args = p.parse_args()
    if args.salida is None:
        args.salida = Path(__file__).parent/f"resultados_catalogo_{args.estrategia}_desarrollo"
    args.salida.mkdir(parents=True, exist_ok=True)
    correr(args.datos, args.salida, args.registros, args.estrategia)


if __name__ == "__main__":
    main()
