"""Guarda primero predicciones ciegas; consulta etiquetas solo para evaluación posterior."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import loadmat
from modelo_ciego import inferir, F


def correr(datos, salida, registros):
    idx = F.leer_indice(str(datos))
    idx = idx[idx.carpeta.isin(["12k_DE", "12k_FE", "Normal"])]
    if registros:
        if set(registros)-set(idx.registro):
            raise ValueError("Registros fuera del conjunto de evaluación")
        idx = idx[idx.registro.isin(registros)]
    predicciones = []
    for _, r in idx.iterrows():
        d = loadmat(datos/r.archivo)
        # Canal de adquisición del conjunto; nunca se pasa el tipo de defecto al modelo.
        sensor = r.sensor_falla
        claves = [k for k in d if k.endswith(f"_{sensor}_time")]
        if not claves:
            raise ValueError(f"Registro {r.registro}: canal {sensor} ausente")
        pred = inferir(np.ravel(d[claves[0]]), float(r.fs))
        predicciones.append(dict(registro=int(r.registro), **pred))
        print(f"{len(predicciones)}/{len(idx)} {int(r.registro)}: " +
              str([round(c['frecuencia_hz'], 2) for c in pred['candidatos']]), flush=True)
    # Las predicciones quedan fijadas antes de calcular frecuencias teóricas.
    (salida/"predicciones.json").write_text(json.dumps(predicciones, indent=2), encoding="utf-8")
    filas = []
    for pred in predicciones:
        reg = F.cargar_registro(pred["registro"], ruta_datos=str(datos))
        fd = reg["fd"]
        candidatos = pred["candidatos"]
        def coincide(c):
            return any(abs(c["frecuencia_hz"]-h*fd) <= .02*h*fd for h in (1, 2, 3))
        filas.append(dict(registro=pred["registro"], zona=reg["zona"], fd_evaluacion=fd,
                          estado=pred["estado"], f_top1=candidatos[0]["frecuencia_hz"] if candidatos else np.nan,
                          acierto_top1=bool(candidatos and coincide(candidatos[0])) if fd else None,
                          acierto_top3=any(coincide(c) for c in candidatos) if fd else None,
                          deteccion_en_sano=bool(candidatos) if fd is None else None))
    df = pd.DataFrame(filas)
    df.to_csv(salida/"evaluacion.csv", index=False)
    fallas = df[df.fd_evaluacion.notna()]
    sanos = df[df.fd_evaluacion.isna()]
    lineas = ["# Evaluación ciega experimental", "",
              "Inferencia: solo x y fs; L=4, rango 10–1000 Hz, hasta tres candidatos.",
              "Coincidencia posterior: ±2% de fd, 2fd o 3fd; no clasifica el tipo de defecto.",
              "No se usan los porcentajes D/P del procedimiento con frecuencia objetivo.", "",
              f"Registros con falla: {len(fallas)}.",
              f"Coincidencia top1: {int(fallas.acierto_top1.sum())} / {len(fallas)}.",
              f"Coincidencia top3: {int(fallas.acierto_top3.sum())} / {len(fallas)}.",
              f"Registros sanos: {len(sanos)}; con periodicidad detectada: {int(sanos.deteccion_en_sano.sum())}.",
              "Periodicidad detectada no implica por sí sola una falla."]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    from evaluar_catalogo import main
    main()
