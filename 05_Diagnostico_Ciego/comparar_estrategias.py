"""Una extracción para cuatro reglas; exclusivamente la mitad de desarrollo fijada."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F, inferir, clasificar_evidencia, ESTRATEGIAS

AQUI = Path(__file__).parent


def ejecutar(datos, salida):
    idx = F.leer_indice(str(datos))
    particion = pd.read_csv(AQUI/"particion_desarrollo.csv")
    dev = set(particion.loc[particion.particion == "desarrollo", "archivo"])
    idx = idx[idx.archivo.isin(dev)]
    if set(idx.archivo) != dev:
        raise ValueError("No están todos los archivos de desarrollo")
    predicciones = []
    for _, r in idx.iterrows():
        d = loadmat(datos/r.archivo)
        clave = F.clave_canal(d, r.sensor_falla, int(r.registro))
        rpm_keys = [k for k in d if k.endswith("RPM")]
        rpm = F.rpm_registro(d, int(r.registro), nominal=float(r.rpm_nominal))
        catalogo = {k: float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
        pred = inferir(np.ravel(d[clave]), float(r.fs), frecuencias_fabricante=catalogo)
        variantes = {e: clasificar_evidencia(pred["evidencia_catalogo"], e) for e in ESTRATEGIAS}
        predicciones.append(dict(archivo=r.archivo, catalogo=catalogo,
                                bandas=pred["bandas"], evidencia=pred["evidencia_catalogo"], variantes=variantes))
        (salida/"predicciones.json").write_text(json.dumps(predicciones, indent=2), encoding="utf-8")
        print(f"{len(predicciones)}/{len(idx)} {r.archivo}", flush=True)
    # Evaluación posterior: no se consulta la zona para generar candidatos.
    etiquetas = idx.set_index("archivo").zona.to_dict()
    filas = []
    for pred in predicciones:
        for estrategia, candidatos in pred["variantes"].items():
            tipos = [c["tipo"] for c in candidatos]
            real = etiquetas[pred["archivo"]]
            filas.append(dict(archivo=pred["archivo"], estrategia=estrategia, real=real,
                tipo_top1=tipos[0] if tipos else "sin_evidencia", candidatos=";".join(tipos),
                acierto_top1=(bool(tipos) and tipos[0] == real), acierto_top3=real in tipos,
                alerta=bool(tipos)))
    df = pd.DataFrame(filas)
    df.to_csv(salida/"evaluacion.csv", index=False)
    resumen = []
    for estrategia in ESTRATEGIAS:
        sub = df[df.estrategia == estrategia]
        fallas, sanos = sub[sub.real != "N"], sub[sub.real == "N"]
        resumen.append(dict(estrategia=estrategia, fallas=len(fallas),
            top1=int(fallas.acierto_top1.sum()), top3=int(fallas.acierto_top3.sum()),
            top1_pct=round(100*fallas.acierto_top1.mean(), 2), top3_pct=round(100*fallas.acierto_top3.mean(), 2),
            alertas_sanos=int(sanos.alerta.sum()), sanos=len(sanos)))
    res = pd.DataFrame(resumen)
    res.to_csv(salida/"resumen.csv", index=False)
    base = pd.read_csv(AQUI/"resultados_catalogo_desarrollo/evaluacion.csv").set_index("archivo")
    actual = df[df.estrategia == "conservador"].set_index("archivo")
    if not base.tipo_top1.equals(actual.loc[base.index, "tipo_top1"]):
        raise AssertionError("La referencia conservadora no reproduce el resultado previo")
    lineas = ["# Estrategias sin frecuencia esperada: desarrollo", "",
        "Mismas 54 señales y mismas bandas L=4. Reserva excluida.",
        "Conservador: dos armónicas, prioridad a su cantidad, corrección por 5×bandas×3.",
        "Sensible: una armónica, orden por suma de márgenes sobre umbral, misma corrección.",
        "Dominante: una armónica, prioriza categoría D y luego margen, misma corrección.",
        "Sin control: regla dominante con p=0,01 por prueba, sin corrección de hipótesis.",
        "La presencia en top3 no es diagnóstico único correcto. Dos sanos no bastan para validar especificidad.",
        "Se eligen variantes con datos de desarrollo; no son resultados de prueba independiente.", "",
        "| " + " | ".join(res.columns) + " |", "| " + " | ".join(["---"]*len(res.columns)) + " |"]
    lineas += ["| " + " | ".join(map(str, r)) + " |" for r in res.itertuples(index=False, name=None)]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n", encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path(F.RAIZ_PROYECTO)/"data2/CWRU")
    parser.add_argument("--salida", type=Path, default=AQUI/"resultados_estrategias_desarrollo")
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    ejecutar(args.datos, args.salida)
