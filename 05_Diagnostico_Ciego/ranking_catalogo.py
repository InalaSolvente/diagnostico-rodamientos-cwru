"""Ordenación experimental y validación por condición; solo evidencia de desarrollo."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

TIPOS = ("IR", "OR", "B", "FTF")
FEATURES = ("fuerza_relativa", "log_margen", "armonicas", "dominante", "respaldo_bandas",
            "margen_medio_bandas", "exclusividad", "contraste")


def grupo_condicion(archivo):
    # Todos los niveles de carga y ambos sensores de la misma condición quedan juntos.
    return re.sub(r"_\d+\.mat$", "", Path(archivo).name)


def caracteristicas(registro):
    """Solo evidencia espectral: no consulta etiquetas, nombre del archivo ni tipo real."""
    evidencia = [r for r in registro["evidencia"] if not r["control"] and r["armonicas"]]
    nodos = {tuple(n) for n in registro["bandas"]}
    maximos = {t: max([r["score"] for r in evidencia if r["tipo"] == t] or [0.]) for t in TIPOS}
    escala = max(np.log1p(list(maximos.values())).max(), 1e-12)
    filas, activos = [], []
    for tipo in TIPOS:
        filas_tipo = [r for r in evidencia if r["tipo"] == tipo]
        activos.append(bool(filas_tipo))
        maximo = maximos[tipo]
        media_bandas = sum(np.log1p(max([r["score"] for r in filas_tipo if tuple(r["nodo"]) == n] or [0.]))
                           for n in nodos)/max(len(nodos), 1)
        picos = {(tuple(r["nodo"]), float(r["frecuencia_hz"]*h)) for r in filas_tipo for h in r["armonicas"]}
        otros = [(tuple(r["nodo"]), r["frecuencia_hz"]*h) for r in evidencia if r["tipo"] != tipo for h in r["armonicas"]]
        exclusivos = sum(not any(n == no and abs(f-fo) <= .02*max(f, fo) for no, fo in otros) for n, f in picos)
        respaldo = len({tuple(r["nodo"]) for r in filas_tipo})/max(len(nodos), 1)
        competencia = max(v for t,v in maximos.items() if t != tipo)
        filas.append([np.log1p(maximo)/escala, np.log1p(maximo),
            max([len(r["armonicas"]) for r in filas_tipo] or [0])/3,
            float(any(r["categoria"] == "D" for r in filas_tipo)), respaldo, media_bandas,
            exclusivos/max(len(picos), 1), np.log1p(maximo)-np.log1p(competencia)])
    x = np.asarray(filas)
    if not np.isfinite(x).all():
        raise ValueError("Evidencia no finita")
    return x, np.asarray(activos)


def ajustar(x, y, alpha=1.):
    """Ridge compartido entre tipos, sin parámetros específicos para IR/OR/B/FTF."""
    if len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Datos de ajuste vacíos o no finitos")
    media, desv = x.mean(axis=0), x.std(axis=0)
    desv[desv < 1e-12] = 1.
    z = np.column_stack([np.ones(len(x)), (x-media)/desv])
    penalizacion = np.eye(z.shape[1])*alpha
    penalizacion[0, 0] = 0
    pesos = np.linalg.solve(z.T@z+penalizacion, z.T@y)
    return dict(media=media.tolist(), desv=desv.tolist(), pesos=pesos.tolist(), alpha=alpha)


def puntuar(x, modelo):
    z = np.column_stack([np.ones(len(x)), (x-np.asarray(modelo["media"]))/np.asarray(modelo["desv"])])
    return z@np.asarray(modelo["pesos"])


def validar_grupos(x, y, grupos, activos):
    pred = np.zeros(y.shape, dtype=float)
    auditoria = []
    for grupo in sorted(set(grupos)):
        test = np.asarray(grupos) == grupo
        train = ~test
        modelo = ajustar(x[train][activos[train]], y[train][activos[train]])
        pred[test] = puntuar(x[test].reshape(-1, x.shape[-1]), modelo).reshape((-1, len(TIPOS)))
        auditoria.append(dict(grupo_excluido=grupo, grupos_entrenamiento=sorted(set(np.asarray(grupos)[train]))))
    return pred, auditoria


def ordenar(scores, activos, margen=None):
    indices = [int(i) for i in np.argsort(-np.asarray(scores), kind="stable") if activos[i]]
    tipos = [TIPOS[i] for i in indices]
    if not tipos:
        return "sin_evidencia", [], None
    diferencia = float(scores[indices[0]]-scores[indices[1]]) if len(indices)>1 else None
    ambiguo = margen is not None and diferencia is not None and diferencia < margen
    return "ambiguo" if ambiguo else tipos[0], tipos[:3], diferencia


def ejecutar(salida):
    aqui = Path(__file__).parent
    fuente = aqui/"resultados_estrategias_desarrollo/predicciones.json"
    registros = json.loads(fuente.read_text(encoding="utf-8"))
    particion = pd.read_csv(aqui/"particion_desarrollo.csv")
    dev = set(particion.loc[particion.particion == "desarrollo", "archivo"])
    if {r["archivo"] for r in registros} != dev or len(registros) != len(dev):
        raise ValueError("La evidencia no corresponde exactamente al desarrollo")
    # Extraer características antes de consultar las etiquetas de evaluación/ajuste.
    pares = [caracteristicas(r) for r in registros]
    x, activos = np.array([p[0] for p in pares]), np.array([p[1] for p in pares])
    etiquetas = pd.read_csv(aqui/"resultados_catalogo_desarrollo/evaluacion.csv").set_index("archivo").real.to_dict()
    reales = [etiquetas[r["archivo"]] for r in registros]
    y = np.array([[float(real == tipo) for tipo in TIPOS] for real in reales])
    grupos = [grupo_condicion(r["archivo"]) for r in registros]
    aprendidos, auditoria = validar_grupos(x, y, grupos, activos)
    # Puntaje fijado antes del ensayo, sin usar etiquetas ni buscar combinaciones.
    manual = x[:,:,0] + x[:,:,2] + 2*x[:,:,3] + x[:,:,4] - .5*(1-x[:,:,6])
    filas = []
    for i, r in enumerate(registros):
        base = [c["tipo"] for c in r["variantes"]["sin_control"]]
        variantes = {"referencia": (base[0] if base else "sin_evidencia", base, None),
                     "coherencia": ordenar(manual[i], activos[i]),
                     "aprendido_oof": ordenar(aprendidos[i], activos[i]),
                     "aprendido_ambiguo": ordenar(aprendidos[i], activos[i], margen=.1)}
        for metodo, (primero, top3, diferencia) in variantes.items():
            filas.append(dict(archivo=r["archivo"], grupo=grupos[i], real=reales[i], metodo=metodo,
                prediccion=primero, top3=";".join(top3), margen=diferencia,
                acierto=reales[i] == primero, contiene_real=reales[i] in top3,
                emitido=primero in TIPOS))
    df = pd.DataFrame(filas)
    df.to_csv(salida/"predicciones_oof.csv", index=False)
    resumen = []
    for metodo, sub in df.groupby("metodo", sort=False):
        fallas, sanos = sub[sub.real != "N"], sub[sub.real == "N"]
        aceptadas = fallas[fallas.emitido]
        resumen.append(dict(metodo=metodo, aciertos=int(fallas.acierto.sum()), fallas=len(fallas),
            top1_pct=round(100*fallas.acierto.mean(), 2), top3=int(fallas.contiene_real.sum()),
            emitidos_fallas=len(aceptadas), cobertura_pct=round(100*len(aceptadas)/len(fallas), 2),
            acierto_emitidos_pct=round(100*aceptadas.acierto.mean(),2) if len(aceptadas) else None,
            ambiguos=int((fallas.prediccion == "ambiguo").sum()),
            alertas_sanos=int(sanos.emitido.sum()), sanos=len(sanos)))
    res = pd.DataFrame(resumen)
    res.to_csv(salida/"resumen.csv", index=False)
    modelo = ajustar(x[activos], y[activos])
    modelo.update(features=list(FEATURES), tipos=list(TIPOS), evidencia_sha256=hashlib.sha256(fuente.read_bytes()).hexdigest(),
                  nota="Ajustado en todo desarrollo para uso experimental; métricas reportadas fuera de grupo")
    (salida/"pesos_experimentales.json").write_text(json.dumps(modelo, indent=2), encoding="utf-8")
    (salida/"auditoria_grupos.json").write_text(json.dumps(auditoria, indent=2), encoding="utf-8")
    lineas = ["# Ordenación de hipótesis: desarrollo", "",
        f"{len(registros)} señales; {len(auditoria)} grupos por condición, uniendo cargas y sensores de la misma condición.",
        "Ridge alpha=1, sin búsqueda de hiperparámetros. Estandarización y ajuste solo en grupos de entrenamiento.",
        "Características compartidas entre tipos: fuerza, armónicas, dominancia, respaldo, exclusividad y contraste.",
        "La exclusividad aproxima solapamientos con frecuencias teóricas significativas: la caché no contiene posiciones exactas de todos los picos.",
        "Sin frecuencia esperada ni identidad del tipo como entrada al puntaje. Las etiquetas se usan solo para aprender pesos y evaluar.",
        "Abstención fija: margen menor que 0,1 unidades del score; el score NO es una probabilidad.",
        "Sin evidencia y ambiguo no significan sano. Se informa cobertura y aciertos condicionados a emitir diagnóstico.",
        "Resultados internos sobre desarrollo ya explorado, no validación externa. Reserva excluida.",
        "Los pesos ajustados en todo desarrollo no son los usados para puntuar el grupo excluido.", "",
        "| " + " | ".join(res.columns) + " |", "| " + " | ".join(["---"]*len(res.columns)) + " |"]
    lineas += ["| " + " | ".join(map(str,r)) + " |" for r in res.itertuples(index=False, name=None)]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n", encoding="utf-8")
    print(res.to_string(index=False))
    print("Grupos:",len(auditoria))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salida", type=Path, default=Path(__file__).parent/"resultados_ranking_desarrollo")
    args = p.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    ejecutar(args.salida)
