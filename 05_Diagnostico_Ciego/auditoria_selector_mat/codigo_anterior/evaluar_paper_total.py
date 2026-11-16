"""Evaluación final del modelo congelado: desarrollo OOF + reserva, sin reajuste."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from predictor_espectral import predecir, MODELO
from evaluar_no_retroceso import calificar,comparar_grado,PAPER,BENCH,F


if __name__=="__main__":
    aqui=Path(__file__).parent
    salida=aqui/"resultados_paper_total"
    salida.mkdir(exist_ok=True)
    datos=aqui.parent/"data2/CWRU"
    indice=F.leer_indice(str(datos))
    indice=indice[indice.carpeta.isin(["12k_DE","12k_FE"])]
    particion=pd.read_csv(aqui/"particion_desarrollo.csv").set_index("archivo").particion.to_dict()
    desarrollo=pd.read_csv(aqui/"resultados_no_retroceso_desarrollo/comparacion.csv").set_index("archivo")
    firma=hashlib.sha256(MODELO.read_bytes()).hexdigest()
    filas=[]
    for _,meta in indice.iterrows():
        if particion[meta.archivo]=="desarrollo":
            anterior=desarrollo.loc[meta.archivo]
            fila=anterior.to_dict()
            fila.update(archivo=meta.archivo,origen="desarrollo_oof")
        else:
            mat=loadmat(datos/meta.archivo)
            clave=next(k for k in mat if k.endswith(f"_{meta.sensor_falla}_time"))
            rpm_keys=[k for k in mat if k.endswith("RPM")]
            rpm=float(np.ravel(mat[rpm_keys[0]])[0]) if rpm_keys else float(meta.rpm_nominal)
            catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[meta.sensor_falla].items()}
            x=np.ravel(mat[clave]); fs=float(meta.fs)
            pred=predecir(x,fs,catalogo)
            calidad=calificar(x,fs,catalogo,pred["tipo"])
            # Predicción y grado fijados antes de consultar defecto real y paper.
            referencia=PAPER[BENCH[int(meta.registro)]]
            correcto=pred["tipo"]==meta.zona
            grado,exito=comparar_grado(referencia,calidad["grado"],correcto)
            fila=dict(archivo=meta.archivo,registro=int(meta.registro),paper=referencia,
                prediccion=pred["tipo"],tipo_real=meta.zona,tipo_correcto=correcto,
                grado_emitido=calidad["grado"],grado_evaluable=grado,sin_retroceso=exito,origen="reserva_congelado")
        # Recalcular también las filas históricas de desarrollo: no heredar ND fabricados.
        fila['grado_evaluable'],fila['sin_retroceso']=comparar_grado(
            fila['paper'],fila['grado_emitido'],fila['tipo_correcto'])
        filas.append(fila)
        pd.DataFrame(filas).to_csv(salida/"comparacion.csv",index=False)
        print(f"{len(filas)}/{len(indice)} {int(meta.registro)}: {fila['paper']} -> {fila['grado_evaluable']} ({fila['origen']})",flush=True)
    df=pd.DataFrame(filas)
    assert len(df)==105 and df.archivo.is_unique
    assert hashlib.sha256(MODELO.read_bytes()).hexdigest()==firma
    resumen=[]
    for nombre,sub in [("desarrollo_oof",df[df.origen=="desarrollo_oof"]),
                       ("reserva_congelado",df[df.origen=="reserva_congelado"]),("total_mixto",df)]:
        resumen.append(dict(conjunto=nombre,total=len(sub),sin_retroceso=int(sub.sin_retroceso.sum()),
            porcentaje=round(100*sub.sin_retroceso.mean(),2),tipo_correcto=int(sub.tipo_correcto.sum())))
    res=pd.DataFrame(resumen)
    res.to_csv(salida/"resumen.csv",index=False)
    pd.crosstab(df.paper,df.grado_evaluable).reindex(index=["ND","PD","D"],columns=["ND","PD","D","tipo_incorrecto"],fill_value=0).to_csv(salida/"transiciones.csv")
    df[~df.sin_retroceso].to_csv(salida/"retrocesos.csv",index=False)
    (salida/"configuracion.json").write_text(json.dumps(dict(modelo_sha256=firma,meta_casos=95,
        total=105,ajuste_con_reserva=False,p_calidad=.01,correccion="bandas x 5 familias x 3 armónicas"),indent=2),encoding="utf-8")
    lineas=["# No retroceso frente al paper: evaluación total", "",
        "Definición: ND→ND/PD/D, PD→PD/D, D→D. PD/D de tipo incorrecto es error, nunca ND fabricado.",
        "Solo un ND realmente emitido puede conservar ND de referencia sin exigir tipo correcto. Esto no es precisión diagnóstica.",
        "Desarrollo: predicciones fuera de grupo. Reserva: modelo congelado entrenado en desarrollo.",
        "El total mezcla ambos protocolos y no sustituye el resultado separado de reserva.",
        "La reserva fue observada históricamente y no es validación externa nueva.",
        "No se usó la frecuencia de falla esperada en inferencia ni al calificar evidencia; sí se comparó después el tipo real.",
        "Referencia S&R transcrita en el proyecto; el grado es operativo, no una reproducción de su algoritmo.", "",
        "| " + " | ".join(res.columns) + " |", "| " + " | ".join(["---"]*len(res.columns)) + " |"]
    lineas += ["| " + " | ".join(map(str,r)) + " |" for r in res.itertuples(index=False,name=None)]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n",encoding="utf-8")
    print(res.to_string(index=False))
