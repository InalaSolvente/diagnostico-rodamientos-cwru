"""Prueba focalizada en 48 kHz: cambia geometría de bandas, conserva pesos y catálogo."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from alinear_bandas import predecir_alineado
from evaluar_no_retroceso import F,calificar
from predictor_espectral import MODELO


if __name__=="__main__":
    aqui=Path(__file__).parent;datos=aqui.parent/"data2/CWRU"
    salida=aqui/"resultados_L_alineado"
    salida.mkdir(exist_ok=True)
    base=pd.read_csv(aqui/"resultados_criterio_usuario_total/comparacion.csv").set_index("archivo")
    idx=F.leer_indice(str(datos));idx=idx[idx.fs==48000]
    firma=hashlib.sha256(MODELO.read_bytes()).hexdigest()
    filas=[];evidencias=[]
    for _,r in idx.iterrows():
        mat=loadmat(datos/r.archivo)
        clave=F.clave_canal(mat, r.sensor_falla, int(r.registro))
        krpm=[k for k in mat if k.endswith("RPM")]
        rpm=F.rpm_registro(mat, int(r.registro), nominal=float(r.rpm_nominal))
        c={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
        x=np.ravel(mat[clave]);pred=predecir_alineado(x,float(r.fs),c)
        q=calificar(x,float(r.fs),c,pred["tipo"],L=pred["L"],criterio="usuario",nodos=pred["nodos"])
        anterior=base.loc[r.archivo]
        # La etiqueta se usa solo después de inferencia y grado.
        filas.append(dict(archivo=r.archivo,carpeta=r.carpeta,real=r.zona,
            pred_antes=anterior.tipo_predicho,pred_ahora=pred["tipo"],
            correcto_antes=anterior.tipo_predicho==r.zona,correcto_ahora=pred["tipo"]==r.zona,
            grado_antes=anterior.grado_emitido,grado_ahora=q["grado"],L=pred["L"],
            ancho_final=pred["ancho_final"]))
        evidencias.append(dict(archivo=r.archivo,L=pred["L"],nodos=pred["nodos"],tipo=pred["tipo"],calidad=q))
        pd.DataFrame(filas).to_csv(salida/"comparacion_parcial.csv",index=False)
        print(f"{len(filas)}/{len(idx)} {r.archivo}: {anterior.tipo_predicho} -> {pred['tipo']} ({q['grado']})",flush=True)
    d=pd.DataFrame(filas);d.to_csv(salida/"comparacion.csv",index=False)
    (salida/"evidencia.json").write_text(json.dumps(evidencias,indent=2,allow_nan=False),encoding="utf-8")
    assert hashlib.sha256(MODELO.read_bytes()).hexdigest()==firma
    res=[]
    for carpeta,g in d.groupby("carpeta"):
        res.append(dict(carpeta=carpeta,total=len(g),correctos_antes=int(g.correcto_antes.sum()),
            correctos_ahora=int(g.correcto_ahora.sum()),D=int((g.grado_ahora=="D").sum()),
            PD=int((g.grado_ahora=="PD").sum()),ND=int((g.grado_ahora=="ND").sum())))
    pd.DataFrame(res).to_csv(salida/"resumen.csv",index=False)
    config=dict(modelo_sha256=firma,reentrenamiento=False,fs_referencia=12000,L_referencia=4,
        L_48k=6,ancho_final_hz=375,nodos_usados=31,nodos_arbol_completo=127,
        rango_usado_hz=[0,6000],remuestreo=False,normalizacion="RMS de la señal completa como antes",
        limitacion="No usa 6-24k; densidad de bins y entropía todavía dependen del muestreo")
    (salida/"configuracion.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    lineas=["# Prueba de alineación física de bandas", "",
        "A 48 kHz se usa L=6 para alcanzar 375 Hz, igual que 12 kHz/L4.",
        "El modelo congelado recibe los 31 nodos físicamente equivalentes de 0–6 kHz (niveles globales 2–6).",
        "No es usar todo el árbol de 127 nodos: su dimensionalidad sería incompatible con los pesos existentes.",
        "Sin remuestreo, ajuste de umbrales ni entrenamiento. RMS global, soporte de entropía y densidad espectral siguen dependiendo de fs.",
        "La calidad se evalúa sobre esos mismos 31 nodos y con la regla del usuario.",
        "Las señales a 12 kHz permanecen iguales por construcción y por prueba de equivalencia numérica.",
        "Los normales de desarrollo habían entrenado con la geometría antigua; esta prueba cambia sus rasgos sin recalibrar pesos.","",
        "| carpeta | total | tipos correctos antes | tipos correctos ahora | D | PD | ND |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    lineas += ["| "+" | ".join(map(str,r.values()))+" |" for r in res]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n",encoding="utf-8")
    print(pd.DataFrame(res).to_string(index=False))
