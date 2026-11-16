"""Aplica la regla del usuario a tipos ya predichos; no entrena ni elige modelos."""
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from evaluar_no_retroceso import calificar,F,BENCH,PAPER
from criterio_usuario import comparar
from predictor_espectral import MODELO,predecir


if __name__=="__main__":
    aqui=Path(__file__).parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--todos",action="store_true",help="Todos los archivos del índice, incluidos 48k y normales")
    args=parser.parse_args()
    salida=aqui/("resultados_criterio_usuario_total" if args.todos else "resultados_criterio_usuario_desarrollo")
    salida.mkdir(exist_ok=True)
    firma=hashlib.sha256(MODELO.read_bytes()).hexdigest()
    pred=pd.read_csv(aqui/"resultados_espectrales_desarrollo/verificacion_oof.csv")
    split=pd.read_csv(aqui/"particion_desarrollo.csv")
    dev=set(split.loc[split.particion=="desarrollo","archivo"])
    assert set(pred.archivo)==dev and pred.archivo.is_unique
    datos=aqui.parent/"data2/CWRU"
    idx=F.leer_indice(str(datos)).set_index("archivo")
    if args.todos:
        anterior=aqui/"resultados_paper_total"
        config_anterior=json.loads((anterior/"configuracion.json").read_text(encoding="utf-8"))
        assert config_anterior["modelo_sha256"]==firma,"El modelo cambió respecto a las predicciones guardadas"
        cache_tipo=pred.set_index("archivo").prediccion.to_dict()
        for r in pd.read_csv(anterior/"comparacion.csv").itertuples():
            cache_tipo.setdefault(r.archivo,r.prediccion)
        pred=pd.DataFrame(dict(archivo=idx.index,prediccion=[cache_tipo.get(a) for a in idx.index]))
        assert len(pred)==len(idx) and pred.archivo.is_unique
    origenes=split.set_index("archivo").particion.to_dict()
    filas=[]; evidencias=[]
    for _,r in pred.iterrows():
        meta=idx.loc[r.archivo]
        mat=loadmat(datos/r.archivo)
        clave=F.clave_canal(mat, meta.sensor_falla, int(meta.registro))
        krpm=[k for k in mat if k.endswith("RPM")]
        rpm=F.rpm_registro(mat, int(meta.registro), nominal=float(meta.rpm_nominal))
        catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[meta.sensor_falla].items()}
        tipo=r.prediccion
        if pd.isna(tipo):
            tipo=predecir(np.ravel(mat[clave]),float(meta.fs),catalogo)["tipo"]
        q=calificar(np.ravel(mat[clave]),float(meta.fs),catalogo,tipo,criterio="usuario")
        evidencia=q.get("evidencia",{})
        evidencias.append(dict(archivo=r.archivo,tipo_predicho=tipo,**q))
        # El grado está emitido antes de consultar tipo real y referencia.
        referencia=PAPER[BENCH[int(meta.registro)]] if meta.carpeta in ("12k_DE","12k_FE") else None
        tipo_correcto=tipo==meta.zona
        evaluable,exito=comparar(referencia,q["grado"],tipo_correcto) if referencia else (None,None)
        filas.append(dict(archivo=r.archivo,registro=int(meta.registro),paper=referencia,
            tipo_predicho=tipo,tipo_real=meta.zona,tipo_correcto=tipo_correcto,
            carpeta=meta.carpeta,fs=float(meta.fs),
            origen="desarrollo_oof" if origenes.get(r.archivo)=="desarrollo" else "reserva_congelado" if r.archivo in origenes else "adicional_48k",
            grado_emitido=q["grado"],grado_evaluable=evaluable,sin_retroceso=exito,
            familia=q.get("familia"),nodo=str(q.get("nodo")),
            armonicas=";".join(map(str,evidencia.get("identificadas",[]))),
            primera_identificada=evidencia.get("primera_identificada",False),
            pico_previo_mayor=evidencia.get("hay_pico_previo_mayor",False)))
        print(f"{len(filas)}/{len(pred)} {meta.registro}: {q['grado']}",flush=True)
        pd.DataFrame(filas).to_csv(salida/"comparacion_parcial.csv",index=False)
    df=pd.DataFrame(filas)
    assert hashlib.sha256(MODELO.read_bytes()).hexdigest()==firma
    df.to_csv(salida/"comparacion.csv",index=False)
    (salida/"evidencia.json").write_text(json.dumps(evidencias,indent=2,allow_nan=False),encoding="utf-8")
    fallas=df[df.paper.notna()]; sanos=df[df.tipo_real=="N"]
    adicionales=df[df.paper.isna()&(df.tipo_real!="N")]
    resumen=[]
    for nombre,sub in [("con_referencia_paper_12k",fallas),("adicionales_sin_referencia_48k",adicionales),("normales",sanos)]:
        resumen.append(dict(conjunto=nombre,total=len(sub),D=int((sub.grado_emitido=="D").sum()),
            PD=int((sub.grado_emitido=="PD").sum()),ND=int((sub.grado_emitido=="ND").sum()),
            tipo_correcto=int(sub.tipo_correcto.sum()),
            sin_retroceso=int(sub.sin_retroceso.sum()) if nombre=="con_referencia_paper_12k" else None))
    pd.DataFrame(resumen).to_csv(salida/"resumen.csv",index=False)
    pd.crosstab(fallas.paper,fallas.grado_evaluable).reindex(index=["ND","PD","D"],columns=["ND","PD","D","tipo_incorrecto"],fill_value=0).to_csv(salida/"transiciones.csv")
    configuracion=dict(modelo_sha256=firma,criterio="usuario_v1",H=6,tol=.02,p=.01,
        dominancia="solo_izquierda",min_armonicas_d=2,min_armonicas_pd=1,
        primera_requerida_d=True,primera_requerida_pd=False,dc_excluido=True,
        referencia_pd="primera armónica identificada, nunca ruido en la ventana de h=1",
        primera_ventana="+-2%: región excluida al comparar izquierda",empates="no son mayor",
        correccion="31 nodos x 5 familias x 6 armónicas",reentrenamiento=False,
        archivos=df.archivo.tolist(),todos=args.todos,
        origen_prediccion="OOF en desarrollo; modelo congelado en reserva y registros adicionales")
    (salida/"configuracion.json").write_text(json.dumps(configuracion,indent=2),encoding="utf-8")
    lineas=["# Criterio del usuario: "+("todos los registros" if args.todos else "desarrollo"), "",
        "D: primera identificada, al menos dos armónicas, ningún pico previo mayor; los picos a la derecha no impiden D.",
        "PD: una o más armónicas identificadas y un pico previo mayor; la primera puede faltar.",
        "Si falta h=1, la comparación PD se hace contra la primera armónica realmente identificada, nunca contra ruido en la fundamental.",
        "ND: todo lo demás. Se excluye DC. Armónicas: picos locales dentro de ±2%, significativos contra el fondo local.",
        "Se buscan hasta seis armónicas por hipótesis, con la corrección nominal documentada en configuracion.json.",
        "La primera es respecto a la frecuencia base de la familia declarada (BSF y 2BSF son hipótesis explícitas diferentes).",
        "El tipo de falla permanece congelado: OOF en desarrollo; modelo ajustado en desarrollo para reserva y adicionales.",
        "Un tipo incorrecto con PD/D se informa como tipo_incorrecto; nunca se transforma a ND para sumar un acierto.",
        "Estos grados no se presentan como homologación de la rúbrica del paper.", "",
        f"Señales ejecutadas: {len(df)}. Con referencia del paper: {len(fallas)}. Sin retroceso estricto: {int(fallas.sin_retroceso.sum())}/{len(fallas)}.",
        "Grados emitidos en fallas: "+str(fallas.grado_emitido.value_counts().to_dict()),
        "Grados emitidos en normales: "+str(sanos.grado_emitido.value_counts().to_dict()),
        "Grados emitidos en adicionales 48k sin referencia transcrita: "+str(adicionales.grado_emitido.value_counts().to_dict()),
        "Los normales y adicionales 48k no se confunden con ND del paper ni se suman al porcentaje de no retroceso.",
        "La comparación del paper mezcla protocolos de desarrollo/reserva y no es validación externa independiente.",
        "No hubo entrenamiento ni ajuste de parámetros con estos resultados."]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n",encoding="utf-8")
    print("\n".join(lineas))
