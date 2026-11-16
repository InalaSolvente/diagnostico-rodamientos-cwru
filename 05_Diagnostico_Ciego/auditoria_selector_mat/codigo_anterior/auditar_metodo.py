"""Auditoría reproducible; no cambia predictor, pesos ni resultados históricos."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_espectral import extraer
from modelo_ciego import F
from ranking_catalogo import grupo_condicion
from ensayar_espectral import modelos


if __name__=="__main__":
    aqui=Path(__file__).parent
    salida=aqui/"auditoria_metodologica"
    salida.mkdir(exist_ok=True)
    artefactos=aqui/"resultados_espectrales_desarrollo"
    modelo=artefactos/"modelo.npz"
    hash_antes=hashlib.sha256(modelo.read_bytes()).hexdigest()
    d=pd.read_csv(aqui/"resultados_paper_total/comparacion.csv")
    split=pd.read_csv(aqui/"particion_desarrollo.csv")
    meta=pd.read_csv(artefactos/"muestra.csv")
    dev=set(split.loc[split.particion=="desarrollo","archivo"])
    reserva=set(split.loc[split.particion=="reserva","archivo"])
    assert dev.isdisjoint(reserva) and set(meta.archivo)==dev
    assert len(d)==105 and d.archivo.is_unique
    grupos_dev={grupo_condicion(a) for a in dev}
    reservaf=split[(split.particion=="reserva")&(split.carpeta!="Normal")]
    solapados=reservaf.archivo.map(grupo_condicion).isin(grupos_dev)
    incorrectos_contados=d[(~d.tipo_correcto)&d.sin_retroceso]
    # Esta variante exige coherencia con la salida emitida: no transforma un
    # diagnóstico incorrecto PD/D en un acierto ND usando la verdad de referencia.
    estricto=((d.paper=="ND")&(d.grado_emitido=="ND")) | (d.tipo_correcto&d.sin_retroceso)
    hechos=dict(total=105,sin_retroceso_publicado=int(d.sin_retroceso.sum()),
        tipo_correcto=int(d.tipo_correcto.sum()),sin_retroceso_sin_aciertos_artificiales=int(estricto.sum()),
        incorrectos_contados=incorrectos_contados[["registro","paper","prediccion","tipo_real","grado_emitido","grado_evaluable"]].to_dict("records"),
        nd_automaticos=int((d.paper=="ND").sum()),grados_emitidos=d.grado_emitido.value_counts().to_dict(),
        reserva_fallas_total=len(reservaf),reserva_fallas_condicion_ya_vista=int(solapados.sum()),
        reserva_condicion_no_vista=reservaf.loc[~solapados,"archivo"].tolist(),
        dimensiones_arbol=1953,senales_entrenamiento=54,condiciones_desarrollo=meta.grupo.nunique(),
        clases_normal_fold=sorted(set(meta.loc[meta.grupo!="Normal","real"])),
        protocolos=d.groupby("origen").agg(total=("archivo","size"),tipo_correcto=("tipo_correcto","sum"),
                                           sin_retroceso=("sin_retroceso","sum")).to_dict("index"))
    datos=aqui.parent/"data2/CWRU"
    indice=F.leer_indice(str(datos)).set_index("archivo")
    with np.load(artefactos/"bancos_desarrollo.npz",allow_pickle=False) as bancos:
        x=bancos["arbol"]
    max_error=0.
    for i,r in enumerate(meta.itertuples()):
        m=indice.loc[r.archivo]
        mat=loadmat(datos/r.archivo)
        clave=next(k for k in mat if k.endswith(f"_{m.sensor_falla}_time"))
        krpm=[k for k in mat if k.endswith("RPM")]
        rpm=float(np.ravel(mat[krpm[0]])[0]) if krpm else float(m.rpm_nominal)
        c={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[m.sensor_falla].items()}
        recalculado=extraer(np.ravel(mat[clave]),float(m.fs),c)["arbol"]
        assert np.allclose(recalculado,x[i],rtol=1e-10,atol=1e-10)
        max_error=max(max_error,float(np.max(np.abs(recalculado-x[i]))))
        if (i+1)%10==0 or i+1==len(meta): print(f"Rasgos auditados {i+1}/{len(meta)}",flush=True)
    ajuste=modelos()["lda"]()
    ajuste.fit(x,meta.real.to_numpy())
    scaler,lda=ajuste.steps[0][1],ajuste.steps[-1][1]
    with np.load(modelo,allow_pickle=False) as exportado:
        assert np.allclose(scaler.mean_,exportado["media"])
        assert np.allclose(scaler.scale_,exportado["escala"])
        assert np.allclose(lda.coef_,exportado["coef"])
        assert np.allclose(lda.intercept_,exportado["intercepto"])
        assert np.array_equal(lda.classes_,exportado["clases"])
    hechos.update(rasgos_recalculados=len(meta),max_error_rasgos=max_error,
                  pesos_reproducidos_desde_desarrollo=True,modelo_sha256=hash_antes)
    assert hashlib.sha256(modelo.read_bytes()).hexdigest()==hash_antes
    assert hechos["sin_retroceso_publicado"]==99 and int(estricto.sum())==96
    (salida/"hechos_verificados.json").write_text(json.dumps(hechos,indent=2),encoding="utf-8")
    d.assign(sin_retroceso_estricto=estricto).to_csv(salida/"recuento_estricto.csv",index=False)
    print("OK: rasgos desde señales reales, pesos desde desarrollo, hash intacto; 99 publicados / 96 estrictos",flush=True)
