"""Repite la validación fuera de grupo del ganador y exporta coeficientes numéricos."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ensayar_espectral import modelos
from integridad_senales import huellas_senales, exigir_independencia
from modelo_ciego import F


if __name__=="__main__":
    aqui=Path(__file__).parent
    salida=aqui/"resultados_espectrales_desarrollo"
    res=pd.read_csv(salida/"resumen.csv")
    ganador=res.iloc[0]
    banco,nombre=str(ganador.banco),str(ganador.modelo)
    meta=pd.read_csv(salida/"muestra.csv")
    if 'version_datos' not in meta or not meta.version_datos.eq('selector_explicito_cwru_v1').all():
        raise ValueError('Rasgos históricos anteriores al selector MAT corregido: reextraiga en un experimento nuevo')
    particion=pd.read_csv(aqui/"particion_desarrollo.csv")
    dev=set(particion.loc[particion.particion=="desarrollo","archivo"])
    reserva=set(particion.loc[particion.particion=="reserva","archivo"])
    assert len(meta)==54 and meta.archivo.is_unique and set(meta.archivo)==dev and not dev&reserva
    datos=aqui.parent/'data2/CWRU'
    indice=F.leer_indice(str(datos))
    exigir_independencia(huellas_senales(indice[indice.archivo.isin(dev|reserva)], datos), dev, reserva)
    with np.load(salida/"bancos_desarrollo.npz",allow_pickle=False) as datos:
        x=datos[banco]
    assert np.isfinite(x).all()
    y=meta.real.to_numpy(); grupos=meta.grupo.to_numpy()
    pred=np.empty(len(y),dtype=object); pruebas=[]
    for grupo in sorted(set(grupos)):
        train=grupos!=grupo; test=~train
        clf=modelos()[nombre]()
        clf.fit(x[train],y[train])
        pred[test]=clf.predict(x[test])
        assert not set(meta.loc[train,"archivo"])&set(meta.loc[test,"archivo"])
        assert grupo not in set(grupos[train])
        pruebas.append(dict(grupo=grupo,entrenamiento=meta.loc[train,"archivo"].tolist(),
                            prueba=meta.loc[test,"archivo"].tolist()))
    anteriores=pd.read_csv(salida/"predicciones_oof.csv")
    anteriores=anteriores[(anteriores.banco==banco)&(anteriores.modelo==nombre)].set_index("archivo")
    assert np.array_equal(pred,anteriores.loc[meta.archivo,"prediccion"].to_numpy())
    fallas=y!="N"
    aciertos=int((pred[fallas]==y[fallas]).sum()); total=int((pred==y).sum())
    # La validez de la ejecución no depende de superar un porcentaje de acierto.
    # Ajuste final solo en desarrollo; sus resultados in-sample NO se reportan como validación.
    final=modelos()[nombre]()
    final.fit(x,y)
    scaler,lda=final.steps[0][1],final.steps[-1][1]
    assert nombre=="lda", "Exportador numérico implementado para el ganador LDA"
    scores=((x-scaler.mean_)/scaler.scale_)@lda.coef_.T+lda.intercept_
    assert np.array_equal(lda.classes_[scores.argmax(axis=1)],final.predict(x))
    np.savez_compressed(salida/"modelo.npz",media=scaler.mean_,escala=scaler.scale_,
        coef=lda.coef_,intercepto=lda.intercept_,clases=lda.classes_,banco=banco,L=4,
        extractor_sha256=hashlib.sha256((aqui/"modelo_espectral.py").read_bytes()).hexdigest())
    pd.DataFrame(dict(archivo=meta.archivo,grupo=grupos,real=y,prediccion=pred,
                      acierto=y==pred)).to_csv(salida/"verificacion_oof.csv",index=False)
    (salida/"verificacion_grupos.json").write_text(json.dumps(pruebas,indent=2),encoding="utf-8")
    clases=["IR","OR","B","N"]
    cm=confusion_matrix(y,pred,labels=clases)
    pd.DataFrame(cm,index=clases,columns=clases).to_csv(salida/"matriz_confusion.csv")
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=max(cm.max(),1))
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,
                    color="white" if cm[i,j]>cm.max()/2 else "black")
    etiquetas=["Interna","Externa","Bola","Normal"]
    ax.set_xticks(range(4),etiquetas); ax.set_yticks(range(4),etiquetas)
    ax.set_xlabel("Primera predicción"); ax.set_ylabel("Etiqueta real (solo evaluación)")
    ax.set_title(f"Validación por 16 grupos · desarrollo\n{aciertos}/52 fallas ({100*aciertos/52:.2f}%) · {total}/54 total ({100*total/54:.2f}%)")
    fig.savefig(salida/"clasificacion_sin_fd.png",dpi=180); plt.close(fig)
    lineas=["# Modelo espectral sin frecuencia de falla esperada", "",
        f"Ganador entre {len(res)} ensayos: banco {banco}, clasificador {nombre}.",
        f"Primera opción correcta: {aciertos}/52 fallas ({100*aciertos/52:.2f}%). Total: {total}/54 ({100*total/54:.2f}%).",
        f"Normales correctamente identificados: {int((pred[~fallas]==y[~fallas]).sum())}/2.",
        "Se repitió la predicción dejando fuera cada uno de los 16 grupos; normalización y ajuste solo en entrenamiento.",
        "Las cargas y los sensores de una condición permanecen en el mismo grupo.",
        "Entrada de inferencia: señal, muestreo y catálogo completo BPFI/BPFO/BSF/FTF; nunca defecto real ni fd esperada.",
        "La etiqueta se utiliza para entrenar el clasificador, no como entrada al predecir.",
        "L=4 fijo. Seis armónicas para todas las familias y los 31 nodos, más energía, curtosis y entropía de cada nodo.",
        "Los 55 archivos de reserva no se cargaron. La reserva ya se había observado históricamente.",
        "Los parámetros y el ganador se eligieron explorando desarrollo: el porcentaje no es estimación independiente ni prueba externa.",
        "El clasificador aprendió IR/OR/B/N; no hay ejemplos de falla de jaula, por lo que no está validado para FTF como clase.",
        "Los dos normales no se reconocieron fuera de grupo; no sirve aún como detector validado sano/falla.",
        "El modelo exportado se ajustó con todo desarrollo; sus aciertos de entrenamiento no sustentan los porcentajes anteriores.", "",
        "## Ensayos", "", "| " + " | ".join(res.columns) + " |",
        "| " + " | ".join(["---"]*len(res.columns)) + " |"]
    lineas += ["| " + " | ".join(map(str,r)) + " |" for r in res.itertuples(index=False,name=None)]
    (salida/"resumen.md").write_text("\n".join(lineas)+"\n",encoding="utf-8")
    print(f"VERIFICADO: {aciertos}/52, {total}/54, 16 grupos, reserva excluida, exportación equivalente")
