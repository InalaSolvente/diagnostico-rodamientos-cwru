"""Regenera las transiciones ordinales desde la evaluación total guardada."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__=="__main__":
    p=Path(__file__).parent/"resultados_paper_total"
    d=pd.read_csv(p/"comparacion.csv")
    fig,axs=plt.subplots(1,3,figsize=(12,4.8),constrained_layout=True)
    conjuntos=[("Desarrollo fuera de grupo",d[d.origen=="desarrollo_oof"]),
               ("Reserva: modelo congelado",d[d.origen=="reserva_congelado"]),("Total mixto",d)]
    for ax,(titulo,sub) in zip(axs,conjuntos):
        t=pd.crosstab(sub.paper,sub.grado_evaluable).reindex(index=["ND","PD","D"],columns=["ND","PD","D"],fill_value=0).to_numpy()
        ax.imshow(t,cmap="Blues",vmin=0,vmax=64,aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j,i,str(t[i,j]),ha="center",va="center",fontsize=13,color="white" if t[i,j]>32 else "black")
        ax.set_xticks(range(3),["ND","PD","D"]); ax.set_yticks(range(3),["ND","PD","D"])
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Modelo: grado evaluable",fontsize=9)
        ax.set_ylabel("Referencia del paper",fontsize=9)
        ax.set_title(titulo+f"\n{int(sub.sin_retroceso.sum())}/{len(sub)} sin retroceso",fontsize=10)
    fig.suptitle("Conservar o mejorar la clasificación del paper\nTipo incorrecto se evalúa como ND; no es precisión diagnóstica",fontsize=12)
    fig.savefig(p/"no_retroceso_paper.png",dpi=180)
    plt.close(fig)
