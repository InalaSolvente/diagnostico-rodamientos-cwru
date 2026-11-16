"""Figura del ensayo completo con la regla ND/PD/D del usuario."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


if __name__=="__main__":
    p=Path(__file__).parent/"resultados_criterio_usuario_total"
    d=pd.read_csv(p/"comparacion.csv")
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    nombres=["Fallas 12 kHz (105)","Fallas 48 kHz (24)","Normales (4)"]
    grupos=[d[d.paper.notna()],d[(d.carpeta=="48k_DE")],d[d.carpeta=="Normal"]]
    base=np.zeros(3)
    for grado,color in [("D","#215eaa"),("PD","#8db4e2"),("ND","#dedede")]:
        valores=np.array([int((g.grado_emitido==grado).sum()) for g in grupos])
        axs[0].barh(range(3),valores,left=base,color=color,label=grado)
        for i,(n,b) in enumerate(zip(valores,base)):
            if n>=5: axs[0].text(b+n/2,i,str(n),ha="center",va="center",color="white" if grado=="D" else "black")
        base+=valores
    axs[0].text(base[1]+1,1,"PD: 1",ha="left",va="center",fontsize=9)
    axs[0].text(base[2]+1,2,"PD: 2 · ND: 2",ha="left",va="center",fontsize=9)
    axs[0].set_yticks(range(3),nombres); axs[0].invert_yaxis()
    axs[0].set_xlabel("Cantidad de señales"); axs[0].set_title("Grados emitidos por el método")
    axs[0].legend(frameon=False,ncol=3,loc="lower right")
    t=pd.crosstab(grupos[0].paper,grupos[0].grado_evaluable).reindex(index=["ND","PD","D"],columns=["ND","PD","D","tipo_incorrecto"],fill_value=0).to_numpy()
    axs[1].imshow(t,cmap="Blues",vmin=0,vmax=max(1,t.max()),aspect="auto")
    for i in range(3):
        for j in range(4):
            axs[1].text(j,i,str(t[i,j]),ha="center",va="center",color="white" if t[i,j]>t.max()/2 else "black")
    axs[1].set_xticks(range(4),["ND","PD","D","Tipo\nincorrecto"])
    axs[1].set_yticks(range(3),["ND","PD","D"])
    axs[1].set_xlabel("Resultado evaluado"); axs[1].set_ylabel("Referencia del paper")
    axs[1].set_title("90/105 sin retroceso (85,71%)\nProtocolos de desarrollo/reserva combinados")
    fig.suptitle("133 señales · criterio explícito del usuario · modelo congelado",fontsize=12)
    fig.savefig(p/"comparacion_completa.png",dpi=180); plt.close(fig)
