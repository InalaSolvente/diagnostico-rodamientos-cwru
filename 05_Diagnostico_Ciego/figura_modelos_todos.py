from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__=='__main__':
    p=Path(__file__).parent/'resultados_modelos_todos'
    d=pd.read_csv(p/'comparacion_48k.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4.8),constrained_layout=True)
    grupos=[('Desarrollo: fuera de condición',d[d.protocolo=='desarrollo_fuera_de_condicion'],['B','IR']),
            ('Reserva: sin copias de entrenamiento',d[(d.particion=='reserva')&d.pred_evaluacion.notna()],['B','IR','N'])]
    nombres={'B':'Bola','IR':'Interna','N':'Normal'}
    for ax,(titulo,g,reales) in zip(axs,grupos):
        t=pd.crosstab(g.real,g.pred_evaluacion).reindex(index=reales,columns=['B','IR','N'],fill_value=0).to_numpy()
        ax.imshow(t,cmap='Blues',vmin=0,vmax=6,aspect='auto')
        for i in range(len(reales)):
            for j in range(3):
                ax.text(j,i,str(t[i,j]),ha='center',va='center',fontsize=14,color='white' if t[i,j]>3 else 'black')
        ax.set_xticks(range(3),['Bola','Interna','Normal'])
        ax.set_yticks(range(len(reales)),[nombres[r] for r in reales])
        ax.set_xlabel('Primera predicción');ax.set_ylabel('Tipo real')
        n=int(g.correcto_evaluacion.sum())
        ax.set_title(titulo+f'\n{n}/{len(g)} correctos ({100*n/len(g):.2f}%)',fontsize=11)
    fig.suptitle('Modelo propio de 48 kHz · sin frecuencia de falla indicada\nReserva por archivos con condiciones compartidas: no validación externa',fontsize=11)
    fig.savefig(p/'validacion_48k.png',dpi=180);plt.close(fig)
