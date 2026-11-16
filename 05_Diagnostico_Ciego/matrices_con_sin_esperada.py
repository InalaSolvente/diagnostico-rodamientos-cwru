"""Comparación pareada: mejor método corregido frente a su oráculo de frecuencia.

No reentrena ni ajusta métodos. Solo recalcula los casos cuya familia predicha
difirió de la real, entregando esta última como información oracular explícita.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from modelo_ciego import F
from modelo_por_dominio import calidad
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
SALIDA=AQUI/'comparacion_con_sin_esperada'


if __name__=='__main__':
    SALIDA.mkdir(exist_ok=True)
    # Solo resultados corregidos, mismo criterio y comprobación temporal.
    candidatos=[]
    for nombre in ['resultados_verificados_desarrollo','resultados_verificados_ventanas',
                   'resultados_verificados_fisicos','resultados_por_dominio',
                   'resultados_jerarquico','resultados_jerarquico_laterales']:
        p=AQUI/nombre
        if not (p/'comparacion.csv').exists():continue
        d=pd.read_csv(p/'comparacion.csv');f=d[d.paper.notna()]
        if len(f)!=52:continue
        delta=f.grado.map(ORDEN)-f.paper.map(ORDEN);ok=f.tipo_correcto.eq(True)
        normal=d[d.real.eq('N')]
        alarmas=int((normal.prediccion.ne('N')&normal.grado.ne('ND')).sum())
        candidatos.append((int((ok&delta.ge(0)).sum()),-alarmas,nombre,p,d))
    # Empate: menos falsas alarmas observadas; orden estable, sin escoger casos individualmente.
    ganador=max(candidatos,key=lambda r:(r[0],r[1]))
    _,_,nombre,p,d=ganador
    f=d[d.paper.notna()].copy().sort_values('archivo')
    assert len(f)==52 and f.archivo.is_unique
    evidencia=json.loads((p/'predicciones.json').read_text(encoding='utf-8'))
    por_archivo={r['archivo']:r for r in evidencia if r['archivo'] in set(f.archivo)}
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo',drop=False)
    filas=[];trazas=[]
    for i,r in enumerate(f.itertuples()):
        if r.prediccion==r.real:
            grado=r.grado;q=por_archivo[r.archivo]
        else:
            m=idx.loc[r.archivo];mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            x=np.ravel(mat[F.clave_canal(mat,m.sensor_falla,int(m.registro))])
            rpm=F.rpm_registro(mat,int(m.registro),nominal=float(m.rpm_nominal))
            catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[m.sensor_falla].items()}
            # Aquí la familia verdadera se entrega deliberadamente: es el escenario oracular.
            q=calidad(x,int(m.fs),catalogo,r.real);grado=q['grado']
            print(f'Oráculo {i+1}/52: {r.archivo}',flush=True)
        filas.append(dict(archivo=r.archivo,paper=r.paper,tipo_real=r.real,
            tipo_sin_esperada=r.prediccion,grado_sin_esperada=r.grado,
            tipo_correcto=r.tipo_correcto,grado_con_esperada=grado))
        trazas.append(dict(archivo=r.archivo,oraculo=q))
    datos=pd.DataFrame(filas)
    assert datos.archivo.is_unique and len(datos)==52
    datos.to_csv(SALIDA/'comparacion_pareada.csv',index=False)
    (SALIDA/'evidencia_oraculo.json').write_text(json.dumps(trazas,ensure_ascii=False,indent=2),encoding='utf-8')
    colores={'mejora':'#ccebd7','igual':'#dce8f4','baja':'#f5d2cc','error':'#e1e1e1'}
    grados=['ND','PD','D'];columnas=grados+['Tipo incorrecto']
    fig,axs=plt.subplots(1,2,figsize=(14,6.6));resumen=[]
    for ax,(titulo,col,oraculo) in zip(axs,[('Con frecuencia esperada · ORÁCULO','grado_con_esperada',True),
                                           ('Sin frecuencia esperada','grado_sin_esperada',False)]):
        actual=datos[col] if oraculo else datos[col].where(datos.tipo_correcto.eq(True),'Tipo incorrecto')
        t=pd.crosstab(datos.paper,actual).reindex(index=grados,columns=columnas,fill_value=0)
        assert t.to_numpy().sum()==52
        mejora=sum(int(t.iloc[i,j]) for i in range(3) for j in range(3) if j>i)
        igual=sum(int(t.iloc[i,i]) for i in range(3))
        baja=sum(int(t.iloc[i,j]) for i in range(3) for j in range(3) if j<i)
        error=int(t.iloc[:,3].sum());nr=mejora+igual
        assert mejora+igual+baja+error==52
        resumen.append(dict(escenario=titulo,total=52,mejoran=mejora,iguales=igual,empeoran=baja,
                            tipo_incorrecto=error,sin_retroceso=nr,porcentaje=100*nr/52))
        t.to_csv(SALIDA/('matriz_con_esperada.csv' if oraculo else 'matriz_sin_esperada.csv'))
        for i in range(3):
            for j in range(4):
                estado='error' if j==3 else 'mejora' if j>i else 'baja' if j<i else 'igual'
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=colores[estado],edgecolor='white',linewidth=2))
                n=int(t.iloc[i,j]);total=int(t.iloc[i].sum())
                ax.text(j,i,f'{n}\n({100*n/total:.1f} %)',ha='center',va='center',fontsize=11)
        ax.set_xlim(-.5,3.5);ax.set_ylim(2.5,-.5)
        ax.set_xticks(range(4),['ND','PD','D','Tipo\nincorrecto']);ax.set_yticks(range(3),grados)
        ax.set_xlabel('Diagnóstico obtenido');ax.set_ylabel('Clasificación del paper')
        ax.set_title(f'{titulo}\n{nr}/52 sin retroceso ({100*nr/52:.2f} %)',fontsize=12,pad=14)
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.text(.5,-.29,f'Mejoran {mejora} · Iguales {igual}\nEmpeoran {baja} · Tipo incorrecto {error}',
                transform=ax.transAxes,ha='center',fontsize=10,linespacing=1.5)
    fig.suptitle('Mejor resultado corregido disponible y su contraparte con frecuencia conocida',fontsize=15,y=.97)
    fig.text(.5,.89,'Mismos 52 casos de desarrollo de 12 kHz · misma regla ND/PD/D · misma comprobación temporal',ha='center',fontsize=11)
    fig.text(.5,.045,'La izquierda conoce la familia real antes de elegir la banda: es una referencia con información privilegiada.\n'
             'La derecha usa tipos OOF. Son resultados de desarrollo, sin validación externa; grados propios, no homologados al paper.',
             ha='center',fontsize=10,linespacing=1.5)
    fig.subplots_adjust(left=.06,right=.98,bottom=.30,top=.76,wspace=.23)
    fig.savefig(SALIDA/'matrices_con_sin_esperada.png',dpi=180);plt.close(fig)
    pd.DataFrame(resumen).to_csv(SALIDA/'resumen.csv',index=False)
    (SALIDA/'procedencia.json').write_text(json.dumps(dict(modelo_elegido=nombre,
        criterio_eleccion='máximo sin retroceso en desarrollo; empate por menos falsas alarmas observadas',
        fuente_sha256=hashlib.sha256((p/'comparacion.csv').read_bytes()).hexdigest(),
        recalculados=int(datos.tipo_correcto.eq(False).sum()),
        advertencia='Elección exploratoria sobre desarrollo. El oráculo no es un modelo ciego ni una prueba de generalización.'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print(nombre);print(pd.DataFrame(resumen).to_string(index=False))
