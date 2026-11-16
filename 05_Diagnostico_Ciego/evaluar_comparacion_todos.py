"""Con/sin frecuencia esperada en los 133 registros, con pesos congelados.

Conserva OOF de desarrollo; evalúa el resto con el modelo elegido previamente.
Los normales no reciben una frecuencia/familia de falla oracular inventada.
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
from modelo_jerarquico import predecir
from modelo_por_dominio import calidad,clave_cache
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
DATOS=AQUI.parent/'data2/CWRU'
SALIDA=AQUI/'comparacion_con_sin_esperada_todos'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def figura(d):
    f=d[d.paper.notna()].copy();assert len(f)==105
    grados=['ND','PD','D'];columnas=grados+['Tipo incorrecto']
    colores={'mejora':'#ccebd7','igual':'#dce8f4','baja':'#f5d2cc','error':'#e1e1e1'}
    fig,axs=plt.subplots(1,2,figsize=(14,6.8));resumen=[]
    for ax,(titulo,col,oracle) in zip(axs,[('Con frecuencia esperada · ORÁCULO','grado_con_esperada',True),
                                        ('Sin frecuencia esperada','grado_sin_esperada',False)]):
        actual=f[col] if oracle else f[col].where(f.tipo_correcto.eq(True),'Tipo incorrecto')
        t=pd.crosstab(f.paper,actual).reindex(index=grados,columns=columnas,fill_value=0)
        assert int(t.to_numpy().sum())==105
        mejoran=sum(int(t.iloc[i,j]) for i in range(3) for j in range(3) if j>i)
        iguales=sum(int(t.iloc[i,i]) for i in range(3))
        bajan=sum(int(t.iloc[i,j]) for i in range(3) for j in range(3) if j<i)
        errores=int(t.iloc[:,3].sum());nr=mejoran+iguales
        assert mejoran+iguales+bajan+errores==105
        resumen.append(dict(escenario=titulo,total=105,mejoran=mejoran,iguales=iguales,
            empeoran=bajan,tipo_incorrecto=errores,sin_retroceso=nr,porcentaje=100*nr/105))
        t.to_csv(SALIDA/('matriz_con_esperada_105.csv' if oracle else 'matriz_sin_esperada_105.csv'))
        for i in range(3):
            for j in range(4):
                estado='error' if j==3 else 'mejora' if j>i else 'baja' if j<i else 'igual'
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=colores[estado],edgecolor='white',linewidth=2))
                n=int(t.iloc[i,j]);total=int(t.iloc[i].sum())
                ax.text(j,i,f'{n}\n({100*n/total:.1f} %)',ha='center',va='center',fontsize=11)
        ax.set_xlim(-.5,3.5);ax.set_ylim(2.5,-.5)
        ax.set_xticks(range(4),['ND','PD','D','Tipo\nincorrecto']);ax.set_yticks(range(3),grados)
        ax.set_xlabel('Diagnóstico obtenido');ax.set_ylabel('Clasificación del paper');ax.tick_params(length=0)
        ax.set_title(f'{titulo}\n{nr}/105 sin retroceso ({100*nr/105:.2f} %)',fontsize=12,pad=13)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.text(.5,-.28,f'Mejoran {mejoran} · Iguales {iguales}\nEmpeoran {bajan} · Tipo incorrecto {errores}',
                transform=ax.transAxes,ha='center',fontsize=10,linespacing=1.5)
    fig.suptitle('Con y sin frecuencia esperada · todos los casos con referencia del paper',fontsize=15,y=.97)
    fig.text(.5,.89,'133 registros cubiertos: 105 en estas matrices; 24 fallas de 48 kHz y 4 normales se informan aparte.',ha='center',fontsize=11)
    fig.text(.5,.035,'Mismo criterio y comprobación temporal. Oráculo: familia real conocida antes de elegir la banda.\n'
        'Sin esperada: OOF en desarrollo y modelo congelado en el resto; protocolo mixto, sin validación externa.\n'
        'Grados propios, no homologados al paper. No se reentrenó ni se eligió otro modelo mirando estos resultados.',
        ha='center',fontsize=9,linespacing=1.5)
    fig.subplots_adjust(left=.06,right=.98,bottom=.31,top=.76,wspace=.23)
    fig.savefig(SALIDA/'matrices_con_sin_esperada_105.png',dpi=180);plt.close(fig)
    pd.DataFrame(resumen).to_csv(SALIDA/'resumen_matrices_105.csv',index=False)
    return resumen


if __name__=='__main__':
    SALIDA.mkdir(exist_ok=True)
    padre=AQUI/'resultados_jerarquico'
    eleccion=json.loads((AQUI/'comparacion_con_sin_esperada/procedencia.json').read_text(encoding='utf-8'))
    assert eleccion['modelo_elegido']=='resultados_jerarquico'
    congelados=[padre/'modelo_tipo.npz',AQUI/'resultados_por_dominio/modelo_DE.npz',
        AQUI/'resultados_por_dominio/modelo_FE.npz',AQUI/'bandas_estables.py',AQUI/'criterio_usuario.py',
        AQUI/'ventanas_verificadas.py',AQUI/'modelo_espectral.py']
    firmas={str(p.relative_to(AQUI)):sha(p) for p in congelados}
    config=dict(modelo='resultados_jerarquico, elegido antes de esta ejecución sobre desarrollo',
        hashes=firmas,reentrenamiento=False,seleccion_por_resultado_total=False,
        protocolos='OOF por condición en desarrollo; modelo congelado en archivos restantes',
        no_externo=True,normales_oraculo='no aplica: no existe frecuencia de falla esperada en un normal',
        cortas='se intenta cada registro; menos de 5 s se informa sin diagnóstico, sin excluirlo de los recuentos',
        referencia_paper=105,total_archivos=133)
    rp=SALIDA/'configuracion.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==config,'Modelo cambió: no sobrescribir comparación'
    else:rp.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    idx=F.leer_indice(str(DATOS));assert len(idx)==133 and idx.archivo.is_unique
    meta=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv').set_index('vista_uid',drop=False)
    cp=json.loads((padre/'protocolo.json').read_text(encoding='utf-8'))
    assert cp['bandas']==firmas['bandas_estables.py'] and cp['criterio']==firmas['criterio_usuario.py']
    cache={}
    for q in json.loads((padre/'predicciones.json').read_text(encoding='utf-8')):
        r=meta.loc[q['vista_uid']];cache[clave_cache(r)]={k:q[k] for k in ('tipo','grado','seleccion','verificacion')}
    oraculos={r['archivo']:r['oraculo'] for r in json.loads((AQUI/'comparacion_con_sin_esperada/evidencia_oraculo.json').read_text(encoding='utf-8'))}
    referencia=pd.read_csv(AQUI/'resultados_criterio_usuario_total/comparacion.csv').set_index('archivo').paper
    filas=[];evidencia=[]
    for i,r in enumerate(idx.itertuples()):
        mat=loadmat(DATOS/r.archivo);clave=F.clave_canal(mat,r.sensor_falla,int(r.registro))
        x=np.ravel(mat[clave]);rpm=F.rpm_registro(mat,int(r.registro),nominal=float(r.rpm_nominal))
        c={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
        huella=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        key=(r.archivo,clave,huella,int(r.fs),json.dumps(c,sort_keys=True))
        suficiente=len(x)//2>=2.5*r.fs
        if not suficiente:
            sin=dict(tipo='sin_diagnostico',grado='ND',motivo='menos_de_5_segundos')
            origen='sin_diagnostico_por_duracion'
        elif key in cache:
            sin=cache[key];origen='desarrollo_oof_reutilizado'
        else:
            sin=predecir(x,int(r.fs),c,modelo=padre/'modelo_tipo.npz');origen='modelo_congelado'
        # Se entrega la familia verdadera SOLO al escenario explícitamente oracular.
        if r.zona=='N':
            con=None
        elif not suficiente:
            con=dict(tipo=r.zona,grado='ND',motivo='menos_de_5_segundos')
        elif r.archivo in oraculos and key in cache:
            con=oraculos[r.archivo]
        elif sin['tipo']==r.zona:
            con=sin
        else:
            con=calidad(x,int(r.fs),c,r.zona)
        filas.append(dict(archivo=r.archivo,registro=int(r.registro),fs=int(r.fs),sensor=r.sensor_falla,
            real=r.zona,segundos=len(x)/r.fs,suficiente=suficiente,origen=origen,
            tipo_sin_esperada=sin['tipo'],grado_sin_esperada=sin['grado'],tipo_correcto=sin['tipo']==r.zona,
            oraculo_aplica=r.zona!='N',grado_con_esperada=con['grado'] if con else None,
            paper=referencia.get(r.archivo),sha256_senal=huella))
        evidencia.append(dict(archivo=r.archivo,sin_esperada=sin,con_esperada=con))
        pd.DataFrame(filas).to_csv(SALIDA/'comparacion_133_parcial.csv',index=False)
        (SALIDA/'evidencia_parcial.json').write_text(json.dumps(evidencia,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{i+1}/133 {r.archivo}: sin={sin["tipo"]}/{sin["grado"]}; con={con["grado"] if con else "no_aplica"}; {origen}',flush=True)
    d=pd.DataFrame(filas)
    assert len(d)==133 and d.archivo.is_unique and d.sha256_senal.is_unique
    assert d.paper.notna().sum()==105 and d.real.eq('N').sum()==4
    assert d.loc[d.real.eq('N'),'grado_con_esperada'].isna().all()
    assert all(sha(AQUI/p)==h for p,h in firmas.items())
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencia,ensure_ascii=False,indent=2),encoding='utf-8')
    resumen=figura(d)
    extra=d[d.fs.eq(48000)&d.real.ne('N')];normales=d[d.real.eq('N')]
    lineas=['# Comparación completa con/sin frecuencia esperada','',
        '133 registros cubiertos, sin reajustar modelos. Se conservan predicciones OOF de desarrollo.',
        'Las matrices de cambios solo incluyen los 105 registros de 12 kHz con referencia del paper.',
        'El total mezcla protocolos y condiciones compartidas; no es validación externa ni un ensayo de un único ajuste independiente.',
        'Los normales no reciben un oráculo de salud: no tienen una frecuencia de falla esperada.',
        'Las señales insuficientes se conservan como sin diagnóstico y no se eliminan del denominador.',
        '',f'48 kHz con falla: {int(extra.tipo_correcto.sum())}/{len(extra)} tipos correctos sin frecuencia esperada; {int((~extra.suficiente).sum())} sin duración suficiente.',
        f'Normales: {int(normales.tipo_correcto.sum())}/{len(normales)} correctamente identificados sin frecuencia esperada.',
        '48 kHz se evaluó con el mismo método jerárquico normalizado; no con el modelo histórico separado de 48 kHz.',
        '',json.dumps(resumen,ensure_ascii=False,indent=2)]
    (SALIDA/'resumen.md').write_text('\n'.join(lineas)+'\n',encoding='utf-8')
    print(pd.DataFrame(resumen).to_string(index=False));print('\n'.join(lineas[8:11]))
