"""Verifica la exclusión conjunta de posiciones OR y compara los protocolos."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modelo_por_dominio import firma
from doble_banda_fijada import disjuntas
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_grupos_por_defecto'


def contar(d,col):
    f=d[d.paper.notna()];ok=f.real.eq(f.prediccion);delta=f[col].map(ORDEN)-f.paper.map(ORDEN)
    r=dict(total=len(f),mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()))
    assert sum(r[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto'))==len(f)
    return r


def main():
    p=json.loads((SALIDA/'protocolo.json').read_text(encoding='utf-8'))
    assert p['codigo']==firma(AQUI/'evaluar_grupos_por_defecto.py')
    for campo,nombre in [('control','control_ordenes_giro.py'),('pd','ensayar_pd_persistente.py'),('bandas','doble_banda_fijada.py')]:
        assert p[campo]==firma(AQUI/nombre)
    assert p['rasgos']==firma(AQUI/'resultados_globales_arbol/rasgos.npz')
    meta=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv').set_index('vista_uid',drop=False)
    grupo=meta.grupo.str.split('@').str[0]
    folds=json.loads((SALIDA/'pliegues.json').read_text(encoding='utf-8'))
    por_grupo={f['grupo_excluido']:f for f in folds}
    for f in folds:
        g=f['grupo_excluido'];esperado=set(meta.loc[grupo.ne(g),'vista_uid'])
        assert set(f['train'])==esperado and g not in f['grupos_train']
        if g.startswith('OR'):
            assert not meta.loc[f['train'],'grupo'].str.startswith(g).any()
    d=pd.read_csv(SALIDA/'comparacion_133.csv').set_index('archivo',drop=False)
    anterior=pd.read_csv(AQUI/'resultados_pd_doble/comparacion_133.csv').set_index('archivo',drop=False)
    assert len(d)==133 and d.index.equals(anterior.index) and d.sha256_senal.is_unique
    assert d.paper.notna().sum()==105 and d.real.eq('N').sum()==4
    for campo in ('real','paper','sha256_senal','grupo','suficiente'):
        assert d[campo].equals(anterior[campo])
    for r in d[d.suficiente].itertuples():
        f=por_grupo[r.grupo_auditoria]
        assert r.archivo not in set(meta.loc[f['train'],'archivo'])
    # Solo se agruparon de manera distinta las posiciones OR.
    otros=d.real.ne('OR')
    assert d.loc[otros,'prediccion'].equals(anterior.loc[otros,'prediccion'])
    assert d.loc[otros,'grado'].equals(anterior.loc[otros,'grado31'].rename('grado'))
    evidencia=json.loads((SALIDA/'evidencia.json').read_text(encoding='utf-8'))
    assert [q['archivo'] for q in evidencia]==d.index.tolist()
    for q in evidencia:
        r=d.loc[q['archivo']];assert q['tipo']==r.prediccion and q['grado']==r.grado
        ramas=q['ramas']
        if not ramas:
            assert r.prediccion in ('N','sin_diagnostico');continue
        assert 1<=len(ramas)<=2
        if len(ramas)==2:assert disjuntas(ramas[0]['seleccion']['nodo'],ramas[1]['seleccion']['nodo'],int(r.fs))
        for rama in ramas:
            s=rama['seleccion'];v=rama['verificacion']
            assert s['p_prueba']==.01/(31*5*6*2)
            assert v['grado']==min([s['grado'],v['evidencia']['grado']],key=ORDEN.get)
            if v['grado']=='D':
                for e in (s['evidencia'],v['evidencia']):
                    assert e['primera_identificada'] and len(e['identificadas'])>=2
                    assert not e['hay_pico_previo_mayor']
        assert r.grado==max([v['verificacion']['grado'] for v in ramas],key=ORDEN.get)
    resumen=json.loads((SALIDA/'resumen.json').read_text(encoding='utf-8'))
    assert all(resumen[k]==v for k,v in contar(d,'grado').items())
    config=json.loads((AQUI/'comparacion_con_sin_esperada_todos/configuracion.json').read_text(encoding='utf-8'))
    assert all(firma(AQUI/n)==h for n,h in config['hashes'].items())
    grados=['ND','PD','D'];columnas=grados+['Tipo incorrecto']
    fig,axs=plt.subplots(1,2,figsize=(13,6.3))
    for ax,(titulo,tabla,col) in zip(axs,[('Excluyendo condición y posición OR',anterior,'grado31'),('Excluyendo juntas todas las posiciones OR',d,'grado')]):
        f=tabla[tabla.paper.notna()];actual=f[col].where(f.real.eq(f.prediccion),'Tipo incorrecto')
        m=pd.crosstab(f.paper,actual).reindex(index=grados,columns=columnas,fill_value=0)
        assert int(m.to_numpy().sum())==105
        m.to_csv(SALIDA/('matriz_posiciones_separadas_105.csv' if tabla is anterior else 'matriz_posiciones_juntas_105.csv'))
        for i in range(3):
            for j in range(4):
                color='#e3e3e3' if j==3 else '#ccebd7' if j>i else '#dce8f4' if j==i else '#f5d2cc'
                ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor='white',linewidth=2))
                ax.text(j,i,str(int(m.iloc[i,j])),ha='center',va='center',fontsize=16)
        ax.set_xlim(-.5,3.5);ax.set_ylim(2.5,-.5)
        ax.set_xticks(range(4),['ND','PD','D','Tipo\nincorrecto']);ax.set_yticks(range(3),grados)
        ax.set_xlabel('Diagnóstico obtenido');ax.set_ylabel('Clasificación del paper')
        r=contar(tabla,col);ax.set_title(f'{titulo}\n{r["sin_retroceso"]}/105 sin retroceso',pad=14,fontsize=12)
        ax.text(.5,-.37,f'Mejoran {r["mejoran"]} · Iguales {r["iguales"]}\nEmpeoran {r["empeoran"]} · Tipo incorrecto {r["tipo_incorrecto"]}',transform=ax.transAxes,ha='center',linespacing=1.5)
    fig.suptitle('Sensibilidad a la agrupación · reglas y banco31 sin reajustar',fontsize=15)
    fig.text(.5,.035,'Mismos 133 archivos; matrices: 105 con referencia. Evaluación interna sobre datos ya inspeccionados; sin validación externa.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.79,bottom=.29,left=.06,right=.98,wspace=.24)
    fig.savefig(SALIDA/'matrices_agrupacion_OR.png',dpi=170);plt.close(fig)
    r=contar(d,'grado')
    informe=['# Auditoría agrupando las posiciones OR','',
        f'Resultado con todas las posiciones del mismo tipo/tamaño excluidas juntas: {r}.',
        f'Resultado previo, distinguiendo posición: {contar(anterior,"grado31")}.',
        'Se reentrenaron tanto el detector como el tipo sin el grupo completo; únicamente se usaron los 52 casos de desarrollo y las dos señales normales de entrenamiento. Las reglas y el banco31 se fijaron antes de esta auditoría.',
        'Se verificó que los casos ajenos a OR conservan exactamente su tipo y grado. La comparación mantiene los mismos 133 archivos, los 105 con referencia y todas las señales insuficientes.', '',
        '## Interpretación metodológica','',
        'Agrupar por tipo y tamaño es una precaución conservadora: no demuestra que cada grupo equivalga a un rodamiento físico distinto.',
        'CWRU documenta experimentos con distintas posiciones OR y explica que la posición afecta la vibración. No publica aquí identificadores suficientes para demostrar independencia entre todos los rodamientos.',
        'Fuente: [CWRU, aparato y procedimientos](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures) y [tabla de datos DE](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data).',
        'La misma página indica SKF para defectos 7/14/21 mil y NTN equivalentes para 28/40 mil. El catálogo configurado usa las razones DE/FE publicadas para SKF; faltan coeficientes NTN específicos para confirmar esa equivalencia, y no se deben estimar conociendo el defecto verdadero.',
        'Las reglas también se desarrollaron tras inspeccionar el dataset. Este control no elimina el sesgo por selección reiterada ni sustituye validación externa.',
        'Los p siguen siendo nominales; el desempeño sobre cuatro normales no acredita una tasa poblacional de falsas alarmas.', '',
        'La meta sigue siendo al menos 90/105 sin retroceso, con muchas más mejoras que empeoramientos. No se cambia el denominador ni se convierten errores de tipo en ND usando la etiqueta real.',
        'Verificar: `python 05_Diagnostico_Ciego/verificar_grupos_por_defecto.py`.','']
    (SALIDA/'INFORME.md').write_text('\n'.join(informe),encoding='utf-8')
    (SALIDA/'verificacion.json').write_text(json.dumps(dict(casos=133,referencia=105,
        OR_posiciones_excluidas_juntas=True,no_OR_sin_cambios=True,modelos_historicos_intactos=True,
        protocolo_sha256=firma(SALIDA/'protocolo.json'),resumen=r),indent=2),encoding='utf-8')
    print('OK: posiciones OR fuera juntas, 133 casos, ramas D y modelos históricos verificados.')
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()
