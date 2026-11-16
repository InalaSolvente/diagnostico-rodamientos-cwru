"""Auditoría reproducible de tipos OOF, dos bandas y reglas temporales."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modelo_ciego import F
from modelo_por_dominio import crear,firma
from ensayar_tipo_por_sensor import predecir_tipo
from control_ordenes_giro import elegir as elegir_d
from ensayar_pd_persistente import elegir as elegir_pd
from doble_banda_fijada import seleccionar_dos,disjuntas
from predictor_global_doble import predecir,MODELO
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_pd_doble'


def contar(d,col):
    f=d[d.paper.notna()]; ok=f.real.eq(f.prediccion); delta=f[col].map(ORDEN)-f.paper.map(ORDEN)
    r=dict(total=len(f),mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()))
    assert sum(r[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == len(f)
    return r


def main():
    d=pd.read_csv(SALIDA/'comparacion_133.csv').set_index('archivo',drop=False)
    base_dir=AQUI/'resultados_total_condicion_excluida'
    base=pd.read_csv(base_dir/'comparacion_133.csv').set_index('archivo',drop=False)
    assert len(d)==133 and d.index.equals(base.index) and d.sha256_senal.is_unique
    assert d.paper.notna().sum()==105 and d.real.eq('N').sum()==4
    for campo in ('real','paper','sha256_senal','grupo','suficiente'):
        assert d[campo].equals(base[campo])
    config=json.loads((AQUI/'comparacion_con_sin_esperada_todos/configuracion.json').read_text(encoding='utf-8'))
    assert all(firma(AQUI/n)==h for n,h in config['hashes'].items()),'Se alteró el modelo histórico'
    protocolos=[('resultados_control_ordenes_giro','control_ordenes_giro.py'),
        ('resultados_doble_banda_fijada','doble_banda_fijada.py'),
        ('resultados_globales_arbol','ensayar_globales_arbol.py'),
        ('resultados_globales_bancos','evaluar_globales_bancos.py'),
        ('resultados_pd_persistente','ensayar_pd_persistente.py'),
        ('resultados_pd_doble','evaluar_pd_doble.py')]
    for carpeta,script in protocolos:
        p=json.loads((AQUI/carpeta/'protocolo.json').read_text(encoding='utf-8'))
        assert p['codigo']==firma(AQUI/script)
        for campo,nombre in [('bandas','bandas_estables.py'),('criterio','criterio_usuario.py'),
                             ('banco31','doble_banda_fijada.py'),('banco63','banco_63.py')]:
            if campo in p:assert p[campo]==firma(AQUI/nombre)
    f=AQUI/'resultados_globales_arbol'
    globales=pd.read_csv(f/'tipos_133.csv').set_index('archivo',drop=False)
    with np.load(f/'rasgos.npz',allow_pickle=False) as z:x=z['x']
    assert x.shape==(133,4,2021)
    meta=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    meta=meta[meta.real.ne('N')].copy()
    posiciones=d.reset_index(drop=True).reset_index().set_index('archivo')['index'].reindex(meta.archivo).to_numpy()
    xd=x[posiciones]; y=meta.real.to_numpy(); grupos=meta.grupo.to_numpy()
    folds={q['grupo_excluido']:q for q in json.loads((f/'pliegues.json').read_text(encoding='utf-8'))}
    origen=json.loads((AQUI/'resultados_correccion_d_persistente/evidencia.json').read_text(encoding='utf-8'))
    alternativas={q['archivo']:q['alternativas_seleccion'] for q in origen}
    # Reajusta y reproduce TODOS los tipos OOF, no solo una muestra favorable.
    with threadpool_limits(limits=1):
        for grupo,fold in folds.items():
            tr=grupos!=grupo
            assert set(fold['train'])==set(meta.loc[tr,'archivo'])
            assert grupo not in set(grupos[tr])
            clf=crear().fit(xd[tr].reshape(-1,2021),np.repeat(y[tr],4))
            casos=base[base.grupo.eq(grupo) & base.suficiente & base.prediccion.ne('N')]
            for archivo,r in casos.iterrows():
                i=int(np.flatnonzero(d.index==archivo)[0])
                tb=predecir_tipo(clf,x[i])
                assert tb==globales.loc[archivo,'tipo_sin_control']
                td,_=elegir_d(tb,alternativas[archivo],F.MULTIPLICADORES[r.sensor])
                assert td==globales.loc[archivo,'prediccion']
                assert elegir_pd(td,alternativas[archivo])==d.loc[archivo,'prediccion']
            print('OOF y reglas reproducidos:',grupo,flush=True)
    evidencia=json.loads((SALIDA/'evidencia.json').read_text(encoding='utf-8'))
    assert [q['archivo'] for q in evidencia]==d.index.tolist()
    for q in evidencia:
        r=d.loc[q['archivo']]
        assert q['tipo']==r.prediccion
        for banco,col in [(31,'grado31'),(63,'grado63')]:
            v=q[f'banco{banco}']; ramas=v['ramas']
            assert v['grado']==r[col]
            if not ramas:
                assert not r.suficiente or r.prediccion=='N'
                continue
            assert 1<=len(ramas)<=2
            if len(ramas)==2:assert disjuntas(ramas[0]['seleccion']['nodo'],ramas[1]['seleccion']['nodo'],int(r.fs))
            for rama in ramas:
                s=rama['seleccion']; a=rama['verificacion']
                assert s['p_prueba']==.01/(banco*5*6*2)
                assert a['grado']==min([s['grado'],a['evidencia']['grado']],key=ORDEN.get)
                if a['grado']=='D':
                    for e in (s['evidencia'],a['evidencia']):
                        assert e['primera_identificada'] and len(e['identificadas'])>=2
                        assert not e['hay_pico_previo_mayor']
            assert v['grado']==max([a['verificacion']['grado'] for a in ramas],key=ORDEN.get)
    resumen=pd.read_csv(SALIDA/'resumen.csv').set_index('banco')
    for banco,col in [(31,'grado31'),(63,'grado63')]:
        assert all(int(resumen.loc[banco,k])==v for k,v in contar(d,col).items())
    # Comprueba las selecciones PD que cambian el tipo desde la señal cruda.
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    cambios=d[d.prediccion.ne(globales.prediccion)]
    futuro_probado=False
    for archivo,r in cambios.iterrows():
        fila=idx.loc[archivo]; mat=loadmat(AQUI.parent/'data2/CWRU'/archivo)
        senal=np.ravel(mat[F.clave_canal(mat,r.sensor,int(fila.registro))])
        rpm=F.rpm_registro(mat,int(fila.registro),nominal=float(fila.rpm_nominal))
        c={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor].items()}
        mitad=len(senal)//2
        a={t:seleccionar_dos(senal[:mitad],int(r.fs),c,t)[0] for t in ('IR','OR','B')}
        assert a==alternativas[archivo]
        assert elegir_pd(globales.loc[archivo,'prediccion'],a)==r.prediccion
        if not futuro_probado:
            antes=predecir(senal,int(r.fs),c)
            alterada=senal.copy();alterada[mitad:]=np.random.default_rng(814).normal(size=len(senal)-mitad)
            despues=predecir(alterada,int(r.fs),c)
            assert antes['ramas'] and despues['ramas']
            for campo in ('tipo','tipo_base','tipo_control_d','selecciones','alternativas_seleccion'):
                assert antes[campo]==despues[campo]
            assert antes['ramas']!=despues['ramas']
            futuro_probado=True
        print('PD recalculado desde MAT:',archivo,flush=True)
    assert futuro_probado
    grados=['ND','PD','D']; columnas=grados+['Tipo incorrecto']
    fig,axs=plt.subplots(1,2,figsize=(13,6.3))
    for ax,(titulo,tabla,col) in zip(axs,[('Base: condición excluida',base,'grado'),('Globales + doble banda + PD persistente',d,'grado31')]):
        ref=tabla[tabla.paper.notna()]; actual=ref[col].where(ref.real.eq(ref.prediccion),'Tipo incorrecto')
        m=pd.crosstab(ref.paper,actual).reindex(index=grados,columns=columnas,fill_value=0)
        assert int(m.to_numpy().sum())==105
        m.to_csv(SALIDA/('matriz_base_105.csv' if tabla is base else 'matriz_actual_105.csv'))
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
    fig.suptitle('Sin frecuencia esperada · mismo juez y exclusión por condición',fontsize=15)
    fig.text(.5,.035,'Ensayo exploratorio: reglas desarrolladas con datos ya inspeccionados. Sin validación externa. Matrices: 105; cobertura total: 133.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.79,bottom=.29,left=.06,right=.98,wspace=.24)
    fig.savefig(SALIDA/'matrices_base_vs_actual.png',dpi=170);plt.close(fig)
    r=contar(d,'grado31')
    informe=['# Globales, control de giro, doble banda y PD persistente','',
        f'Resultado: {r}. Cuatro normales correctos. Cobertura: 133 archivos; referencia del paper: 105.',
        'Se elige el banco de 31 por simplicidad: el de 63 arroja los mismos recuentos y exige más cómputo. No se elige un banco distinto por registro.',
        'Los 52 casos de desarrollo siguen separados por condición en evaluación; se reprodujeron todos los tipos y se recalcularon desde MAT las correcciones PD.',
        'Cambiar completamente la segunda mitad conservó tipo, alternativas y ambas bandas del predictor público; solo cambió la verificación. No acepta etiquetas, nombres de archivos ni frecuencia de falla esperada.',
        'Los grados D fueron comprobados por rama: primera armónica significativa, al menos dos armónicas y ningún pico previo mayor. Nunca se suman armónicas de bandas distintas. El umbral conserva la corrección por el banco completo.', '',
        '## Límites metodológicos','',
        'La meta de al menos 90/105 sin retroceso todavía no se alcanza.',
        'Las reglas fueron desarrolladas después de inspeccionar resultados del dataset. Estos recuentos son exploratorios: no equivalen a una selección de modelos ciega ni a validación externa.',
        'La exclusión por condición todavía distingue posiciones OR de una misma dimensión de defecto. La independencia entre rodamientos físicos no está demostrada. Debe auditarse con una exclusión más conservadora antes de afirmar generalización.',
        'Los p son nominales y no constituyen una garantía empírica de falsas alarmas; cuatro normales y dos archivos normales de desarrollo son pocos.',
        'El modelo original se conserva. Los pesos nuevos se ajustan solo con los 52 registros de desarrollo y no se usan para fabricar las cifras OOF.', '',
        'Inferencia: `from predictor_global_doble import predecir`; `predecir(senal, fs, catalogo_completo)`.',
        'Exportar pesos: `python 05_Diagnostico_Ciego/predictor_global_doble.py --exportar`.',
        'Verificar: `python 05_Diagnostico_Ciego/verificar_global_doble.py`.','']
    (SALIDA/'INFORME.md').write_text('\n'.join(informe),encoding='utf-8')
    (SALIDA/'verificacion.json').write_text(json.dumps(dict(casos=133,referencia=105,
        OOF_completo_reproducido=True,correcciones_PD_recalculadas=len(cambios),futuro_independiente=True,
        modelo_sha256=firma(MODELO),predictor_sha256=firma(AQUI/'predictor_global_doble.py'),
        resumen=r),indent=2),encoding='utf-8')
    print('OK: OOF completo, reglas, ramas D, cache, futuro independiente y recuentos verificados.')
    print(json.dumps(r),flush=True)


if __name__=='__main__':
    main()
