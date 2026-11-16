"""Audita las correcciones contra las señales y genera comparación completa."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modelo_ciego import F
from modelo_por_dominio import firma
from bandas_estables import seleccionar, verificar
from criterio_usuario import ORDEN
from correccion_d_persistente import elegir, comprobar
from predictor_d_persistente import predecir

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_correccion_d_persistente'


def contar(d):
    f = d[d.paper.notna()]; delta = f.grado.map(ORDEN)-f.paper.map(ORDEN)
    ok = f.real.eq(f.prediccion)
    return dict(total=len(f),mejoran=int((ok & delta.gt(0)).sum()),
        iguales=int((ok & delta.eq(0)).sum()),empeoran=int((ok & delta.lt(0)).sum()),
        tipo_incorrecto=int((~ok).sum()),sin_retroceso=int((ok & delta.ge(0)).sum()))


def main():
    comprobar()
    p = json.loads((SALIDA/'protocolo.json').read_text(encoding='utf-8'))
    assert p['codigo'] == firma(AQUI/'correccion_d_persistente.py')
    assert p['bandas'] == firma(AQUI/'bandas_estables.py')
    assert p['criterio'] == firma(AQUI/'criterio_usuario.py')
    fuente = AQUI/'resultados_total_condicion_excluida'
    assert p['base'] == firma(fuente/'comparacion_133.csv')
    base = pd.read_csv(fuente/'comparacion_133.csv').set_index('archivo',drop=False)
    d = pd.read_csv(SALIDA/'comparacion_133.csv').set_index('archivo',drop=False)
    assert base.index.equals(d.index) and len(d) == 133 and d.sha256_senal.is_unique
    assert d.paper.notna().sum() == 105 and d.real.eq('N').sum() == 4
    for campo in ('real','paper','grupo','sha256_senal'):
        assert base[campo].equals(d[campo])
    pruebas = json.loads((SALIDA/'evidencia.json').read_text(encoding='utf-8'))
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU')).set_index('archivo')
    recalculadas = 0; futuro_probado = False
    for fila in pruebas:
        archivo = fila['archivo']; b = base.loc[archivo]; nuevo = d.loc[archivo]
        if not b.suficiente or b.prediccion == 'N':
            assert b.prediccion == nuevo.prediccion and b.grado == nuevo.grado
            continue
        tipo,_ = elegir(b.prediccion,fila['alternativas_seleccion'])
        assert tipo == nuevo.prediccion
        assert fila['seleccion'] == fila['alternativas_seleccion'][tipo]
        assert nuevo.grado == fila['verificacion']['grado']
        if nuevo.corregido:
            r = idx.loc[archivo]
            mat = loadmat(AQUI.parent/'data2/CWRU'/archivo)
            senal = np.ravel(mat[F.clave_canal(mat,r.sensor_falla,int(r.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == b.sha256_senal
            rpm = F.rpm_registro(mat,int(r.registro),nominal=float(r.rpm_nominal))
            c = {k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
            mitad = len(senal)//2
            # Recalcular TODAS las alternativas sin consultar tipo real/paper.
            alternativas = {t:seleccionar(senal[:mitad],int(r.fs),c,t) for t in ('IR','OR','B')}
            assert alternativas == fila['alternativas_seleccion']
            assert elegir(b.prediccion,alternativas)[0] == tipo
            # Cambiar por completo el futuro conserva la decisión.
            alterada = senal.copy()
            alterada[mitad:] = np.random.default_rng(184).normal(size=len(senal)-mitad)
            assert np.array_equal(senal[:mitad],alterada[:mitad])
            if not futuro_probado:
                original = predecir(senal,int(r.fs),c)
                cambiado = predecir(alterada,int(r.fs),c)
                for campo in ('tipo','tipo_base','seleccion','alternativas_seleccion'):
                    assert original[campo] == cambiado[campo]
                assert original['verificacion'] != cambiado['verificacion']
                futuro_probado = True
            v = verificar(senal[mitad:],int(r.fs),c,alternativas[tipo])
            assert v == fila['verificacion']
            recalculadas += 1
            print('Correccion recalculada desde MAT:',archivo,flush=True)
    assert recalculadas == int(d.corregido.sum()) and futuro_probado
    resumen = json.loads((SALIDA/'resumen.json').read_text(encoding='utf-8'))
    assert all(resumen[k] == v for k,v in contar(d).items())
    # La etiqueta solo se utiliza ahora para medir los cambios ya fijados.
    delta_base = base.grado.map(ORDEN)-base.paper.map(ORDEN)
    delta_nuevo = d.grado.map(ORDEN)-d.paper.map(ORDEN)
    nr_base = base.real.eq(base.prediccion) & delta_base.ge(0)
    nr_nuevo = d.real.eq(d.prediccion) & delta_nuevo.ge(0)
    f = d[d.paper.notna()].copy()
    f['sin_retroceso_base'] = nr_base.loc[f.index]
    f['sin_retroceso_nuevo'] = nr_nuevo.loc[f.index]
    f['recupera'] = ~f.sin_retroceso_base & f.sin_retroceso_nuevo
    f['pierde'] = f.sin_retroceso_base & ~f.sin_retroceso_nuevo
    f.to_csv(SALIDA/'cambios_pareados_105.csv',index=False)
    recupera = int(f.recupera.sum()); pierde = int(f.pierde.sum())
    assert int(nr_nuevo.sum()-nr_base.sum()) == recupera-pierde
    grados = ['ND','PD','D']; columnas = grados+['Tipo incorrecto']
    fig,axs = plt.subplots(1,2,figsize=(13,6.2))
    for ax,(titulo,tabla) in zip(axs,[('Base: condicion excluida',base),('Correccion D persistente',d)]):
        ref = tabla[tabla.paper.notna()]
        salida = ref.grado.where(ref.real.eq(ref.prediccion),'Tipo incorrecto')
        matriz = pd.crosstab(ref.paper,salida).reindex(index=grados,columns=columnas,fill_value=0)
        assert int(matriz.to_numpy().sum()) == 105
        matriz.to_csv(SALIDA/('matriz_base_105.csv' if tabla is base else 'matriz_correccion_105.csv'))
        for i in range(3):
            for j in range(4):
                color = '#e3e3e3' if j == 3 else '#ccebd7' if j > i else '#dce8f4' if j == i else '#f5d2cc'
                ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor='white',linewidth=2))
                ax.text(j,i,str(int(matriz.iloc[i,j])),ha='center',va='center',fontsize=16)
        ax.set_xlim(-.5,3.5); ax.set_ylim(2.5,-.5)
        ax.set_xticks(range(4),['ND','PD','D','Tipo\nincorrecto']); ax.set_yticks(range(3),grados)
        ax.set_xlabel('Diagnóstico obtenido'); ax.set_ylabel('Clasificación del paper')
        r = contar(tabla)
        ax.set_title(f'{titulo}\n{r["sin_retroceso"]}/105 sin retroceso',pad=14)
        ax.text(.5,-.37,f'Mejoran {r["mejoran"]} · Iguales {r["iguales"]}\nEmpeoran {r["empeoran"]} · Tipo incorrecto {r["tipo_incorrecto"]}',
            transform=ax.transAxes,ha='center',linespacing=1.5)
    fig.suptitle('Mismo juez y exclusión por condición · sin frecuencia esperada',fontsize=15)
    fig.text(.5,.035,'Ensayo interno sobre datos ya inspeccionados; sin validación externa. Se conservan 133 archivos; matrices: 105 con referencia.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.79,bottom=.29,left=.06,right=.98,wspace=.24)
    fig.savefig(SALIDA/'matrices_base_vs_correccion.png',dpi=170); plt.close(fig)
    informe = ['# Corrección por una única familia D persistente', '',
        'Se comparan IR/OR/B usando exclusivamente la primera mitad. Se cambia el tipo solo si una única familia cumple D en los tres subsegmentos de selección. La banda y la familia se fijan antes de verificar la segunda mitad.',
        'BSF y 2BSF son dos representaciones de B; no se elige una familia conociendo el defecto verdadero.', '',
        f'Base: {contar(base)}',f'Corrección: {contar(d)}',
        f'Recupera {recupera} casos que antes retrocedían; pierde {pierde} casos que antes no retrocedían. Neto: {recupera-pierde:+d}.',
        f'Se recalcularon desde MAT todas las alternativas y la verificación de los {recalculadas} registros corregidos, sin usar caché para esas comprobaciones.', '',
        'El predictor público solo recibe señal, fs y catálogo completo. Se ejecutó dos veces sobre una señal real, reemplazando por ruido toda su segunda mitad: tipo, selección y alternativas se conservaron exactamente, mientras la verificación cambió.',
        'Ese predictor usa los pesos congelados para señales nuevas; no sustituye las predicciones fuera de condición usadas para puntuar los 133 registros.', '',
        '## Límites', '',
        'El objetivo de al menos 90/105 sin retroceso y muchas más mejoras que empeoramientos no está alcanzado.',
        'Esta regla fue propuesta después de inspeccionar errores del dataset: la comparación es exploratoria y no equivale a una validación externa o a una prueba ciega de selección de modelos.',
        'Persistencia temporal y grado D no garantizan acertar el tipo. Las nuevas equivocaciones muestran que otra familia puede presentar picos compatibles y persistentes.',
        'Se preservan las cargas, sensores y muestreos de cada condición fuera del entrenamiento. La posición OR sigue formando parte del grupo; la independencia por rodamiento físico no está demostrada.',
        'El modelo histórico no fue reemplazado. El ranker compartido de armónicas y el clasificador basado solo en evidencia de selección empeoraron y no se adoptan.', '',
        'Reproducir: `python 05_Diagnostico_Ciego/verificar_correccion_d_persistente.py`.', '']
    (SALIDA/'INFORME.md').write_text('\n'.join(informe),encoding='utf-8')
    (SALIDA/'verificacion.json').write_text(json.dumps(dict(casos=133,referencia=105,
        correcciones_recalculadas=recalculadas,recupera=recupera,pierde=pierde,
        neto=recupera-pierde,independencia_futuro_probada=futuro_probado,
        predictor_sha256=firma(AQUI/'predictor_d_persistente.py'),
        protocolo_sha256=firma(SALIDA/'protocolo.json')),indent=2),encoding='utf-8')
    print('OK: decisiones sin futuro, correcciones recalculadas desde MAT y recuentos pareados verificados.')
    print(json.dumps(dict(recupera=recupera,pierde=pierde,resumen=resumen)))


if __name__ == '__main__':
    main()
