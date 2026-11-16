"""Comprueba procedencia, exclusiones y recuentos; publica matriz e informe."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_total_condicion_excluida'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def contar(d):
    f = d[d.paper.notna()]; ok = f.real.eq(f.prediccion)
    delta = f.grado.map(ORDEN)-f.paper.map(ORDEN)
    r = dict(total=len(f), mejoran=int((ok & delta.gt(0)).sum()),
        iguales=int((ok & delta.eq(0)).sum()), empeoran=int((ok & delta.lt(0)).sum()),
        tipo_incorrecto=int((~ok).sum()), sin_retroceso=int((ok & delta.ge(0)).sum()))
    assert sum(r[k] for k in ('mejoran', 'iguales', 'empeoran', 'tipo_incorrecto')) == len(f)
    return r


def main():
    meta = pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    m = meta.set_index('vista_uid')
    p = json.loads((SALIDA/'protocolo.json').read_text(encoding='utf-8'))
    assert sha(AQUI/'evaluar_total_condicion_excluida.py') == p['codigo']
    for nombre, h in p['hashes'].items():
        assert sha(AQUI/nombre) == h
    assert sha(AQUI/'resultados_dominios_desarrollo/rasgos.npz') == p['fuente']
    d = pd.read_csv(SALIDA/'comparacion_133.csv')
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    assert d.paper.notna().sum() == 105 and d.real.eq('N').sum() == 4
    assert d.loc[d.paper.notna(), 'paper'].value_counts().to_dict() == {'D':70, 'ND':18, 'PD':17}
    folds = json.loads((SALIDA/'pliegues.json').read_text(encoding='utf-8'))
    por_grupo = {f['grupo_excluido']:f for f in folds}
    for f in folds:
        train = m.loc[f['train']]
        assert train.grupo.ne(f['grupo_excluido']).all()
        assert set(f['train']) == set(meta.loc[meta.grupo.ne(f['grupo_excluido']), 'vista_uid'])
    for r in d[d.suficiente].itertuples():
        assert r.grupo in por_grupo
        assert r.archivo not in set(m.loc[por_grupo[r.grupo]['train'], 'archivo'])
    anterior = AQUI/'comparacion_con_sin_esperada_todos'
    config = json.loads((anterior/'configuracion.json').read_text(encoding='utf-8'))
    assert all(sha(AQUI/n) == h for n,h in config['hashes'].items()), 'Se altero el modelo historico'
    resumen = json.loads((SALIDA/'resumen.json').read_text(encoding='utf-8'))
    assert all(resumen[k] == v for k,v in contar(d).items())
    base52 = pd.read_csv(AQUI/'resultados_jerarquico/comparacion.csv')
    j = base52[base52.real.ne('N')].merge(d, on='archivo', suffixes=('_a', '_b'), validate='one_to_one')
    assert len(j) == 52 and j.prediccion_a.eq(j.prediccion_b).all() and j.grado_a.eq(j.grado_b).all()
    evidencia = {}
    for q in json.loads((anterior/'evidencia.json').read_text(encoding='utf-8')):
        for nombre in ('sin_esperada', 'con_esperada'):
            v = q[nombre]
            if v:
                evidencia[(q['archivo'], v['tipo'])] = v
    variantes = []
    for carpeta in ('resultados_tipo_por_sensor', 'resultados_tipo_promedio', 'resultados_tipo_envolvente_cuadrada'):
        ruta = AQUI/carpeta
        pred = pd.read_csv(ruta/'tipos_oof.csv')
        assert pred.vista_uid.is_unique and len(pred) == 56
        protocolo = json.loads((ruta/'protocolo.json').read_text(encoding='utf-8'))
        script = {'resultados_tipo_por_sensor':'ensayar_tipo_por_sensor.py',
            'resultados_tipo_promedio':'ensayar_tipo_promedio.py',
            'resultados_tipo_envolvente_cuadrada':'ensayar_tipo_envolvente_cuadrada.py'}[carpeta]
        assert sha(AQUI/script) == protocolo['codigo']
        for fold in json.loads((ruta/'pliegues.json').read_text(encoding='utf-8')):
            tr, te = m.loc[fold['train']], m.loc[fold['test']]
            assert set(tr.grupo).isdisjoint(te.grupo)
            assert set(tr.archivo).isdisjoint(te.archivo)
            assert tr.real.ne('N').all()
            if carpeta == 'resultados_tipo_por_sensor':
                assert tr.sensor.eq(fold['sensor']).all()
        pred['paper'] = pred.vista_uid.map(base52.set_index('vista_uid').paper)
        pred['grado'] = [evidencia.get((r.archivo,r.prediccion), {}).get('grado') for r in pred.itertuples()]
        # Para toda prediccion correcta hay evidencia del juez fijo ya calculada.
        # Los casos sin cache siguen contando como error de tipo, nunca como ND.
        assert pred.loc[pred.real.ne('N') & pred.real.eq(pred.prediccion), 'grado'].notna().all()
        r = contar(pred); r['variante'] = carpeta
        r['cambios_tipo_vs_base'] = int(pred.prediccion.ne(pred.vista_uid.map(base52.set_index('vista_uid').prediccion)).sum())
        variantes.append(r)
    pd.DataFrame(variantes).to_csv(SALIDA/'experimentos_desarrollo_52.csv', index=False)
    f = d[d.paper.notna()].copy()
    f['salida'] = f.grado.where(f.tipo_correcto, 'Tipo incorrecto')
    grados = ['ND','PD','D']; columnas = grados+['Tipo incorrecto']
    matriz = pd.crosstab(f.paper, f.salida).reindex(index=grados, columns=columnas, fill_value=0)
    assert int(matriz.to_numpy().sum()) == 105
    matriz.to_csv(SALIDA/'matriz_105.csv')
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.imshow(np.zeros((3,4)), cmap='Blues', vmin=0, vmax=1, alpha=.06)
    for i in range(3):
        for j in range(4):
            color = '#e3e3e3' if j == 3 else '#ccebd7' if j > i else '#dce8f4' if j == i else '#f5d2cc'
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor='white',linewidth=2))
            ax.text(j,i,str(int(matriz.iloc[i,j])),ha='center',va='center',fontsize=17)
    ax.set_xticks(range(4), ['ND','PD','D','Tipo\nincorrecto']); ax.set_yticks(range(3), grados)
    ax.set_xlabel('Diagnóstico obtenido'); ax.set_ylabel('Clasificación del paper')
    ax.set_title(f'Condición completa excluida del entrenamiento\n{resumen["sin_retroceso"]}/105 sin retroceso',pad=16)
    fig.text(.5,.09, f'Mejoran {resumen["mejoran"]} · Iguales {resumen["iguales"]} · Empeoran {resumen["empeoran"]} · Tipo incorrecto {resumen["tipo_incorrecto"]}',ha='center')
    fig.text(.5,.035,'Evaluación interna sobre datos ya inspeccionados. Grados propios; sin validación externa.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,1)); fig.savefig(SALIDA/'matriz_105.png',dpi=170); plt.close(fig)
    lineas = ['# Revisión de condiciones y experimentos de mejora', '',
        'El resultado histórico 88/105 mezcla OOF en desarrollo con un ajuste congelado en registros que comparten condición con entrenamiento.',
        'No se elimina: se conserva como resultado del protocolo mixto, pero no prueba generalización a condiciones nuevas.', '',
        'La reevaluación excluye del entrenamiento todas las cargas, sensores y muestreos de la condición evaluada.',
        'Conserva los 133 archivos, incluidos el corto sin diagnóstico y los cuatro normales. Reproduce los 52 OOF históricos caso por caso.',
        'Las condiciones OR conservan la posición angular en su identificador. Esto no garantiza independencia entre rodamientos físicos: los identificadores disponibles no la demuestran.', '',
        f'Resultado corregido: {resumen["sin_retroceso"]}/105 sin retroceso; {resumen["mejoran"]} mejoras, {resumen["empeoran"]} empeoramientos y {resumen["tipo_incorrecto"]} errores de tipo.',
        f'48 kHz: {resumen["fallas_48k_tipo_correcto"]}/24 tipos correctos, incluido un archivo insuficiente. Normales: 4/4.', '',
        '## Experimentos fijos sobre los 52 casos de desarrollo', '',
        'Clasificador histórico: 42/52 tipos correctos y 37/52 sin retroceso.',
        'Las variantes conservan detector, juez y separación temporal; no se ha reemplazado el modelo histórico.',
        'Los grados de las predicciones correctas se recuperan de la evidencia ya calculada para el mismo archivo y tipo, después de congelar la predicción. La etiqueta no participa en la selección del tipo ni del filtro.', '',
        pd.DataFrame(variantes).to_string(index=False), '',
        '## Alcance', '',
        'No se alcanzó el objetivo de al menos 90/105 sin retroceso y muchas más mejoras que empeoramientos.',
        'El dataset y el desarrollo han sido inspeccionados repetidamente. Ninguno de estos ensayos constituye validación externa ni demuestra ausencia de sobreajuste por selección reiterada.',
        'No se usa la frecuencia verdadera en inferencia, no se fabrican ND para errores y no se modifica la clasificación de referencia.', '',
        'Reproducir la verificación: `python 05_Diagnostico_Ciego/verificar_revision_condiciones.py`.', '']
    (SALIDA/'INFORME.md').write_text('\n'.join(lineas), encoding='utf-8')
    print('OK: 133 señales únicas, 105 referencias, exclusión de condición, OOF histórico reproducido y modelos originales intactos.')
    print(json.dumps(resumen)); print(pd.DataFrame(variantes).to_string(index=False))


if __name__ == '__main__':
    main()
