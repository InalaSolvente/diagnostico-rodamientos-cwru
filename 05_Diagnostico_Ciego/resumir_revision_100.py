"""Comprueba y publica el balance de ensayos, incluidos los resultados negativos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from modelo_por_dominio import firma
from puntuar_tipos_fijos import contar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI / 'revision_meta_100'
ENSAYOS = [
    ('Referencia, grupos completos', 'resultados_grupos_por_defecto'),
    ('Preblanqueo AR32', 'resultados_preblanqueo_ar'),
    ('Plantillas coherentes', 'resultados_plantillas_coherentes'),
    ('Selector anidado con ranking', 'resultados_plantillas_ranking_anidado'),
    ('Plantillas y modulación', 'resultados_plantillas_modulacion'),
    ('Ranking físico monótono', 'resultados_ranker_fisico'),
    ('Retirar control PD', 'resultados_sin_control_PD'),
    ('Retirar ambos controles', 'resultados_sin_forzar_tipo'),
    ('Invariancia de portadora', 'resultados_invariancia_portadora'),
    ('Clasificador y control anidados', 'resultados_selector_controles_anidado'),
    ('Banco desplazado de 57 bandas', 'resultados_banco_desplazado'),
    ('Coherencia cíclica sola', 'resultados_coherencia_ciclica'),
    ('Crudo y coherencia cíclica', 'resultados_crudo_coherencia'),
    ('Tres vistas temporales crudo+ACP', 'resultados_temporales_acp'),
    ('Evidencia84 aprendida, sin forzar', 'resultados_juez_aprendido_acp'),
]


def main():
    SALIDA.mkdir(exist_ok=True)
    referencia = pd.read_csv(AQUI / ENSAYOS[0][1] / 'comparacion_133.csv').set_index('archivo')
    filas, fuentes = [], []
    for nombre, carpeta in ENSAYOS:
        archivo = AQUI / carpeta / 'comparacion_133.csv'
        d = pd.read_csv(archivo)
        assert len(d) == 133 and d.archivo.is_unique and d.paper.notna().sum() == 105
        b = d.set_index('archivo').loc[referencia.index]
        for columna in ('sha256_senal', 'real', 'sensor', 'fs'):
            assert b[columna].equals(referencia[columna]), (carpeta, columna)
        assert b.paper.fillna('').equals(referencia.paper.fillna(''))
        assert d.tipo_correcto.equals(d.real.eq(d.prediccion))
        c, nd = contar(d), contar(d, nd_emitido=True)
        guardado = json.loads((archivo.parent / 'resumen.json').read_text(encoding='utf-8'))
        for clave in c:
            assert c[clave] == guardado[clave], (carpeta, clave)
        filas.append(dict(ensayo=nombre, **c, desfavorables=c['empeoran'] + c['tipo_incorrecto'],
                          sin_retroceso_ND_emitido=nd['sin_retroceso']))
        fuentes.append(dict(carpeta=carpeta, sha256_comparacion=firma(archivo)))
    tabla = pd.DataFrame(filas)
    assert tabla.iloc[0].sin_retroceso == 79
    cotas = pd.read_csv(AQUI / 'auditoria_control_tipo/cotas.csv').set_index('control')
    assert cotas.loc['control_D_y_PD', 'techo_estricto'] == 98
    assert cotas.loc['control_D', 'techo_estricto'] == 100
    cotas_verificadas = pd.read_csv(AQUI / 'auditoria_cota_verificada/cotas.csv').set_index('banco')
    assert cotas_verificadas.cota_sin_retroceso.to_dict() == {31: 97, 57: 99, 63: 97}
    tabla.to_csv(SALIDA / 'comparacion.csv', index=False)
    desplazado = pd.read_csv(AQUI / 'resultados_banco_desplazado/comparacion_133.csv').set_index('archivo').loc[referencia.index]
    assert desplazado.prediccion.equals(referencia.prediccion)
    cambios = referencia[['paper', 'prediccion', 'tipo_correcto', 'grado']].rename(columns={'grado': 'grado31'}).copy()
    cambios['grado57'] = desplazado.grado
    cambios['delta_grado'] = cambios.grado57.map(ORDEN) - cambios.grado31.map(ORDEN)
    cambios[cambios.delta_grado.ne(0)].to_csv(SALIDA / 'cambios_banco57_vs31.csv')
    nuevo = pd.read_csv(AQUI / 'resultados_temporales_acp/comparacion_133.csv')
    nuevo['cohorte'] = np.select([nuevo.paper.notna(), nuevo.real.eq('N')],
        ['fallas_12k_con_paper', 'normales'], default='fallas_48k_sin_paper')
    nuevo['sin_diagnostico'] = nuevo.prediccion.eq('sin_diagnostico')
    cohortes = nuevo.groupby('cohorte').agg(total=('archivo', 'size'),
        tipo_correcto=('tipo_correcto', 'sum'), sin_diagnostico=('sin_diagnostico', 'sum'))
    assert cohortes.total.sum() == 133 and cohortes.loc['normales', 'tipo_correcto'] == 4
    cohortes.to_csv(SALIDA / 'tipos_por_cohorte.csv')
    verificacion_acp = json.loads((AQUI / 'resultados_crudo_coherencia/verificacion.json').read_text(encoding='utf-8'))
    verificacion_temporal = json.loads((AQUI / 'resultados_temporales_acp/verificacion.json').read_text(encoding='utf-8'))
    assert verificacion_temporal['archivos_recalculados'] == 132
    assert verificacion_temporal['ajustes_reproducidos'] == 22
    assert verificacion_temporal['codigo'] == firma(AQUI / 'verificar_temporales_y_juez_acp.py')
    # La reutilización de grados es válida solo con idéntica señal, catálogo y receta.
    cache_dir = AQUI / 'resultados_banco_desplazado'
    cache = pd.read_csv(cache_dir / 'comparacion_133.csv').set_index('archivo')
    pcache = json.loads((cache_dir / 'protocolo.json').read_text(encoding='utf-8'))
    assert pcache['selector'] == firma(AQUI / 'banco_desplazado.py')
    assert pcache['verificador'] == firma(AQUI / 'doble_banda_fijada.py')
    assert pcache['criterio'] == firma(AQUI / 'criterio_usuario.py')
    for carpeta in ('resultados_coherencia_ciclica', 'resultados_crudo_coherencia',
                    'resultados_temporales_acp', 'resultados_juez_aprendido_acp'):
        datos = pd.read_csv(AQUI / carpeta / 'comparacion_133.csv').set_index('archivo').loc[cache.index]
        assert datos.sha256_senal.equals(cache.sha256_senal) and datos.fs.equals(cache.fs)
        assert datos.catalogo.map(json.loads).tolist() == cache.catalogo.map(json.loads).tolist()
    columnas = [('ensayo', 'Ensayo'), ('sin_retroceso', 'Sin retroceso'), ('mejoran', 'Mejoras'),
                ('iguales', 'Iguales'), ('empeoran', 'Descensos'), ('tipo_incorrecto', 'Tipo incorrecto'),
                ('desfavorables', 'Desfavorables')]
    lineas = ['| ' + ' | '.join(titulo for _, titulo in columnas) + ' |',
              '| ' + ' | '.join('---' for _ in columnas) + ' |']
    lineas += ['| ' + ' | '.join(str(f[k]) for k, _ in columnas) + ' |' for f in filas]
    informe = '''# Revisión de la meta 100/105

    La meta sigue pendiente. La referencia más estricta obtiene 79/105 sin retroceso (75,2 %).
Tiene 13 mejoras y 26 resultados desfavorables: 4 descensos de grado y 22 tipos incorrectos.
El objetivo exige también muchas más mejoras que resultados desfavorables; tampoco se cumple.

El nuevo banco de 57 bandas desplazadas obtiene 80/105 con los mismos tipos predichos:
14 mejoras, 66 iguales, 3 descensos y 22 tipos incorrectos (25 desfavorables).
Es un ensayo exploratorio; no se han reemplazado los modelos históricos.

El ensayo aditivo de coherencia cíclica obtiene 82/105: 13 mejoras, 69 iguales,
3 descensos y 20 tipos incorrectos (23 desfavorables). Se verifica cada grupo fuera
del entrenamiento. La coherencia sola dio 64/105 y no se adopta. La meta sigue pendiente:
13 mejoras todavía son menos que 23 resultados desfavorables.

La ampliación temporal a tres vistas de selección obtiene 84/105: 14 mejoras, 70 iguales,
4 descensos y 17 tipos incorrectos (21 desfavorables). Son las mismas 52 señales de desarrollo;
las doce ventanas por archivo son vistas correlacionadas, no registros independientes.
El experimento que aprende la evidencia D/PD como 84 rasgos adicionales y elimina los
controles forzados da 74/105; se conserva como resultado negativo.

Se evalúan los mismos 133 archivos, de los cuales 105 tienen clasificación del paper.
El entrenamiento conserva los 52 registros con falla del desarrollo histórico. Cada grupo
de tipo y tamaño queda completamente fuera del ajuste al evaluarlo, incluyendo todas las
posiciones OR, cargas, sensores y muestreos. Las vistas de un mismo archivo normal permanecen juntas.
Los modelos reciben señal y catálogo completo del fabricante; la etiqueta y la frecuencia
verdadera de falla no son entradas de inferencia. Los ensayos siguientes son exploratorios:
la reutilización reiterada del dataset impide presentarlos como validación externa.

''' + '\n'.join(lineas) + '''

Desfavorables incluye todos los tipos incorrectos, además de los descensos con tipo correcto.
No se selecciona retrospectivamente la mejor salida para cada archivo. Los modelos históricos
no se han reemplazado. Retirar el control PD o ambos controles es una regla uniforme para todos los registros.
Sus tipos se congelaron antes de consultar la verificación de las bandas en el segmento futuro.

## Hallazgo estructural

Al incorporar la verificación futura a la auditoría, los selectores de los bancos31 y63
solo permiten conservar 97/105 aun concediéndoles un clasificador y un detector perfectos.
Esto prueba que mejorar exclusivamente el clasificador no basta con esos selectores.
El banco57 aumenta esa cota a 99/105: tampoco permite todavía 100 con su selector fijo.
La auditoría usa etiquetas exclusivamente fuera de la inferencia; no representa el
desempeño de un modelo ciego. Véase [la cota verificada](../auditoria_cota_verificada/INFORME.md).

El control PD puede forzar una familia incorrecta incluso cuando el clasificador ya había
identificado la correcta. Su cota optimista, combinada con el banco31 actual, es 98/105.
Sin ese control, la cota de selección es 100/105. Son límites superiores calculados en una
auditoría con etiquetas, no aciertos de un modelo ni predicciones utilizables en inferencia.
La verificación futura puede reducir aún más esos límites. Véase
[la auditoría](../auditoria_control_tipo/INFORME.md).

Retirarlo resta rendimiento observado (77/105 frente a 79/105), por lo que una solución para
100 necesita resolver tanto la clasificación como la evidencia de las bandas. No se aplican
excepciones a los dos archivos perjudicados ni se altera el criterio de diagnóstico.

El banco57 añade todas las bandas desplazadas medio ancho de forma uniforme. Conserva
dos ramas disjuntas, los armónicos del catálogo, el juez y la separación entre selección
y verificación. Ajusta la probabilidad nominal a .01/(57*5*6*2) para incluir las búsquedas
adicionales. Esa corrección nominal no sustituye una calibración empírica de falsos positivos.
La selección anidada de clasificadores y controles dio 74/105; completar el ensayo de
invariancia de portadora dio 77/105. Se conservan ambos resultados negativos.

## Criterio ND y siguiente etapa

Se publican dos conteos mientras se resuelve la consulta sobre ND. La tabla mantiene la
convención estricta de las matrices recientes. La función original cuenta ND→ND como acierto
sin exigir tipo interno: solo cambia AR32, de 73 a 74; la referencia permanece en 79.
Ambos conteos están en comparacion.csv y no modifican diagnósticos.

La ampliación del entrenamiento está propuesta, pero no ejecutada: continúa vigente la
restricción del usuario a la mitad de los registros con falla. El plan excluye completamente
el grupo evaluado y está en [plan_entrenamiento_ampliado](../plan_entrenamiento_ampliado/).
Requiere resolver la consulta pendiente antes de utilizar más etiquetas de entrenamiento.

La comprobación reproducible de este balance es `python 05_Diagnostico_Ciego/resumir_revision_100.py`.
'''
    informe += '\n## Comparación pareada y cobertura completa\n\n'
    pares = verificacion_acp['comparacion_pareada']
    informe += (f"Frente al banco57 previo, el ensayo aditivo recupera {pares['recuperados']} casos "
                f"y pierde {pares['perdidos']}; la ganancia neta es 2. No se mezclan las mejores "
                'salidas de ambos modelos por archivo.\n\n')
    informe += 'Los 133 archivos permanecen en la evaluación; el archivo corto de 48 kHz sigue como sin diagnóstico:\n\n'
    informe += '```\n' + cohortes.to_string() + '\n```\n'
    pares_temporales = verificacion_temporal['comparacion_pareada']
    informe += (f"\nFrente al ensayo de 82/105, las tres vistas recuperan {pares_temporales['recuperados']} casos "
                f"y pierden {pares_temporales['perdidos']}. Se conserva el mismo criterio y el mismo banco57. "
                'Las vistas de 2.5 s comienzan al inicio, al centro y al final de la primera mitad; '
                'ninguna termina después de esa frontera.\n')
    (SALIDA / 'INFORME.md').write_text(informe, encoding='utf-8')
    texto_temporal = f'''# Tres vistas temporales de selección

El ensayo obtiene **84/105 sin retroceso**: 14 mejoras, 70 iguales, 4 descensos y
17 tipos incorrectos. Los 21 resultados desfavorables superan las 14 mejoras:
la meta de 100/105 y el criterio de muchas más mejoras siguen pendientes.

Frente a 82/105, recupera {pares_temporales['recuperados']} casos y pierde
{pares_temporales['perdidos']}. No se elige la mejor respuesta por archivo.
Los cambios se detallan en `cambios_vs_82.csv`.

Se extraen tres vistas de 2.5 segundos al inicio, centro y final de la primera
mitad. Cada una aporta cuatro ventanas de rasgos crudos más ACP; el LDA agrega
sus puntuaciones. Son doce vistas correlacionadas de cada archivo, no doce
casos independientes. El corpus de desarrollo sigue teniendo 52 archivos con
falla; cada ajuste excluye todo el tipo y tamaño del grupo evaluado, incluidas
todas las posiciones OR, cargas, sensores y muestreos.

El catálogo completo disponible, fs y la señal son las entradas físicas.
Las etiquetas solo se emplean en entrenamiento y evaluación. No se utiliza
la frecuencia verdadera de falla. Se mantienen el detector, los controles D/PD,
el banco57 y el criterio de diagnóstico. La segunda mitad solo verifica las
bandas fijadas en selección; no aporta ventanas al clasificador.

La verificación conjunta reprodujo los 22 ajustes exteriores de este ensayo y
del clasificador con evidencia84, y recalculó las doce vistas desde los 132 MAT
de longitud suficiente. También reprodujo los grados de los diez archivos cuyo
tipo cambió, comprobó las fronteras temporales y modificó el futuro a 12/48 kHz
sin alterar los rasgos. El archivo corto sigue dentro de los 133 como sin
diagnóstico. Véase `verificacion.json`.

El clasificador alternativo que añade los 84 indicadores D/PD como rasgos y
elimina los forzados obtuvo 74/105; no se adopta. Sus indicadores se reprodujeron
desde la primera mitad de los 127 archivos donde se usan, para las tres hipótesis,
con diferencia máxima cero. Esa alternativa aprende el tipo, no cambia el juez.

Todo es exploratorio, sin validación externa. Los modelos históricos no se han
reemplazado. Los supuestos previos del catálogo y la geometría siguen vigentes.

Reproducción desde la raíz:

```powershell
python .\\05_Diagnostico_Ciego\\rasgos_temporales_acp.py --preparar
python .\\05_Diagnostico_Ciego\\evaluar_temporales_acp.py
python .\\05_Diagnostico_Ciego\\evaluar_juez_aprendido_acp.py
python .\\05_Diagnostico_Ciego\\verificar_juez84_fuente.py
python .\\05_Diagnostico_Ciego\\verificar_temporales_y_juez_acp.py
```

La comparación completa y la cobertura de los 133 registros están en
[la revisión de la meta](../revision_meta_100/INFORME.md).
'''
    (AQUI / 'resultados_temporales_acp/INFORME.md').write_text(texto_temporal, encoding='utf-8')
    (SALIDA / 'verificacion.json').write_text(json.dumps(dict(codigo=firma(Path(__file__)),
        mismos_133_archivos_y_105_referencias=True, conteos_recalculados=True,
        cotas_verificadas=firma(AQUI / 'auditoria_cota_verificada/cotas.csv'),
        verificacion_ACP=firma(AQUI / 'resultados_crudo_coherencia/verificacion.json'),
        verificacion_temporal=firma(AQUI / 'resultados_temporales_acp/verificacion.json'),
        verificacion_juez84=firma(AQUI / 'resultados_juez_aprendido_acp/verificacion_fuente84.json'),
        cache_banco57_catalogo_senal_fs_y_codigo_verificados=True,
        fuentes=fuentes, objetivo_alcanzado=False, modelos_sustituidos=False), indent=2), encoding='utf-8')
    print(tabla.to_string(index=False))
    print(cohortes.to_string())


if __name__ == '__main__':
    main()
