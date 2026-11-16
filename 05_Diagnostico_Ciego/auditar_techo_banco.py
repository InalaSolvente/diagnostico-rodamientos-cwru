"""Cota optimista del banco31; NO es una predicción ni una métrica del modelo.

Todas las familias se calcularon antes sin etiqueta. Solo en este análisis se
consulta la familia real para saber si el banco podría conservar el grado del
paper incluso con identificación perfecta. Nunca modifica diagnósticos.
"""
from pathlib import Path
import json
import pandas as pd
from modelo_por_dominio import firma
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
SALIDA=AQUI/'auditoria_techo_banco'


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente=AQUI/'resultados_correccion_d_persistente'
    elecciones={q['archivo']:q['alternativas_seleccion'] for q in json.loads((fuente/'evidencia.json').read_text(encoding='utf-8'))}
    protocolo=json.loads((fuente/'protocolo.json').read_text(encoding='utf-8'))
    assert protocolo['bandas']==firma(AQUI/'bandas_estables.py')
    assert protocolo['criterio']==firma(AQUI/'criterio_usuario.py')
    d=pd.read_csv(AQUI/'resultados_grupos_por_defecto/comparacion_133.csv')
    filas=[]
    for r in d[d.paper.notna()].itertuples():
        a=elecciones[r.archivo]
        assert set(a)=={'IR','OR','B'}
        assert all(q['p_prueba']==.01/(31*5*6*2) for q in a.values())
        # Exclusivamente análisis posterior: la inferencia no recibe r.real.
        q=a[r.real]
        filas.append(dict(archivo=r.archivo,paper=r.paper,tipo_real_solo_auditoria=r.real,
            max_grado_seleccion=q['grado'],posible_sin_retroceso=ORDEN[q['grado']]>=ORDEN[r.paper]))
    b=pd.DataFrame(filas)
    assert len(b)==105 and b.archivo.is_unique
    # El mejor grado de selección domina el de cualquier rama del mismo banco:
    # ordenar primero por grado y exigir selección+verificación impide superarlo.
    for q in json.loads((AQUI/'resultados_pd_doble/evidencia.json').read_text(encoding='utf-8')):
        if q['archivo'] not in set(b.archivo):continue
        limite=ORDEN[elecciones[q['archivo']][q['tipo']]['grado']]
        for rama in q['banco31']['ramas']:
            assert ORDEN[rama['seleccion']['grado']]<=limite
            assert ORDEN[rama['verificacion']['grado']]<=ORDEN[rama['seleccion']['grado']]
    b.to_csv(SALIDA/'cota_seleccion_105.csv',index=False)
    resumen=dict(tipo='cota optimista, no resultado de diagnostico',total=105,
        techo_banco31=int(b.posible_sin_retroceso.sum()),
        casos_imposibles_con_banco_y_juez_actuales=int((~b.posible_sin_retroceso).sum()),
        regla='grado final de cada rama <= grado en seleccion <= mejor grado de seleccion para la familia verdadera',
        alcance='banco31, envolvente amplitud/cuadrada, H6, mismo umbral; no aplica a otros preprocesados',
        codigo=firma(Path(__file__)),elecciones_sha256=firma(fuente/'evidencia.json'))
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resumen,ensure_ascii=False))
    print(b[~b.posible_sin_retroceso].to_string(index=False))


if __name__=='__main__':main()
