"""Compara los dos conteos ND sin modificar una sola predicción del modelo."""
from pathlib import Path
import json
import pandas as pd
from criterio_usuario import comparar
from puntuar_tipos_fijos import contar
from modelo_por_dominio import firma

AQUI=Path(__file__).parent
SALIDA=AQUI/'auditoria_conteo_ND'


def main():
    assert comparar('ND','ND',False)==('ND',True)
    assert comparar('D','ND',False)==('ND',False)
    assert comparar('ND','PD',False)==('tipo_incorrecto',False)
    assert comparar('PD','D',True)==('D',True)
    SALIDA.mkdir(exist_ok=True)
    fuentes=[('referencia_grupos_completos','resultados_grupos_por_defecto','grado','OR_juntas'),
        ('preblanqueo_AR32','resultados_preblanqueo_ar','grado','OR_juntas'),
        ('curvas_coherentes_LDA_SVM','resultados_plantillas_coherentes','grado','OR_juntas'),
        ('curvas_menu_con_ranker','resultados_plantillas_ranking_anidado','grado','OR_juntas'),
        ('curvas_y_modulacion','resultados_plantillas_modulacion','grado','OR_juntas'),
        ('protocolo_anterior_88','resultados_pd_doble','grado31','OR_separadas')]
    filas=[];diferencias=[]
    referencia=pd.read_csv(AQUI/'resultados_grupos_por_defecto/comparacion_133.csv').set_index('archivo')
    for nombre,carpeta,col,grupos in fuentes:
        p=AQUI/carpeta/'comparacion_133.csv'
        if not p.exists():continue
        d=pd.read_csv(p)
        assert len(d)==133 and d.archivo.is_unique and d.paper.notna().sum()==105
        assert d.set_index('archivo').paper.equals(referencia.paper)
        d=d.assign(grado=d[col],tipo_correcto=d.real.eq(d.prediccion))
        estricto=contar(d);original=contar(d,nd_emitido=True)
        cambia=d.paper.eq('ND') & d.grado.eq('ND') & ~d.tipo_correcto
        assert original['sin_retroceso']-estricto['sin_retroceso']==int(cambia.sum())
        for r in d[cambia].itertuples():
            diferencias.append(dict(modelo=nombre,archivo=r.archivo,grado_emitido=r.grado,
                hipotesis_interna=r.prediccion,tipo_real_solo_evaluacion=r.real,paper=r.paper))
        for etiqueta,c in [('tipo_exigido_en_ND',estricto),('ND_emitido_sin_diagnostico',original)]:
            negativos=c['empeoran']+c['tipo_incorrecto']
            filas.append(dict(modelo=nombre,agrupacion=grupos,criterio=etiqueta,**c,
                desfavorables_total=negativos,
                razon_mejoras_desfavorables=c['mejoran']/negativos if negativos else None,
                diagnosticos_sha256=firma(p)))
    tabla=pd.DataFrame(filas)
    tabla.to_csv(SALIDA/'comparacion_criterios.csv',index=False)
    pd.DataFrame(diferencias,columns=['modelo','archivo','grado_emitido','hipotesis_interna',
        'tipo_real_solo_evaluacion','paper']).to_csv(SALIDA/'casos_que_cambian.csv',index=False)
    informe=['# Dos formas de contar un ND ya emitido','',
        'La función original `criterio_usuario.comparar` cuenta ND→ND como acierto sin exigir el tipo interno. Varias matrices posteriores usaron tipo correcto como condición para todo acierto, incluido ND.',
        'Este informe calcula ambos criterios sobre los mismos diagnósticos. No transforma un PD/D equivocado en ND, no modifica etiquetas y no vuelve a entrenar modelos.',
        'La elección del criterio para la meta de 100/105 se consultó al usuario y permanece pendiente. Se conserva el conteo estricto como referencia provisional.',
        'Los resultados desfavorables incluyen tanto disminuciones de grado como tipos incorrectos. Comparar solo mejoras contra disminuciones de grado ocultaría diagnósticos erróneos.',
        'Los protocolos con posiciones OR separadas se etiquetan aparte y no deben presentarse como equivalentes al control que excluye todas las posiciones juntas.', '',
        tabla.drop(columns=['diagnosticos_sha256']).to_string(index=False),'',
        'Fuente del criterio: [criterio_usuario.py](../criterio_usuario.py).',
        'Reproducir: `python 05_Diagnostico_Ciego/auditar_conteo_nd.py`.','']
    (SALIDA/'INFORME.md').write_text('\n'.join(informe),encoding='utf-8')
    (SALIDA/'verificacion.json').write_text(json.dumps(dict(diagnosticos_modificados=False,
        comparador_sha256=firma(AQUI/'criterio_usuario.py'),registros_por_modelo=133,
        referencia_por_modelo=105,criterio_usuario_pendiente=True),indent=2),encoding='utf-8')
    print(tabla[['modelo','agrupacion','criterio','sin_retroceso','mejoran','desfavorables_total']].to_string(index=False))


if __name__=='__main__':main()
