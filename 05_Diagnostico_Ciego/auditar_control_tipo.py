"""Cotas de alcanzabilidad de los controladores, nunca usadas para diagnosticar."""
from pathlib import Path
import json
import pandas as pd
from modelo_ciego import F
from modelo_por_dominio import firma
from control_ordenes_giro import elegir as elegir_d
from ensayar_pd_persistente import elegir as elegir_pd
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
SALIDA=AQUI/'auditoria_control_tipo'
TIPOS=('IR','OR','B')


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente=AQUI/'resultados_correccion_d_persistente/evidencia.json'
    alternativas={q['archivo']:q['alternativas_seleccion'] for q in json.loads(fuente.read_text(encoding='utf-8'))}
    d=pd.read_csv(AQUI/'resultados_grupos_por_defecto/comparacion_133.csv')
    filas=[]
    for r in d[d.paper.notna()].itertuples():
        a=alternativas[r.archivo];assert set(a)==set(TIPOS)
        alcanzables_d={elegir_d(t,a,F.MULTIPLICADORES[r.sensor])[0] for t in TIPOS}
        alcanzables_pd={elegir_pd(t,a) for t in alcanzables_d}
        assert r.prediccion in alcanzables_pd
        # Las etiquetas se consultan solo tras enumerar todos los posibles tipos.
        grado_posible=ORDEN[a[r.real]['grado']]>=ORDEN[r.paper]
        filas.append(dict(archivo=r.archivo,paper=r.paper,tipo_real_solo_auditoria=r.real,
            max_grado_seleccion=a[r.real]['grado'],alcanzables_D=';'.join(sorted(alcanzables_d)),
            alcanzables_D_PD=';'.join(sorted(alcanzables_pd)),
            sin_control_posible=grado_posible,
            con_D_posible=grado_posible and r.real in alcanzables_d,
            con_D_PD_posible=grado_posible and r.real in alcanzables_pd,
            control_impide_tipo_real=r.real not in alcanzables_pd))
    b=pd.DataFrame(filas);assert len(b)==105 and b.archivo.is_unique
    b.to_csv(SALIDA/'alcanzabilidad_105.csv',index=False)
    cotas=[]
    for nombre,col in [('sin_forzar_tipo','sin_control_posible'),('control_D','con_D_posible'),('control_D_y_PD','con_D_PD_posible')]:
        estricto=int(b[col].sum())
        nd_optimista=int((b[col] | b.paper.eq('ND')).sum())
        cotas.append(dict(control=nombre,techo_estricto=estricto,techo_ND_emitido_optimista=nd_optimista,
            total=105,tipo='cota optimista, no desempeño de un modelo'))
    pd.DataFrame(cotas).to_csv(SALIDA/'cotas.csv',index=False)
    extras=b[~b.con_D_PD_posible & b.sin_control_posible]
    extras.to_csv(SALIDA/'perdidas_por_control.csv',index=False)
    informe=['# Límite estructural del control de tipo','',
        'Se enumeran los tres posibles resultados del clasificador, sin utilizar la etiqueta para elegir ninguno. Después se comprueba qué tipos permite emitir cada controlador.',
        'Esto es una auditoría de alcanzabilidad, no una predicción ni un resultado oracular presentado como modelo.', '',
        pd.DataFrame(cotas).to_string(index=False),'',
        'Casos adicionales que el controlador completo impide conservar, aunque el clasificador inicialmente identificara bien el elemento:', '',
        extras.to_string(index=False),'',
        'La cota combina el tipo alcanzable y el mejor grado de selección del banco31. La verificación solo puede mantener o reducir ese grado, por lo que la cifra es optimista.',
        'Para ND se muestra también una cota que permite cualquier ND efectivamente emitido: no se modifica el conteo de las matrices ni se inventan diagnósticos.',
        'La conclusión corresponde a estas reglas y este banco sobre señal cruda; no es un límite universal del dataset ni de otros filtros.',
        'No se excluye ningún registro y estas listas no son entradas de ningún predictor.', '',
        'Reproducir: `python 05_Diagnostico_Ciego/auditar_control_tipo.py`.','']
    (SALIDA/'INFORME.md').write_text('\n'.join(informe),encoding='utf-8')
    (SALIDA/'procedencia.json').write_text(json.dumps(dict(codigo=firma(Path(__file__)),
        fuente=firma(fuente),control_D=firma(AQUI/'control_ordenes_giro.py'),
        control_PD=firma(AQUI/'ensayar_pd_persistente.py'),diagnosticos_modificados=False),indent=2),encoding='utf-8')
    print(pd.DataFrame(cotas).to_string(index=False))
    print(extras[['archivo','paper','alcanzables_D_PD']].to_string(index=False))


if __name__=='__main__':main()
