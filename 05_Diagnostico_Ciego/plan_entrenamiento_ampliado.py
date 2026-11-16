"""Prepara una propuesta de particiones; no ajusta ni modifica modelos."""
from pathlib import Path
import json
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma

AQUI=Path(__file__).parent
SALIDA=AQUI/'plan_entrenamiento_ampliado'


def main():
    SALIDA.mkdir(exist_ok=True)
    idx=F.leer_indice(str(AQUI.parent/'data2/CWRU'))
    meta=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    base=pd.read_csv(AQUI/'resultados_grupos_por_defecto/comparacion_133.csv').set_index('archivo')
    vistas=[]
    for r in idx.itertuples():
        grupo=base.loc[r.archivo,'grupo_auditoria']
        vistas.append(dict(archivo=r.archivo,vista_uid=r.archivo+'|'+r.sensor_falla,
            grupo=grupo,sensor=r.sensor_falla,real=r.zona,primaria=True,
            util=bool(base.loc[r.archivo,'suficiente'])))
        if r.zona=='N':
            m=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            de=m[F.clave_canal(m,'DE',int(r.registro))].size
            fe=m[F.clave_canal(m,'FE',int(r.registro))].size
            assert de==fe,'No asumir mismo fs entre canales con distinta longitud'
            vistas.append(dict(archivo=r.archivo,vista_uid=r.archivo+'|FE',grupo=grupo,
                sensor='FE',real='N',primaria=False,util=True))
    d=pd.DataFrame(vistas)
    assert len(d)==137 and d.vista_uid.is_unique and d.archivo.nunique()==133
    particiones=[]
    for grupo in sorted(set(d.grupo)):
        prueba=d.grupo.eq(grupo) & d.primaria
        ampliar=d.grupo.ne(grupo) & d.util
        actual=ampliar & d.vista_uid.isin(meta.vista_uid)
        assert set(d.loc[ampliar,'archivo']).isdisjoint(d.loc[prueba,'archivo'])
        assert not d.loc[ampliar,'grupo'].eq(grupo).any()
        particiones.append(dict(grupo_evaluado=grupo,prueba=d.loc[prueba,'archivo'].tolist(),
            entrenamiento_actual=d.loc[actual,'vista_uid'].tolist(),
            entrenamiento_propuesto=d.loc[ampliar,'vista_uid'].tolist(),
            archivos_train_actual=d.loc[actual,'archivo'].nunique(),
            archivos_train_propuesto=d.loc[ampliar,'archivo'].nunique()))
    propuesta=dict(estado='propuesta_pendiente_de_respuesta; ningun modelo entrenado',
        evaluacion_archivos=133,referencia_paper=105,desarrollo_actual_fallas=52,
        propuesta='usar todos los registros útiles fuera del grupo completo evaluado',
        invariantes='tipo+tamaño juntos, todas las posiciones OR/cargas/sensores/fs fuera; normal DE/FE por archivo juntos',
        senal_insuficiente='conservada en evaluación; nunca usada en entrenamiento',
        interpretacion='validación cruzada por grupos; no reserva nueva ni validación externa',
        codigo=firma(Path(__file__)),particiones=particiones)
    (SALIDA/'propuesta.json').write_text(json.dumps(propuesta,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([{k:p[k] for k in ('grupo_evaluado','archivos_train_actual','archivos_train_propuesto')} for p in particiones]).to_csv(SALIDA/'conteos.csv',index=False)
    print('OK: propuesta revisable, 133 casos de evaluación y 105 referencias; cero grupos compartidos.')
    print(pd.DataFrame([{k:p[k] for k in ('grupo_evaluado','archivos_train_actual','archivos_train_propuesto')} for p in particiones]).to_string(index=False))


if __name__=='__main__':main()
