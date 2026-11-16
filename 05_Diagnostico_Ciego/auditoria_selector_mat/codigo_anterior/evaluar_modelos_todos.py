"""Modelo propio por muestreo: ejecutar 48k completo y consolidar los 133 archivos."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from predictor_48k import predecir,MODELO_48K
from predictor_espectral import MODELO as MODELO_12K
from alinear_bandas import geometria
from evaluar_no_retroceso import F,calificar
from integridad_senales import huellas_senales, detectar_filtraciones


if __name__=='__main__':
    aqui=Path(__file__).parent;datos=aqui.parent/'data2/CWRU'
    salida=aqui/'resultados_modelos_todos';salida.mkdir(exist_ok=True)
    firma48=hashlib.sha256(MODELO_48K.read_bytes()).hexdigest()
    firma12=hashlib.sha256(MODELO_12K.read_bytes()).hexdigest()
    fuente12=aqui/'resultados_criterio_usuario_total'
    anterior=pd.read_csv(fuente12/'comparacion.csv')
    assert json.loads((fuente12/'configuracion.json').read_text(encoding='utf-8'))['modelo_sha256']==firma12
    particion=pd.read_csv(MODELO_48K.parent/'particion.csv').set_index('archivo').particion.to_dict()
    oof=pd.read_csv(MODELO_48K.parent/'validacion.csv').set_index('archivo')
    idx=F.leer_indice(str(datos));idx48=idx[idx.fs==48000]
    huellas=huellas_senales(idx48,datos)
    fugas=detectar_filtraciones(huellas,
        {a for a,v in particion.items() if v=='desarrollo'},
        {a for a,v in particion.items() if v=='reserva'})
    nodos,L=geometria(48000)
    filas=[];evidencias=[]
    for _,r in idx48.iterrows():
        mat=loadmat(datos/r.archivo)
        clave=next(k for k in mat if k.endswith(f'_{r.sensor_falla}_time'))
        krpm=[k for k in mat if k.endswith('RPM')]
        rpm=float(np.ravel(mat[krpm[0]])[0]) if krpm else float(r.rpm_nominal)
        catalogo={k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
        x=np.ravel(mat[clave]);final=predecir(x,48000,catalogo)['tipo']
        q=calificar(x,48000,catalogo,final,L=L,criterio='usuario',nodos=nodos)
        parte=particion[r.archivo]
        if parte=='reserva':
            if r.archivo in fugas:
                evaluado=None;protocolo='reserva_copia_entrenamiento_no_validacion'
            else:
                evaluado=final;protocolo='reserva_modelo_congelado'
        elif bool(oof.loc[r.archivo,'evaluado']):
            evaluado=oof.loc[r.archivo,'prediccion_oof'];protocolo='desarrollo_fuera_de_condicion'
        else:
            evaluado=None;protocolo='entrenamiento_normal_sin_validacion'
        qe=(q if evaluado==final else calificar(x,48000,catalogo,evaluado,L=L,criterio='usuario',nodos=nodos)) if evaluado else None
        # La etiqueta se consulta después de fijar predicción y grado.
        filas.append(dict(archivo=r.archivo,registro=int(r.registro),carpeta=r.carpeta,real=r.zona,
            particion=parte,protocolo=protocolo,pred_final=final,grado_final=q['grado'],
            correcto_final=final==r.zona,pred_evaluacion=evaluado,
            grado_evaluacion=qe['grado'] if qe else None,
            correcto_evaluacion=evaluado==r.zona if evaluado else None))
        evidencias.append(dict(archivo=r.archivo,tipo_final=final,calidad_final=q,
                              tipo_evaluacion=evaluado,calidad_evaluacion=qe))
        pd.DataFrame(filas).to_csv(salida/'comparacion_48k_parcial.csv',index=False)
        print(f"{len(filas)}/{len(idx48)} {r.archivo}: final={final}; validacion={evaluado}; grado={qe['grado'] if qe else 'no_evaluado'}",flush=True)
    d=pd.DataFrame(filas);d.to_csv(salida/'comparacion_48k.csv',index=False)
    (salida/'evidencia_48k.json').write_text(json.dumps(evidencias,indent=2,allow_nan=False),encoding='utf-8')
    assert len(d)==28 and d.archivo.is_unique
    assert hashlib.sha256(MODELO_48K.read_bytes()).hexdigest()==firma48
    assert hashlib.sha256(MODELO_12K.read_bytes()).hexdigest()==firma12
    resumen=[]
    for protocolo,g in d.groupby('protocolo',sort=False):
        valid=g[g.pred_evaluacion.notna()]
        resumen.append(dict(protocolo=protocolo,total=len(g),evaluados=len(valid),
            correctos=int(valid.correcto_evaluacion.sum()) if len(valid) else None,
            porcentaje=round(100*valid.correcto_evaluacion.mean(),2) if len(valid) else None,
            D=int((valid.grado_evaluacion=='D').sum()),PD=int((valid.grado_evaluacion=='PD').sum()),
            ND=int((valid.grado_evaluacion=='ND').sum())))
    res=pd.DataFrame(resumen);res.to_csv(salida/'resumen_48k.csv',index=False)
    reserva=d[(d.particion=='reserva')&d.pred_evaluacion.notna()]
    pd.crosstab(reserva.real,reserva.pred_evaluacion).reindex(index=['B','IR','N'],columns=['B','IR','N'],fill_value=0).to_csv(salida/'matriz_reserva_48k.csv')
    globales=[]
    for r in anterior[anterior.fs==12000].itertuples():
        globales.append(dict(archivo=r.archivo,fs=12000,modelo='12k',origen=r.origen,real=r.tipo_real,
            prediccion=r.tipo_predicho,grado=r.grado_emitido,evaluable=True,tipo_correcto=r.tipo_correcto))
    for r in d.itertuples():
        globales.append(dict(archivo=r.archivo,fs=48000,modelo='48k',origen=r.protocolo,real=r.real,
            prediccion=r.pred_evaluacion,grado=r.grado_evaluacion,
            evaluable=pd.notna(r.pred_evaluacion),tipo_correcto=r.correcto_evaluacion))
    globales=pd.DataFrame(globales)
    assert len(globales)==133 and globales.archivo.is_unique and set(globales.archivo)==set(idx.archivo)
    globales.to_csv(salida/'consolidado_133.csv',index=False)
    config=dict(modelo_12k_sha256=firma12,modelo_48k_sha256=firma48,reentrenamiento=False,
        copias_reserva_excluidas=fugas,
        H=6,tol=.02,p=.01,nodos_48k=31,L_48k=6,ancho_final=375,
        advertencia='No confundir aciertos de entrenamiento con validación. Protocolos distintos, sin validación externa nueva.')
    (salida/'configuracion.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    lineas=['# Ejecución completa: modelo por frecuencia de muestreo','',
        '48 kHz: las 28 señales se ejecutaron con el nuevo modelo congelado y se calificaron con la regla del usuario.',
        'Reserva: 14 archivos ejecutados; dos excluidos por contener señales idénticas a entrenamiento. Quedan 12 evaluables.',
        'Desarrollo con falla: se usan sus 12 predicciones fuera de condición, en lugar de aciertos de entrenamiento.',
        'Normales de desarrollo: se ejecuta el modelo final, pero se excluyen de validación por haber entrenado con ellos.',
        'La reserva comparte condiciones con entrenamiento y ya se había visto históricamente; no es validación externa.',
        'No hay ejemplos de pista externa en el modelo de 48k. No se calcula no-retroceso del paper para estos archivos sin referencia transcrita.',
        '12 kHz: se conservan los 105 resultados ya calculados del modelo de 12k; los pesos y el criterio no cambiaron.',
        'El consolidado cubre 133 archivos: 129 evaluables, dos normales de entrenamiento y dos copias de señales de entrenamiento sin puntuar. Mezcla protocolos; no es validación independiente.',
        '', '| protocolo | total | evaluados | correctos | porcentaje | D | PD | ND |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    lineas += ['| '+' | '.join(map(str,r.values()))+' |' for r in resumen]
    fallas_res=reserva[reserva.real!='N'];normales_res=reserva[reserva.real=='N']
    lineas += ['',f'Reserva con falla: {int(fallas_res.correcto_evaluacion.sum())}/{len(fallas_res)} tipos correctos.',
        f'Normales de reserva: {int(normales_res.correcto_evaluacion.sum())}/{len(normales_res)} clasificaciones correctas.']
    (salida/'resumen.md').write_text('\n'.join(lineas)+'\n',encoding='utf-8')
    print(res.to_string(index=False));print('\n'.join(lineas[-2:]))
