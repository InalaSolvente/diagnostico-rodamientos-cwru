from pathlib import Path
import json,hashlib
import pandas as pd


if __name__=='__main__':
    p=Path(__file__).parent;s=p/'resultados_modelos_todos'
    d=pd.read_csv(s/'comparacion_48k.csv');g=pd.read_csv(s/'consolidado_133.csv')
    split=pd.read_csv(p/'resultados_modelo_48k/particion.csv').set_index('archivo')
    oof=pd.read_csv(p/'resultados_modelo_48k/validacion.csv').set_index('archivo')
    assert len(d)==28 and d.archivo.is_unique and len(g)==133 and g.archivo.is_unique
    assert d.particion.eq('reserva').sum()==14
    for r in d.itertuples():
        assert split.loc[r.archivo,'particion']==r.particion
        if r.protocolo=='reserva_copia_entrenamiento_no_validacion':
            assert pd.isna(r.pred_evaluacion) and pd.isna(r.correcto_evaluacion)
        elif r.particion=='reserva':assert r.pred_final==r.pred_evaluacion
        elif r.real=='N':assert pd.isna(r.pred_evaluacion) and pd.isna(r.correcto_evaluacion)
        else:assert r.pred_evaluacion==oof.loc[r.archivo,'prediccion_oof']
    assert d.pred_evaluacion.notna().sum()==24 and g.evaluable.sum()==129
    assert set(d.loc[d.protocolo.eq('reserva_copia_entrenamiento_no_validacion'),'archivo'])=={
        '48k_DE/IR021_2.mat','Normal/Normal_1.mat'}
    config=json.loads((s/'configuracion.json').read_text(encoding='utf-8'))
    for nombre,key in [('resultados_modelo_48k/modelo_48k.npz','modelo_48k_sha256'),('resultados_espectrales_desarrollo/modelo.npz','modelo_12k_sha256')]:
        assert hashlib.sha256((p/nombre).read_bytes()).hexdigest()==config[key]
    print('OK: 133 archivos, 129 evaluables; dos normales de entrenamiento y dos copias excluidos; pesos intactos')
