"""Verifica cobertura completa, invariantes de criterio y separación de referencias."""
from pathlib import Path
import hashlib,json
import pandas as pd
from criterio_usuario import comparar


if __name__=="__main__":
    p=Path(__file__).parent
    salida=p/"resultados_criterio_usuario_total"
    d=pd.read_csv(salida/"comparacion.csv")
    idx=pd.read_csv(p.parent/"data2/CWRU/indice_cwru.csv")
    assert len(d)==133 and d.archivo.is_unique and set(d.archivo)==set(idx.archivo)
    assert d.groupby('carpeta').size().to_dict()=={'12k_DE':60,'12k_FE':45,'48k_DE':24,'Normal':4}
    assert d.paper.notna().sum()==105 and d.loc[d.carpeta.isin(['Normal','48k_DE']),'paper'].isna().all()
    for r in d[d.paper.notna()].itertuples():
        grado,exito=comparar(r.paper,r.grado_emitido,r.tipo_correcto)
        assert grado==r.grado_evaluable and exito==r.sin_retroceso
    desarrollo=pd.read_csv(p/'resultados_criterio_usuario_desarrollo/comparacion.csv').set_index('archivo')
    actual=d.set_index('archivo')
    assert desarrollo.grado_emitido.equals(actual.loc[desarrollo.index,'grado_emitido'])
    evidencia=json.loads((salida/'evidencia.json').read_text(encoding='utf-8'))
    assert {e['archivo'] for e in evidencia}==set(d.archivo)
    for e in evidencia:
        q=e.get('evidencia',{})
        if e['grado']=='D': assert q['primera_identificada'] and len(q['identificadas'])>=2 and not q['hay_pico_previo_mayor']
        if e['grado']=='PD': assert len(q['identificadas'])>=1 and q['max_izquierda']>q['potencia_referencia_izquierda']
    config=json.loads((salida/'configuracion.json').read_text(encoding='utf-8'))
    assert config['modelo_sha256']==hashlib.sha256((p/'resultados_espectrales_desarrollo/modelo.npz').read_bytes()).hexdigest()
    assert d.loc[d.paper.notna(),'sin_retroceso'].sum()==90
    print('OK: 133 archivos, regla por salida, referencia en 105, desarrollo reproducido y modelo intacto')
    print(d.groupby('carpeta').agg(total=('archivo','size'),tipo_correcto=('tipo_correcto','sum')).to_string())
    print('Tipos en 48k:',d.loc[d.carpeta=='48k_DE','tipo_predicho'].value_counts().to_dict())
