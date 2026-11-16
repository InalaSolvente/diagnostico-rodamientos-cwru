"""Verificación cruzada de cobertura, procedencia y conservación del ensayo de 52."""
from pathlib import Path
import hashlib,json
import pandas as pd


if __name__=='__main__':
    p=Path(__file__).parent;s=p/'comparacion_con_sin_esperada_todos'
    d=pd.read_csv(s/'comparacion_133.csv')
    previo=pd.read_csv(p/'comparacion_con_sin_esperada/comparacion_pareada.csv').set_index('archivo')
    actual=d.set_index('archivo')
    assert len(d)==133 and d.archivo.is_unique and d.sha256_senal.is_unique
    assert d.paper.notna().sum()==105 and d.real.eq('N').sum()==4
    assert d[d.real.eq('N')].grado_con_esperada.isna().all()
    assert d.oraculo_aplica.sum()==129
    for columna in ['paper','tipo_sin_esperada','grado_sin_esperada','tipo_correcto','grado_con_esperada']:
        assert previo[columna].equals(actual.loc[previo.index,columna]),columna
    cortas=d[~d.suficiente]
    assert cortas.tipo_sin_esperada.eq('sin_diagnostico').all()
    assert cortas.origen.eq('sin_diagnostico_por_duracion').all()
    config=json.loads((s/'configuracion.json').read_text(encoding='utf-8'))
    assert not config['reentrenamiento'] and not config['seleccion_por_resultado_total']
    for ruta,firma in config['hashes'].items():
        assert hashlib.sha256((p/ruta).read_bytes()).hexdigest()==firma
    for nombre in ['matriz_con_esperada_105.csv','matriz_sin_esperada_105.csv']:
        t=pd.read_csv(s/nombre,index_col=0);assert t.to_numpy().sum()==105
        assert t.sum(axis=1).to_dict()=={'ND':18,'PD':17,'D':70}
    print('OK: 133 registros únicos; 52 resultados previos idénticos; matrices con los mismos 105 casos; normales sin oráculo inventado; pesos congelados.')
    print('Señales insuficientes:',cortas.archivo.tolist())
