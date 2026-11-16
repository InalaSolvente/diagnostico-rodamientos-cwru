"""Métricas de tipo después de congelar predicciones; nunca cambian decisiones."""
from pathlib import Path
import json
import pandas as pd


def guardar(v, salida, *, corpus_fallas):
    salida = Path(salida)
    congelado = pd.read_csv(salida / 'tipos_133.csv')
    assert len(v) == 133 and v.archivo.is_unique and v.sha256_senal.is_unique
    assert congelado.archivo.equals(v.archivo) and congelado.prediccion.equals(v.prediccion)
    assert v.tipo_correcto.equals(v.real.eq(v.prediccion))
    f = v.paper.notna()
    assert f.sum() == 105 and set(v.prediccion) <= {'B', 'IR', 'OR', 'N', 'sin_diagnostico'}
    incorrectos = int((~v.loc[f, 'tipo_correcto']).sum())
    r = dict(total_objetivo=105, tipo_incorrecto=incorrectos, tipo_correcto=105-incorrectos,
        meta_tipos=5, meta_tipos_alcanzada=incorrectos <= 5, total_archivos=133,
        tipo_incorrecto_133=int((~v.tipo_correcto).sum()), corpus_fallas=int(corpus_fallas),
        entrenamiento='solo fuera del grupo evaluado; cantidades exactas en pliegues.json',
        normales_correctos=int(v.loc[v.real.eq('N'), 'tipo_correcto'].sum()),
        sin_diagnostico=int(v.prediccion.eq('sin_diagnostico').sum()),
        grados_evaluados=False, validacion_externa=False)
    (salida / 'resumen_tipos.json').write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
    v[f & ~v.tipo_correcto].to_csv(salida / 'errores_tipo_105.csv', index=False)
    for nombre, datos in [('105', v[f]), ('133', v)]:
        m = pd.crosstab(datos.real, datos.prediccion).reindex(index=['B', 'IR', 'OR', 'N'],
            columns=['B', 'IR', 'OR', 'N', 'sin_diagnostico'], fill_value=0)
        assert int(m.to_numpy().sum()) == len(datos)
        m.to_csv(salida / f'matriz_tipo_{nombre}.csv')
    print(json.dumps(dict(variante=salida.name, **r), ensure_ascii=False), flush=True)
    return r
