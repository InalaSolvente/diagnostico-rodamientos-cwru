"""Mismo LDA con crudos2021+ACP990; congela variantes con y sin controles."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from modelo_por_dominio import crear, firma
from ensayar_tipo_por_sensor import predecir_tipo
from evaluar_retirada_control_pd import controlar

AQUI = Path(__file__).parent
FUENTE = AQUI / 'resultados_acp_denso'


def guardar_resumen_tipos(v, salida):
    assert len(v) == 133 and v.archivo.is_unique
    assert v.tipo_correcto.equals(v.real.eq(v.prediccion))
    f = v.paper.notna()
    assert int(f.sum()) == 105
    assert set(v.prediccion) <= {'B', 'IR', 'OR', 'N', 'sin_diagnostico'}
    errores = int((~v.loc[f, 'tipo_correcto']).sum())
    resumen = dict(total_objetivo=105, tipo_incorrecto=errores, tipo_correcto=105-errores,
        meta_tipos=5, meta_tipos_alcanzada=errores <= 5, total_archivos=133,
        tipo_incorrecto_133=int((~v.tipo_correcto).sum()),
        normales_correctos=int(v.loc[v.real.eq('N'), 'tipo_correcto'].sum()),
        sin_diagnostico=int(v.prediccion.eq('sin_diagnostico').sum()),
        entrenamiento_fallas=52, grados_evaluados=False, validacion_externa=False)
    (salida / 'resumen_tipos.json').write_text(json.dumps(resumen, indent=2), encoding='utf-8')
    v[f & ~v.tipo_correcto].to_csv(salida / 'errores_tipo_105.csv', index=False)
    for nombre, datos in [('105', v[f]), ('133', v)]:
        matriz = pd.crosstab(datos.real, datos.prediccion).reindex(
            index=['B', 'IR', 'OR', 'N'], columns=['B', 'IR', 'OR', 'N', 'sin_diagnostico'], fill_value=0)
        assert int(matriz.to_numpy().sum()) == len(datos)
        matriz.to_csv(salida / f'matriz_tipo_{nombre}.csv')
    print(json.dumps(dict(variante=salida.name, **resumen)), flush=True)
    return resumen


def main():
    original = pd.read_csv(FUENTE / 'muestra.csv')
    columnas = ['archivo', 'sensor', 'fs', 'suficiente', 'sha256_senal', 'registro', 'catalogo',
        'real', 'grupo', 'grupo_auditoria', 'detector_auditoria', 'paper']
    d = original[columnas].copy()
    prep = json.loads((FUENTE / 'protocolo_preparacion.json').read_text(encoding='utf-8'))
    assert prep['codigo'] == firma(AQUI / 'rasgos_acp_denso.py')
    assert prep['nucleo'] == firma(AQUI / 'acp_denso.py')
    with np.load(AQUI / 'resultados_temporales_acp/rasgos.npz', allow_pickle=False) as z:
        crudos = z['x'][:, :, :2021]
    with np.load(FUENTE / 'rasgos.npz', allow_pickle=False) as z:
        densos = z['x']
    orden = pd.read_csv(AQUI / 'resultados_temporales_acp/muestra.csv')
    assert orden.archivo.equals(d.archivo) and orden.sha256_senal.equals(d.sha256_senal)
    x = np.concatenate([crudos, np.repeat(densos[:, None, :], 12, axis=1)], axis=2)
    assert x.shape == (133, 12, 3011) and np.isfinite(x[d.suficiente]).all()
    meta = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    dev = d.archivo.isin(meta.loc[meta.real.ne('N'), 'archivo']).to_numpy()
    y, g = d.real.to_numpy(), d.grupo_auditoria.to_numpy()
    assert dev.sum() == 52 and not any('@' in q for q in g)
    alternativas = {q['archivo']: q['alternativas_seleccion'] for q in json.loads(
        (AQUI / 'resultados_correccion_d_persistente/evidencia.json').read_text(encoding='utf-8'))}
    protocolo = dict(codigo=firma(Path(__file__)), preparador=firma(AQUI / 'rasgos_acp_denso.py'),
        nucleo=firma(AQUI / 'acp_denso.py'), rasgos=firma(FUENTE / 'rasgos.npz'),
        crudos=firma(AQUI / 'resultados_temporales_acp/rasgos.npz'),
        modelo=firma(AQUI / 'modelo_por_dominio.py'), controlador=firma(AQUI / 'evaluar_retirada_control_pd.py'),
        receta='LDA shrinkage auto, doce vistas, crudo2021+coherencia densa990',
        entrenamiento_fallas=52, exclusiones='tipo+tamaño completos, todas las posiciones OR juntas',
        variantes='dos recetas completas: con controles históricos y sin forzar tipo',
        comparacion='solo clasificación de tipo; no calcular grados ni usar la segunda mitad para puntuar bandas',
        objetivo='como máximo 5 tipos incorrectos entre los mismos 105 registros con paper',
        exploratorio=True, validacion_externa=False)
    rp = FUENTE / 'protocolo_evaluacion.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    modelos, folds, bases, controlados = {}, [], [], []
    with threadpool_limits(limits=1):
        for i, r in enumerate(d.itertuples()):
            if not r.suficiente or r.detector_auditoria == 'N':
                tb = t = 'N' if r.suficiente else 'sin_diagnostico'
            else:
                if r.grupo_auditoria not in modelos:
                    tr = dev & (g != r.grupo_auditoria)
                    assert r.grupo_auditoria not in set(g[tr]) and r.archivo not in set(d.loc[tr, 'archivo'])
                    modelos[r.grupo_auditoria] = crear().fit(x[tr].reshape(-1, 3011), np.repeat(y[tr], 12))
                    folds.append(dict(grupo_exterior=r.grupo_auditoria, train=d.loc[tr, 'archivo'].tolist()))
                    print('LDA ACP denso, fuera:', r.grupo_auditoria, flush=True)
                tb = predecir_tipo(modelos[r.grupo_auditoria], x[i])
                t = controlar(tb, r.sensor, alternativas[r.archivo], usar_d=True, usar_pd=True)
            bases.append(tb)
            controlados.append(t)
    pendientes = []
    for nombre, pred in [('resultados_acp_denso_sin_control', bases), ('resultados_acp_denso_control', controlados)]:
        salida = AQUI / nombre
        salida.mkdir(exist_ok=True)
        v = d.copy()
        v['prediccion'], v['tipo_base_denso'] = pred, bases
        v['tipo_correcto'] = v.real.eq(v.prediccion)
        v.to_csv(salida / 'tipos_133.csv', index=False)
        (salida / 'protocolo.json').write_text(json.dumps(dict(protocolo, variante=nombre), ensure_ascii=False, indent=2), encoding='utf-8')
        (salida / 'pliegues.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        pendientes.append((v, salida))
    # Ambas clasificaciones quedan fijadas antes de comparar sus etiquetas.
    for v, salida in pendientes:
        guardar_resumen_tipos(v, salida)


if __name__ == '__main__':
    main()
