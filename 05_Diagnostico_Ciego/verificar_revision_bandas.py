"""Verifica exclusiones anidadas, geometría desplazada y reproducción desde MAT."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from banco_desplazado import seleccionar_57, nodos_57, comprobar
from banco_63 import seleccionar_63
from doble_banda_fijada import intervalo, disjuntas, seleccionar_dos, verificar_dos
from puntuar_tipos_fijos import contar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent


def diccionarios(v):
    if isinstance(v, dict):
        yield v
        for a in v.values():
            yield from diccionarios(a)
    elif isinstance(v, list):
        for a in v:
            yield from diccionarios(a)


def main():
    comprobar()
    assert intervalo((4, .5), 12000) == (187.5, 562.5)
    assert len({intervalo(q, 12000) for q in nodos_57(12000)}) == 57
    base = pd.read_csv(AQUI / 'resultados_grupos_por_defecto/comparacion_133.csv')
    dev = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    permitidos = set(dev.loc[dev.real.ne('N'), 'archivo'])
    assert len(permitidos) == 52
    mapa = base.set_index('archivo').grupo_auditoria.to_dict()
    p = AQUI / 'resultados_selector_controles_anidado'
    protocolo = json.loads((p / 'protocolo.json').read_text(encoding='utf-8'))
    assert protocolo['codigo'] == firma(AQUI / 'evaluar_selector_controles_anidado.py')
    for q in json.loads((p / 'auditoria_anidada.json').read_text(encoding='utf-8')):
        esperado = {a for a in permitidos if mapa[a] != q['grupo_exterior']}
        assert set(q['train']) == esperado
        assert set(q['test']) == {a for a in mapa if mapa[a] == q['grupo_exterior']}
        assert len(q['scores']) == 9
        assert q['configuracion'] == max(q['scores'], key=lambda k: (
            q['scores'][k]['equilibrada'], q['scores'][k]['exactitud']))
        for interior in q['internos']:
            vi = {a for a in esperado if mapa[a] == interior['grupo_interior']}
            assert set(interior['validacion']) == vi
            assert set(interior['train']) == esperado - vi
    p = AQUI / 'resultados_banco_desplazado'
    protocolo = json.loads((p / 'protocolo.json').read_text(encoding='utf-8'))
    assert protocolo['codigo'] == firma(AQUI / 'evaluar_banco_desplazado.py')
    assert protocolo['selector'] == firma(AQUI / 'banco_desplazado.py')
    d = pd.read_csv(p / 'comparacion_133.csv')
    assert d.archivo.equals(base.archivo) and d.prediccion.equals(base.prediccion)
    assert d.sha256_senal.equals(base.sha256_senal)
    assert d.paper.fillna('').equals(base.paper.fillna(''))
    assert len(d) == 133 and d.paper.notna().sum() == 105
    resumen = json.loads((p / 'resumen.json').read_text(encoding='utf-8'))
    assert all(resumen[k] == v for k, v in contar(d).items())
    evidencia = {q['archivo']: q for q in json.loads((p / 'evidencia.json').read_text(encoding='utf-8'))}
    assert set(evidencia) == set(d.archivo)
    chequeos_d = 0
    for r in d.itertuples():
        seleccionadas = []
        for q in diccionarios(evidencia[r.archivo]):
            if 'frecuencia_base' in q and 'nodo' in q:
                assert tuple(q['nodo']) in nodos_57(int(r.fs))
                assert q['p_prueba'] == .01/(57*5*6*2)
                familias = {'IR': {'BPFI'}, 'OR': {'BPFO'}, 'B': {'BSF', '2BSF'}}
                assert q['familia'] in familias[r.prediccion]
                catalogo = json.loads(r.catalogo)
                f = 2*catalogo['BSF'] if q['familia'] == '2BSF' else catalogo[q['familia']]
                assert q['frecuencia_base'] == f
                seleccionadas.append(q)
            if q.get('grado') == 'D' and 'armonicas' in q:
                primero = next(a for a in q['armonicas'] if a['h'] == 1)
                assert primero['significativa'] and primero['potencia'] >= q['max_izquierda']
                assert sum(a['significativa'] for a in q['armonicas']) >= 2
                chequeos_d += 1
        assert len(seleccionadas) <= 2
        if len(seleccionadas) == 2:
            assert disjuntas(seleccionadas[0]['nodo'], seleccionadas[1]['nodo'], int(r.fs))
    assert chequeos_d > 0

    def cargar(r):
        mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
        x = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
        assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
        return x, json.loads(r.catalogo)

    reproducidos = []
    # Casos elegidos por orden de archivo y muestreo, no por desempeño.
    for fs in (12000, 48000):
        r = next(d[d.fs.eq(fs) & d.prediccion.isin(['B', 'IR', 'OR'])].itertuples())
        x, c = cargar(r)
        mitad = len(x)//2
        s = seleccionar_57(x[:mitad], fs, c, r.prediccion)
        cambiado = x.copy()
        cambiado[mitad:] = np.random.default_rng(943).normal(size=len(x)-mitad)
        assert seleccionar_57(cambiado[:mitad], fs, c, r.prediccion) == s
        v = verificar_dos(x[mitad:], fs, c, s)
        assert v == {k: evidencia[r.archivo][k] for k in ('grado', 'ramas')}
        reproducidos.append(r.archivo)
    # Recalcula desde señal cruda TODOS los casos que prueban la cota inferior a 100.
    cota = pd.read_csv(AQUI / 'auditoria_cota_verificada/cota_105_por_banco.csv')
    cuentas_cota = cota.groupby('banco').conserva.sum().to_dict()
    assert cuentas_cota[31] == cuentas_cota[63] == 97 and len(cota) == 315
    prueba_cota = []
    for archivo in sorted(set(cota.loc[~cota.conserva, 'archivo'])):
        r = next(d[d.archivo.eq(archivo)].itertuples())
        x, c = cargar(r)
        mitad = len(x)//2
        for banco, selector in [(31, seleccionar_dos), (63, seleccionar_63), (57, seleccionar_57)]:
            esperado = cota[cota.archivo.eq(archivo) & cota.banco.eq(banco)].iloc[0]
            if esperado.conserva:
                continue
            s = selector(x[:mitad], int(r.fs), c, r.real)
            v = verificar_dos(x[mitad:], int(r.fs), c, s)
            assert v['grado'] == esperado.grado_con_tipo_conocido
            assert ORDEN[v['grado']] < ORDEN[esperado.paper]
            prueba_cota.append(dict(archivo=archivo, banco=banco, grado=v['grado']))
    salida = dict(codigo=firma(Path(__file__)), grupos_exteriores_e_interiores_excluidos=True,
        mismo_entrenamiento_52=True, tipos_banco57_sin_cambios=True,
        geometria_y_multiplicidad_correctas=True, evidencias_D_comprobadas=chequeos_d,
        banco57_reproducido_y_futuro_invariante=reproducidos, cota_reproducida=prueba_cota)
    (p / 'verificacion.json').write_text(json.dumps(salida, indent=2), encoding='utf-8')
    print(json.dumps(dict(verificado=True, casos=133, referencia=105, chequeos_D=chequeos_d,
                         cotas=cuentas_cota, recalculos_cota=len(prueba_cota))), flush=True)


if __name__ == '__main__':
    main()
