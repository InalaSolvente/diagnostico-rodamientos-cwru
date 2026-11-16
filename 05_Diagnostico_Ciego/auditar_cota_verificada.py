"""Cota del selector con tipo conocido: auditoría etiquetada, nunca predictor."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from doble_banda_fijada import seleccionar_dos, verificar_dos
from banco_63 import seleccionar_63
from banco_desplazado import seleccionar_57
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI / 'auditoria_cota_verificada'


def main():
    SALIDA.mkdir(exist_ok=True)
    fuente = AQUI / 'resultados_plantillas_coherentes/tipos_133.csv'
    d = pd.read_csv(fuente)
    d = d[d.paper.notna()].copy()
    assert len(d) == 105 and d.suficiente.all() and d.archivo.is_unique
    indice = d.set_index('archivo', drop=False)
    cache, fuentes = {}, []
    for carpeta, bancos, anidado in [
        ('resultados_globales_bancos', [31, 63], True), ('resultados_pd_doble', [31, 63], True),
        ('resultados_grupos_por_defecto', [31], False), ('resultados_plantillas_coherentes', [31], False),
        ('resultados_sin_control_PD', [31], False), ('resultados_sin_forzar_tipo', [31], False),
        ('resultados_invariancia_portadora', [31], False), ('resultados_banco_desplazado', [57], False)]:
        p = AQUI / carpeta
        datos = pd.read_csv(p / 'comparacion_133.csv').set_index('archivo')
        f = p / 'evidencia.json'
        fuentes.append(dict(carpeta=carpeta, comparacion=firma(p / 'comparacion_133.csv'), evidencia=firma(f)))
        for q in json.loads(f.read_text(encoding='utf-8')):
            if q['archivo'] not in set(d.archivo) or q['tipo'] not in ('IR', 'OR', 'B'):
                continue
            r = indice.loc[q['archivo']]
            assert datos.loc[r.archivo, 'sha256_senal'] == r.sha256_senal
            for banco in bancos:
                v = q[f'banco{banco}'] if anidado else {k: q[k] for k in ('grado', 'ramas')}
                key = (r.archivo, q['tipo'], banco)
                if key in cache:
                    assert cache[key]['grado'] == v['grado'], key
                cache[key] = v
    protocolo = dict(codigo=firma(Path(__file__)), fuente=firma(fuente), caches=fuentes,
        bandas31=firma(AQUI / 'doble_banda_fijada.py'), bandas63=firma(AQUI / 'banco_63.py'),
        bandas57=firma(AQUI / 'banco_desplazado.py'),
        criterio=firma(AQUI / 'criterio_usuario.py'),
        etiquetas='tipo verdadero consultado exclusivamente para esta cota; no se entrenan modelos',
        alcance='selector fijo, una o dos bandas fijadas antes de verificar, sin control de tipo ni detector',
        rendimiento_de_modelo=False, cambios_en_predictores=False)
    (SALIDA / 'protocolo.json').write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
    filas, evidencia = [], []
    for r in d.itertuples():
        x = None
        for banco, selector in [(31, seleccionar_dos), (63, seleccionar_63), (57, seleccionar_57)]:
            key = (r.archivo, r.real, banco)
            if key in cache:
                v = cache[key]
            else:
                if x is None:
                    mat = loadmat(AQUI.parent / 'data2/CWRU' / r.archivo)
                    x = np.ravel(mat[F.clave_canal(mat, r.sensor, int(r.registro))])
                    assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == r.sha256_senal
                mitad = len(x) // 2
                catalogo = json.loads(r.catalogo)
                # La etiqueta decide solo la familia que se audita; nunca una frecuencia exacta.
                s = selector(x[:mitad], int(r.fs), catalogo, r.real)
                v = verificar_dos(x[mitad:], int(r.fs), catalogo, s)
                print('Cota, verificacion nueva:', banco, r.archivo, flush=True)
            filas.append(dict(archivo=r.archivo, banco=banco, paper=r.paper,
                familia_consultada_solo_auditoria=r.real, grado_con_tipo_conocido=v['grado'],
                conserva=ORDEN[v['grado']] >= ORDEN[r.paper],
                mejora=ORDEN[v['grado']] > ORDEN[r.paper], sha256_senal=r.sha256_senal))
            evidencia.append(dict(archivo=r.archivo, banco=banco, familia_solo_auditoria=r.real, **v))
    a = pd.DataFrame(filas)
    assert len(a) == 315 and not a.duplicated(['archivo', 'banco']).any()
    a.to_csv(SALIDA / 'cota_105_por_banco.csv', index=False)
    (SALIDA / 'evidencia.json').write_text(json.dumps(evidencia, ensure_ascii=False, indent=2), encoding='utf-8')
    tabla = a.groupby('banco').agg(total=('archivo', 'size'),
        cota_sin_retroceso=('conserva', 'sum'), mejoras_con_tipo_conocido=('mejora', 'sum')).reset_index()
    tabla.to_csv(SALIDA / 'cotas.csv', index=False)
    texto = '''# Cota con verificación y tipo conocido

Esto es una auditoría con etiquetas, no el rendimiento de un modelo ciego.
Se concede al método un clasificador y un detector perfectos, pero se conservan el catálogo
completo del fabricante, el selector de dos bandas, las transformaciones y el juez actuales.
La primera mitad selecciona las bandas; la segunda verifica esas mismas bandas. No se
escoge retrospectivamente otra banda ni se combina evidencia de ambas para fabricar armónicos.

'''
    texto += tabla.to_string(index=False) + '\n\n'
    texto += '''La cota es específica de cada selector fijo: un método distinto de selección, otro banco
o un preprocesado legítimo podrían cambiarla. No demuestra una imposibilidad universal.
Si una cota queda por debajo de 100, mejorar solo el clasificador no alcanza la meta con
ese selector. Los resultados no se incorporan a las predicciones ni a sus matrices.

Casos que retroceden aun concediendo el tipo verdadero:\n\n'''
    texto += a.loc[~a.conserva, ['archivo', 'banco', 'paper', 'grado_con_tipo_conocido']].to_string(index=False)
    (SALIDA / 'INFORME.md').write_text(texto + '\n', encoding='utf-8')
    print(tabla.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
