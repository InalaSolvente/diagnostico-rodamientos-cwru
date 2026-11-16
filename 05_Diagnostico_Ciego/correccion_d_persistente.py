"""Corrige el tipo solo ante una unica familia D en toda la seleccion.

Examina IR/OR/B por igual, sin etiqueta verdadera ni referencia del paper. La
verificacion temporal nunca participa en elegir tipo, familia, banda o umbral.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from bandas_estables import seleccionar, verificar
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_correccion_d_persistente'
TIPOS = ('IR', 'OR', 'B')


def elegir(tipo_base, selecciones):
    if tipo_base == 'N':
        return 'N', 'detector_normal'
    if tipo_base not in TIPOS or set(selecciones) != set(TIPOS):
        raise ValueError('Se requieren tres hipotesis completas y un tipo base valido')
    persistentes = [t for t in TIPOS if selecciones[t]['grado'] == 'D'
                   and min(selecciones[t]['margen_d_segmentos']) >= 0]
    if len(persistentes) == 1:
        return persistentes[0], 'D_unico_en_tres_subsegmentos'
    return tipo_base, 'sin_correccion_por_ambiguedad_o_falta_de_D_persistente'


def comprobar():
    s = {t:dict(grado='PD', margen_d_segmentos=[-1.,-1.,-1.]) for t in TIPOS}
    assert elegir('IR', s)[0] == 'IR'
    s['OR'] = dict(grado='D',margen_d_segmentos=[.1,.2,.3])
    assert elegir('IR', s)[0] == 'OR'
    s['B'] = dict(grado='D',margen_d_segmentos=[.1,.1,.1])
    assert elegir('IR', s)[0] == 'IR'
    assert elegir('N', {})[0] == 'N'
    s['B']['margen_d_segmentos'][0] = -.01
    assert elegir('IR', s)[0] == 'OR'


def main():
    comprobar(); SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_total_condicion_excluida'
    base = pd.read_csv(fuente/'comparacion_133.csv').set_index('archivo',drop=False)
    previo = AQUI/'comparacion_con_sin_esperada_todos'
    config = json.loads((previo/'configuracion.json').read_text(encoding='utf-8'))
    for nombre in ('bandas_estables.py','criterio_usuario.py'):
        assert config['hashes'][nombre] == firma(AQUI/nombre)
    protocolo = dict(version='D_persistente_unico_v1',
        regla='corregir solo si una unica clase satisface D y los tres margenes D de seleccion son no negativos',
        alternativas=list(TIPOS), parametros='fijos; umbral cero significa cumplir el juez, no optimizado',
        entradas='tipo base con condicion excluida, senal y catalogo completo',
        temporal='todas las decisiones solo primera mitad; verificacion fija en segunda mitad',
        limitacion='experimento interno sobre dataset ya observado; no validacion externa',
        codigo=firma(Path(__file__)),base=firma(fuente/'comparacion_133.csv'),
        bandas=firma(AQUI/'bandas_estables.py'),criterio=firma(AQUI/'criterio_usuario.py'))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    cache = {}
    for fila in json.loads((previo/'evidencia.json').read_text(encoding='utf-8')):
        for escenario in ('sin_esperada','con_esperada'):
            q = fila[escenario]
            if q and 'seleccion' in q:
                cache[(fila['archivo'],q['tipo'])] = {k:q[k] for k in ('tipo','grado','seleccion','verificacion')}
    for fila in json.loads((fuente/'evidencia.json').read_text(encoding='utf-8')):
        if 'seleccion' in fila:
            cache[(fila['archivo'],fila['tipo'])] = {k:fila[k] for k in ('tipo','grado','seleccion','verificacion')}
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU'))
    filas = []; evidencias = []
    for i,r in enumerate(idx.itertuples()):
        b = base.loc[r.archivo]
        if not b.suficiente or b.prediccion == 'N':
            tipo = b.prediccion; grado = b.grado; motivo = 'normal_o_senal_insuficiente'
            alternativas = {}; seleccion = None; verificacion = None
        else:
            mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            senal = np.ravel(mat[F.clave_canal(mat,r.sensor_falla,int(r.registro))])
            assert hashlib.sha256(senal.astype('<f8').tobytes()).hexdigest() == b.sha256_senal
            rpm = F.rpm_registro(mat,int(r.registro),nominal=float(r.rpm_nominal))
            c = {k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
            mitad = len(senal)//2
            alternativas = {}
            for candidato in TIPOS:
                guardado = cache.get((r.archivo,candidato))
                alternativas[candidato] = guardado['seleccion'] if guardado else seleccionar(senal[:mitad],int(r.fs),c,candidato)
            tipo,motivo = elegir(str(b.prediccion),alternativas)
            seleccion = alternativas[tipo]
            # Solo ahora se accede al resultado reservado de la hipotesis fijada.
            guardado = cache.get((r.archivo,tipo))
            if guardado:
                assert guardado['seleccion'] == seleccion
                verificacion = guardado['verificacion']; grado = guardado['grado']
            else:
                verificacion = verificar(senal[mitad:],int(r.fs),c,seleccion)
                grado = verificacion['grado']
        filas.append(dict(archivo=r.archivo,sensor=r.sensor_falla,fs=int(r.fs),
            real=b.real,paper=b.paper,grupo=b.grupo,sha256_senal=b.sha256_senal,
            tipo_base=b.prediccion,prediccion=tipo,grado=grado,tipo_correcto=tipo == b.real,
            corregido=tipo != b.prediccion,motivo=motivo))
        evidencias.append(dict(archivo=r.archivo,tipo=tipo,grado=grado,
            alternativas_seleccion=alternativas,seleccion=seleccion,verificacion=verificacion))
        pd.DataFrame(filas).to_csv(SALIDA/'comparacion_parcial.csv',index=False)
        print(f'{i+1}/133: {r.archivo} {b.prediccion}->{tipo}/{grado}',flush=True)
    d = pd.DataFrame(filas)
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencias,ensure_ascii=False,indent=2),encoding='utf-8')
    f = d[d.paper.notna()]; ok = f.tipo_correcto
    delta = f.grado.map(ORDEN)-f.paper.map(ORDEN)
    resumen = dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),
        iguales=int((ok & delta.eq(0)).sum()),empeoran=int((ok & delta.lt(0)).sum()),
        tipo_incorrecto=int((~ok).sum()),sin_retroceso=int((ok & delta.ge(0)).sum()),
        tipos_corregidos=int(f.corregido.sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,desplegado=False)
    assert len(f) == 105 and sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__ == '__main__':
    main()
