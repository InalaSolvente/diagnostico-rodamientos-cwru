"""Dos bandas fijadas en selección; cada una debe cumplir por sí sola el juez.

Conserva el umbral original, corregido por TODO el banco de bandas, familias,
armónicas y transformaciones. No suma picos ni armonías de ramas diferentes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from bandas_estables import (validar_senal, envolvente, transformar, espectro,
    margenes, verificar, TRANSFORMACIONES)
from alinear_bandas import geometria
from criterio_usuario import evaluar, ORDEN
from modelo_ciego import F
from modelo_por_dominio import firma

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_doble_banda_fijada'


def intervalo(nodo, fs):
    l,m = nodo; ancho = fs/2**(l+1)
    return m*ancho,(m+1)*ancho


def disjuntas(a, b, fs):
    ai,af = intervalo(a,fs); bi,bf = intervalo(b,fs)
    return min(af,bf) <= max(ai,bi)


def seleccionar_dos(x, fs, catalogo, tipo):
    x = validar_senal(x,fs,catalogo)
    nodos,_ = geometria(fs)
    familias = {'IR':[('BPFI',catalogo['BPFI'])], 'OR':[('BPFO',catalogo['BPFO'])],
        'B':[('BSF',catalogo['BSF']),('2BSF',2*catalogo['BSF'])]}
    if tipo not in familias:
        raise ValueError('Se requiere una clase de falla predicha')
    p = .01/(len(nodos)*5*6*len(TRANSFORMACIONES))
    candidatos = []
    for nodo in nodos:
        a = envolvente(x,fs,nodo)
        for transformacion in TRANSFORMACIONES:
            b = transformar(a,transformacion); f,s = espectro(b,fs)
            segmentos = [espectro(v,fs) for v in np.array_split(b,3)]
            for familia,base in familias[tipo]:
                q = evaluar(f,s,base,H=6,p=p)
                partes = [evaluar(fp,sp,base,H=6,p=p) for fp,sp in segmentos]
                md = [margenes(v)[0] for v in partes]; mp = [margenes(v)[1] for v in partes]
                robusto = float(np.quantile(md if q['grado'] == 'D' else mp,.25))
                clave = (ORDEN[q['grado']],robusto,len(q['identificadas']),margenes(q)[0])
                seleccion = dict(grado=q['grado'],nodo=list(nodo),familia=familia,
                    frecuencia_base=float(base),transformacion=transformacion,p_prueba=p,
                    margen_d_segmentos=md,margen_p_segmentos=mp,evidencia=q)
                candidatos.append((clave,seleccion))
    candidatos.sort(key=lambda v:v[0],reverse=True)
    primaria = candidatos[0][1]
    segunda = next((s for _,s in candidatos if disjuntas(primaria['nodo'],s['nodo'],fs)),None)
    return [primaria]+([segunda] if segunda is not None else [])


def verificar_dos(x, fs, catalogo, selecciones):
    if not 1 <= len(selecciones) <= 2:
        raise ValueError('Se requieren una o dos selecciones fijadas previamente')
    if len(selecciones) == 2 and not disjuntas(selecciones[0]['nodo'],selecciones[1]['nodo'],fs):
        raise ValueError('Las dos bandas deben ser disjuntas')
    ramas = [dict(seleccion=s,verificacion=verificar(x,fs,catalogo,s)) for s in selecciones]
    return dict(grado=max([r['verificacion']['grado'] for r in ramas],key=ORDEN.get),ramas=ramas)


def comprobar():
    assert disjuntas((4,0),(4,1),12000)
    assert not disjuntas((3,0),(4,1),12000)
    assert not disjuntas((0,0),(4,15),12000)
    assert intervalo((4,1),12000) == intervalo((6,1),48000)


def main():
    comprobar(); SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_control_ordenes_giro'
    base = pd.read_csv(fuente/'comparacion_133.csv').set_index('archivo',drop=False)
    origen = AQUI/'resultados_correccion_d_persistente/evidencia.json'
    anteriores = {q['archivo']:q for q in json.loads(origen.read_text(encoding='utf-8'))}
    protocolo = dict(version='dos_bandas_disjuntas_fijadas_v1',
        seleccion='primaria idéntica al selector histórico; secundaria mejor candidata disjunta, primera mitad',
        verificacion='evaluar ambas bandas fijadas; cada rama cumple por sí sola ND/PD/D en selección y verificación',
        decision='máximo de grados de las dos ramas; nunca combinar armónicas entre ramas',
        multiplicidad='mismo p original corregido por todo el banco; no se relaja por usar solo dos bandas',
        tipo='control de órdenes de giro sobre clasificador con condición excluida',
        limitacion='ensayo exploratorio en dataset ya inspeccionado; p nominal no calibrado empíricamente',
        codigo=firma(Path(__file__)),base=firma(fuente/'comparacion_133.csv'),
        bandas=firma(AQUI/'bandas_estables.py'),criterio=firma(AQUI/'criterio_usuario.py'),
        elecciones_previas=firma(origen))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    filas = []; evidencia = []
    idx = F.leer_indice(str(AQUI.parent/'data2/CWRU'))
    for i,r in enumerate(idx.itertuples()):
        b = base.loc[r.archivo]; selecciones = []; v = dict(grado=b.grado,ramas=[])
        if b.suficiente and b.prediccion != 'N':
            mat = loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
            x = np.ravel(mat[F.clave_canal(mat,r.sensor_falla,int(r.registro))])
            assert hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == b.sha256_senal
            rpm = F.rpm_registro(mat,int(r.registro),nominal=float(r.rpm_nominal))
            c = {k:float(a*rpm/60) for k,a in F.MULTIPLICADORES[r.sensor_falla].items()}
            mitad = len(x)//2
            selecciones = seleccionar_dos(x[:mitad],int(r.fs),c,b.prediccion)
            assert selecciones[0] == anteriores[r.archivo]['alternativas_seleccion'][b.prediccion]
            v = verificar_dos(x[mitad:],int(r.fs),c,selecciones)
            assert v['ramas'][0]['verificacion']['grado'] == b.grado
            assert ORDEN[v['grado']] >= ORDEN[b.grado]
        filas.append(dict(b,grado=v['grado'],bandas=len(selecciones)))
        evidencia.append(dict(archivo=r.archivo,tipo=b.prediccion,selecciones=selecciones,**v))
        pd.DataFrame(filas).to_csv(SALIDA/'comparacion_parcial.csv',index=False)
        print(f'{i+1}/133 doble banda: {r.archivo} {b.prediccion} {b.grado}->{v["grado"]}',flush=True)
    d = pd.DataFrame(filas)
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    (SALIDA/'evidencia.json').write_text(json.dumps(evidencia,ensure_ascii=False,indent=2),encoding='utf-8')
    f = d[d.paper.notna()]; delta = f.grado.map(ORDEN)-f.paper.map(ORDEN); ok = f.tipo_correcto
    resumen = dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,desplegado=False)
    assert len(f) == 105 and sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__ == '__main__':
    main()
