"""Solo corrige PD si una misma familia gana en los tres subsegmentos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from modelo_por_dominio import firma
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_pd_consenso'
TIPOS = ('IR','OR','B')


def elegir(base, alternativas):
    if base not in TIPOS:
        return base
    if set(alternativas) != set(TIPOS):
        raise ValueError('Se requieren las tres familias')
    if not all(alternativas[t]['grado'] == 'PD' for t in TIPOS):
        return base
    m = np.asarray([alternativas[t]['margen_p_segmentos'] for t in TIPOS])
    if m.shape != (3,3) or not np.isfinite(m).all():
        raise ValueError('Tres margenes finitos por familia requeridos')
    ganadores = m.argmax(axis=0)
    if not np.all(ganadores == ganadores[0]):
        return base
    i = int(ganadores[0]); otros = np.delete(m,i,axis=0).max(axis=0)
    return TIPOS[i] if np.all(m[i] > 0) and np.all(m[i] > otros) else base


def comprobar():
    a = {t:dict(grado='PD',margen_p_segmentos=[.1,.2,.3]) for t in TIPOS}
    assert elegir('B',a) == 'B'
    a['IR']['margen_p_segmentos'] = [.4,.4,.4]
    assert elegir('B',a) == 'IR'
    a['OR']['margen_p_segmentos'] = [.1,.2,.5]
    assert elegir('B',a) == 'B'
    a['OR']['margen_p_segmentos'] = [.1,.2,.3]; a['B']['grado'] = 'D'
    assert elegir('B',a) == 'B'
    assert elegir('N',{}) == 'N'


def main():
    comprobar(); SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_globales_bancos'
    base = pd.read_csv(fuente/'comparacion_133.csv')
    anterior_dir = AQUI/'resultados_pd_doble'
    anterior = pd.read_csv(anterior_dir/'comparacion_133.csv').set_index('archivo')
    ep = AQUI/'resultados_correccion_d_persistente/evidencia.json'
    alternativas = {q['archivo']:q['alternativas_seleccion'] for q in json.loads(ep.read_text(encoding='utf-8'))}
    protocolo = dict(version='PD_consenso_tres_segmentos_v1',
        regla='solo si todas son PD: misma familia gana estrictamente en los tres subsegmentos, cada margen positivo',
        entradas='tipo global con control y evidencia de seleccion; sin verificacion ni referencia',
        parametros='regla fija sin umbrales ajustados ni parametros por tipo',
        reutilizacion='grado para mismo archivo, tipo y banco; solo despues de fijar la prediccion',
        limitacion='regla propuesta tras ensayos inspeccionados; exploratoria, no validacion externa',
        codigo=firma(Path(__file__)),base=firma(fuente/'comparacion_133.csv'),
        pd_previo=firma(anterior_dir/'comparacion_133.csv'),evidencia=firma(ep))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    filas = []
    for r in base.itertuples():
        t = elegir(r.prediccion,alternativas[r.archivo])
        a = anterior.loc[r.archivo]
        assert t in (r.prediccion,a.prediccion)
        # Ambos grados fueron calculados para exactamente estas decisiones.
        g = r if t == r.prediccion else a
        fila = r._asdict(); fila.pop('Index',None)
        fila.update(tipo_previo=r.prediccion,prediccion=t,tipo_correcto=t == r.real,
            grado31=g.grado31,grado63=g.grado63)
        filas.append(fila)
    d = pd.DataFrame(filas)
    assert len(d) == 133 and d.archivo.is_unique and d.sha256_senal.is_unique
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    resumen = []
    for banco,col in [(31,'grado31'),(63,'grado63')]:
        f = d[d.paper.notna()]; delta = f[col].map(ORDEN)-f.paper.map(ORDEN); ok = f.tipo_correcto
        q = dict(banco=banco,total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
            empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
            sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
            validacion_externa=False,desplegado=False)
        assert len(f) == 105 and sum(q[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 105
        resumen.append(q)
    pd.DataFrame(resumen).to_csv(SALIDA/'resumen.csv',index=False)
    print(pd.DataFrame(resumen).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
