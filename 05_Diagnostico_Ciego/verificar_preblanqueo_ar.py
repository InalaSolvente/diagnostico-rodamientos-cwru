"""Audita un ensayo AR fallido sin adoptarlo ni cambiar el criterio."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_por_dominio import firma
from preblanqueo_ar import comprobar,preblanquear,rasgos_blancos,FS
from doble_banda_fijada import disjuntas
from criterio_usuario import ORDEN

AQUI=Path(__file__).parent
SALIDA=AQUI/'resultados_preblanqueo_ar'


def main():
    comprobar()
    p=json.loads((SALIDA/'protocolo_evaluacion.json').read_text(encoding='utf-8'))
    for campo,nombre in [('codigo','evaluar_preblanqueo_ar.py'),('preblanqueo','preblanqueo_ar.py'),
        ('control','control_ordenes_giro.py'),('pd','ensayar_pd_persistente.py'),
        ('bandas','doble_banda_fijada.py'),('criterio','criterio_usuario.py')]:
        assert p[campo]==firma(AQUI/nombre)
    assert p['rasgos']==firma(SALIDA/'rasgos.npz') and p['vistas']==firma(SALIDA/'vistas.csv')
    m=pd.read_csv(SALIDA/'vistas.csv').set_index('vista_uid',drop=False)
    original=pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv')
    desarrollo=m.vista_uid.isin(original.vista_uid)
    assert len(m)==135 and m.primaria.sum()==133 and desarrollo.sum()==56
    for f in json.loads((SALIDA/'pliegues.json').read_text(encoding='utf-8')):
        esperado=set(m.loc[desarrollo & m.grupo.ne(f['grupo_excluido']),'vista_uid'])
        assert set(f['train'])==esperado and f['grupo_excluido'] not in f['grupos_train']
        if f['grupo_excluido'].startswith('OR'):
            assert not m.loc[f['train'],'grupo'].eq(f['grupo_excluido']).any()
    d=pd.read_csv(SALIDA/'comparacion_133.csv').set_index('archivo',drop=False)
    base=pd.read_csv(AQUI/'resultados_grupos_por_defecto/comparacion_133.csv').set_index('archivo',drop=False)
    assert len(d)==133 and d.index.equals(base.index) and d.sha256_senal.is_unique
    for campo in ('real','paper','sha256_senal','suficiente'):
        assert d[campo].equals(base[campo])
    assert d.paper.notna().sum()==105 and d.real.eq('N').sum()==4
    for q in json.loads((SALIDA/'evidencia.json').read_text(encoding='utf-8')):
        r=d.loc[q['archivo']];assert q['tipo']==r.prediccion and q['grado']==r.grado
        if not q['ramas']:
            assert r.prediccion in ('N','sin_diagnostico');continue
        ramas=q['ramas'];assert 1<=len(ramas)<=2
        if len(ramas)==2:assert disjuntas(ramas[0]['seleccion']['nodo'],ramas[1]['seleccion']['nodo'],FS)
        for a in ramas:
            s=a['seleccion'];v=a['verificacion']
            assert s['p_prueba']==.01/(31*5*6*2)
            assert v['grado']==min([s['grado'],v['evidencia']['grado']],key=ORDEN.get)
            if v['grado']=='D':
                for e in (s['evidencia'],v['evidencia']):
                    assert e['primera_identificada'] and len(e['identificadas'])>=2 and not e['hay_pico_previo_mayor']
        assert r.grado==max([a['verificacion']['grado'] for a in ramas],key=ORDEN.get)
    with np.load(SALIDA/'rasgos.npz',allow_pickle=False) as z:x=z['x']
    # Un registro real por frecuencia de muestreo, sin elegirlos por resultado.
    for fs in (12000,48000):
        i=int(np.flatnonzero(m.primaria.to_numpy() & m.fs_original.eq(fs).to_numpy())[0]);r=m.iloc[i]
        mat=loadmat(AQUI.parent/'data2/CWRU'/r.archivo)
        cruda=np.ravel(mat[F.clave_canal(mat,r.sensor,int(r.registro))])
        y,info=preblanquear(cruda,fs)
        assert info['coef']==json.loads(r.coef)
        assert np.array_equal(rasgos_blancos(y,json.loads(r.catalogo)),x[i])
        otra=cruda.copy();otra[len(otra)//2:]=np.random.default_rng(100).normal(size=len(otra)-len(otra)//2)
        alterada,otro=preblanquear(otra,fs)
        assert info['coef']==otro['coef'] and np.array_equal(y[:len(y)//2],alterada[:len(y)//2])
    f=d[d.paper.notna()];delta=f.grado.map(ORDEN)-f.paper.map(ORDEN);ok=f.tipo_correcto
    calculado=dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),sin_retroceso=int((ok & delta.ge(0)).sum()),
        normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()))
    resumen=json.loads((SALIDA/'resumen.json').read_text(encoding='utf-8'))
    assert all(resumen[k]==v for k,v in calculado.items())
    informe=['# Preblanqueo AR(32): ensayo descartado','',
        f'Resultado auditado: {calculado}. La referencia estricta sin AR conserva 79/105 y cuatro normales correctos.',
        'Se mantienen los 133 registros y las 105 referencias. Dos vistas FE adicionales se usan solo para entrenar el detector: no aumentan el número de casos.',
        'Se comprobó la exclusión conjunta de posiciones OR y de las vistas de cada normal. El filtro no recibe catálogo ni etiquetas; usa los primeros 2,5 segundos y continúa causalmente.',
        'Las pruebas sintéticas y dos registros reales confirmaron que reemplazar la segunda mitad no altera filtro ni selección. Se verificaron las reglas D por rama, sin fusionar armónicas.',
        'El filtro reduce empeoramientos de grado, pero incrementa errores de tipo y falla en un normal. No se adopta ni se reemplazan modelos anteriores.',
        'El objetivo activo es 100/105 sin retroceso y muchas más mejoras que empeoramientos. No se alcanzó. El ensayo sigue siendo interno sobre datos ya inspeccionados.',
        'Verificar: `python 05_Diagnostico_Ciego/verificar_preblanqueo_ar.py`.','']
    (SALIDA/'INFORME.md').write_text('\n'.join(informe),encoding='utf-8')
    (SALIDA/'verificacion.json').write_text(json.dumps(dict(casos=133,referencia=105,
        futuro_independiente=True,grupos_completos=True,resultado=calculado,adoptado=False),indent=2),encoding='utf-8')
    print('OK: exclusiones, 133 casos, filtro sin futuro, reglas y recuentos auditados; ensayo descartado.')
    print(json.dumps(calculado))


if __name__=='__main__':main()
