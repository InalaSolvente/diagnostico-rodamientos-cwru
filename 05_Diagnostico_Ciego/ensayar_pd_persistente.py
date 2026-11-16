"""Si todas las familias son PD en selección, ordena su peor margen temporal.

Regla fija sin entrenamiento ni parámetros específicos para IR/OR/B. No se usa
ningún grado de verificación para elegir la familia.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from modelo_por_dominio import firma

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_pd_persistente'
TIPOS = ('IR','OR','B')


def elegir(base, alternativas):
    if base not in TIPOS:
        return base
    if set(alternativas) != set(TIPOS):
        raise ValueError('Se requieren las tres familias')
    if not all(alternativas[t]['grado'] == 'PD' for t in TIPOS):
        return base
    margenes = np.array([min(alternativas[t]['margen_p_segmentos']) for t in TIPOS])
    ganadores = np.flatnonzero(np.isclose(margenes,margenes.max(),rtol=1e-12,atol=1e-12))
    return TIPOS[int(ganadores[0])] if len(ganadores) == 1 and margenes.max() > 0 else base


def comprobar():
    a = {t:dict(grado='PD',margen_p_segmentos=[.1,.2,.3]) for t in TIPOS}
    assert elegir('B',a) == 'B'
    a['IR']['margen_p_segmentos'] = [.3,.4,.5]
    assert elegir('B',a) == 'IR'
    a['OR']['grado'] = 'D'
    assert elegir('B',a) == 'B'
    assert elegir('N',{}) == 'N'


def main():
    comprobar(); SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_globales_arbol/tipos_133.csv'
    d = pd.read_csv(fuente)
    ep = AQUI/'resultados_correccion_d_persistente/evidencia.json'
    evidencia = {q['archivo']:q['alternativas_seleccion'] for q in json.loads(ep.read_text(encoding='utf-8'))}
    protocolo = dict(version='PD_tres_familias_peor_margen_v1',
        regla='solo si todas son PD, elegir el mayor minimo de los tres margenes PD; positivo y sin empate',
        entradas='tipo base y evidencia de seleccion; sin verificacion ni etiqueta verdadera',
        parametros='sin ajuste; cero representa significancia en cada subsegmento',
        limitacion='ensayo sobre datos ya inspeccionados; sin validacion externa',
        codigo=firma(Path(__file__)),tipos=firma(fuente),evidencia=firma(ep))
    p = SALIDA/'protocolo.json'
    if p.exists():
        assert json.loads(p.read_text(encoding='utf-8')) == protocolo
    else:
        p.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    original = d.prediccion.copy()
    d['prediccion'] = [elegir(r.prediccion,evidencia[r.archivo]) for r in d.itertuples()]
    d['tipo_correcto'] = d.real.eq(d.prediccion)
    d['tipo_previo'] = original
    d.to_csv(SALIDA/'tipos_133.csv',index=False)
    dev = pd.read_csv(AQUI/'resultados_dominios_desarrollo/muestra.csv').archivo
    f = d[d.paper.notna()]; m = f[f.archivo.isin(dev)]
    resumen = dict(total=105,tipo_correcto=int(f.tipo_correcto.sum()),tipo_correcto_desarrollo=int(m.tipo_correcto.sum()),
        desarrollo=52,cambios_tipo=int(f.prediccion.ne(f.tipo_previo).sum()),calidad_pendiente=True,desplegado=False)
    assert len(d) == 133 and len(f) == 105 and len(m) == 52
    (SALIDA/'resumen_tipo.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__ == '__main__':
    main()
