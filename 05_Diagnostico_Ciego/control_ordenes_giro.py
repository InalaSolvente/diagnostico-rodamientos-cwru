"""No sobreescribe un tipo por evidencia indistinguible de órdenes de giro."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from modelo_ciego import F
from modelo_por_dominio import firma
from rasgos_fisicos import giro_catalogo
from correccion_d_persistente import elegir as elegir_d
from criterio_usuario import ORDEN

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_control_ordenes_giro'


def solo_ordenes_giro(seleccion, catalogo):
    giro = giro_catalogo(catalogo)
    familia = seleccion['familia']
    base = 2*catalogo['BSF'] if familia == '2BSF' else catalogo[familia]
    identificadas = seleccion['evidencia']['identificadas']
    # Misma tolerancia del juez; no se elimina ningún pico del espectro.
    ambiguas = [abs(h*base-round(h*base/giro)*giro) <= .02*h*base for h in identificadas]
    return bool(ambiguas and all(ambiguas))


def elegir(tipo_base, alternativas, catalogo):
    tipo,motivo = elegir_d(tipo_base,alternativas)
    if tipo != tipo_base and solo_ordenes_giro(alternativas[tipo],catalogo):
        return tipo_base,'sin_correccion_por_coincidencia_con_ordenes_de_giro'
    return tipo,motivo


def comprobar():
    s = dict(familia='BPFI',evidencia=dict(identificadas=[1,2,3]))
    assert not solo_ordenes_giro(s,F.MULTIPLICADORES['DE'])
    assert solo_ordenes_giro(s,F.MULTIPLICADORES['FE'])
    b = dict(familia='BSF',evidencia=dict(identificadas=[1,2]))
    assert solo_ordenes_giro(b,F.MULTIPLICADORES['FE'])
    assert not solo_ordenes_giro(b,F.MULTIPLICADORES['DE'])
    assert solo_ordenes_giro(b,{k:30*v for k,v in F.MULTIPLICADORES['FE'].items()})


def main():
    comprobar(); SALIDA.mkdir(exist_ok=True)
    fuente = AQUI/'resultados_correccion_d_persistente'
    base_dir = AQUI/'resultados_total_condicion_excluida'
    base = pd.read_csv(base_dir/'comparacion_133.csv').set_index('archivo',drop=False)
    anterior = pd.read_csv(fuente/'comparacion_133.csv').set_index('archivo',drop=False)
    evidencias = json.loads((fuente/'evidencia.json').read_text(encoding='utf-8'))
    protocolo = dict(version='D_persistente_sin_alias_giro_v1',
        regla='conservar el tipo base si todas las armonicas que justifican corregir tambien admiten ordenes de giro',
        tolerancia=.02, ajuste='ninguno; misma tolerancia que el juez',
        entradas='catalogo completo y evidencia de primera mitad; sin verificacion ni etiqueta verdadera',
        limitacion='regla motivada por errores inspeccionados; exploratoria, sin validacion externa',
        codigo=firma(Path(__file__)),fuente=firma(fuente/'evidencia.json'),base=firma(base_dir/'comparacion_133.csv'),
        control_giro=firma(AQUI/'rasgos_fisicos.py'),regla_d=firma(AQUI/'correccion_d_persistente.py'))
    rp = SALIDA/'protocolo.json'
    if rp.exists():
        assert json.loads(rp.read_text(encoding='utf-8')) == protocolo
    else:
        rp.write_text(json.dumps(protocolo,ensure_ascii=False,indent=2),encoding='utf-8')
    filas = []
    for fila in evidencias:
        b = base.loc[fila['archivo']]; a = anterior.loc[fila['archivo']]
        if not b.suficiente or b.prediccion == 'N':
            tipo = b.prediccion; motivo = 'normal_o_senal_insuficiente'
        else:
            # Solo razones del catalogo: escalar todas por RPM no cambia el control.
            tipo,motivo = elegir(b.prediccion,fila['alternativas_seleccion'],F.MULTIPLICADORES[b.sensor])
        assert tipo in (b.prediccion,a.prediccion)
        origen = b if tipo == b.prediccion else a
        filas.append(dict(b, prediccion=tipo,grado=origen.grado,tipo_correcto=tipo == b.real,motivo=motivo))
    d = pd.DataFrame(filas)
    d.to_csv(SALIDA/'comparacion_133.csv',index=False)
    f = d[d.paper.notna()]; delta = f.grado.map(ORDEN)-f.paper.map(ORDEN); ok = f.tipo_correcto
    resumen = dict(total=105,mejoran=int((ok & delta.gt(0)).sum()),iguales=int((ok & delta.eq(0)).sum()),
        empeoran=int((ok & delta.lt(0)).sum()),tipo_incorrecto=int((~ok).sum()),
        sin_retroceso=int((ok & delta.ge(0)).sum()),normales_correctos=int(d.loc[d.real.eq('N'),'tipo_correcto'].sum()),
        validacion_externa=False,desplegado=False)
    assert len(d) == 133 and len(f) == 105 and d.sha256_senal.is_unique
    assert sum(resumen[k] for k in ('mejoran','iguales','empeoran','tipo_incorrecto')) == 105
    (SALIDA/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen),flush=True)


if __name__ == '__main__':
    main()
