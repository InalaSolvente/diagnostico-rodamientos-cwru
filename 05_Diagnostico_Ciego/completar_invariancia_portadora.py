"""Completa la puntuación interrumpida sin reajustar ni cambiar sus hipótesis."""
from pathlib import Path
import json
import pandas as pd
from modelo_por_dominio import firma
from invariancia_portadora import comprobar
from puntuar_tipos_fijos import evaluar

AQUI = Path(__file__).parent


def main():
    comprobar()
    salida = AQUI / 'resultados_invariancia_portadora'
    protocolo = json.loads((salida / 'protocolo.json').read_text(encoding='utf-8'))
    assert protocolo['codigo'] == firma(AQUI / 'invariancia_portadora.py')
    assert protocolo['rasgos_fuente'] == firma(AQUI / 'resultados_globales_arbol/rasgos.npz')
    assert protocolo['detector_fuente'] == firma(AQUI / 'resultados_grupos_por_defecto/comparacion_133.csv')
    muestra = pd.read_csv(AQUI / 'resultados_dominios_desarrollo/muestra.csv')
    permitidos = set(muestra.loc[muestra.real.ne('N'), 'archivo'])
    assert len(permitidos) == 52
    d = pd.read_csv(salida / 'tipos_133.csv')
    fuente_tipos = firma(salida / 'tipos_133.csv')
    grupos = d.set_index('archivo').grupo_auditoria.to_dict()
    auditoria = json.loads((salida / 'auditoria_anidada.json').read_text(encoding='utf-8'))
    for q in auditoria:
        assert set(q['train']) <= permitidos
        assert q['grupo_exterior'] not in {grupos[a] for a in q['train']}
        for interior in q['internos']:
            assert q['grupo_exterior'] != interior['grupo_interno_fuera']
            assert q['grupo_exterior'] not in interior['grupos_train']
    c = pd.read_csv(AQUI / 'resultados_plantillas_coherentes/tipos_133.csv').set_index('archivo').loc[d.archivo]
    assert c.sha256_senal.tolist() == d.sha256_senal.tolist()
    assert c.sensor.tolist() == d.sensor.tolist() and c.fs.tolist() == d.fs.tolist()
    pred = d.prediccion.copy()
    d['catalogo'] = c.catalogo.tolist()
    d.to_csv(salida / 'tipos_133.csv', index=False)
    resultado, resumen = evaluar(d, salida)
    assert resultado.prediccion.equals(pred)
    (salida / 'estado_reanudado.json').write_text(json.dumps(dict(
        codigo=firma(Path(__file__)), tipos_antes=fuente_tipos, tipos_sin_cambios=True,
        reajustes=0, evaluacion_completa=True, sin_retroceso=resumen['sin_retroceso']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
