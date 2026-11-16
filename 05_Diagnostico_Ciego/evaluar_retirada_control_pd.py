"""Ablación uniforme de controles de tipo; conserva entrenamiento y bandas."""
from pathlib import Path
import json
import pandas as pd
from modelo_ciego import F
from modelo_por_dominio import firma
from control_ordenes_giro import elegir as elegir_d
from ensayar_pd_persistente import elegir as elegir_pd
from puntuar_tipos_fijos import evaluar

AQUI = Path(__file__).parent


def controlar(tipo_base, sensor, alternativas, *, usar_d, usar_pd):
    """Solo recibe hipótesis y evidencia ciega del segmento de selección."""
    if tipo_base in ('N', 'sin_diagnostico'):
        return tipo_base
    tipo = elegir_d(tipo_base, alternativas, F.MULTIPLICADORES[sensor])[0] if usar_d else tipo_base
    return elegir_pd(tipo, alternativas) if usar_pd else tipo


def main():
    fuente = AQUI / 'resultados_grupos_por_defecto/tipos_133.csv'
    evidencia = AQUI / 'resultados_correccion_d_persistente/evidencia.json'
    d = pd.read_csv(fuente)
    catalogos = AQUI / 'resultados_plantillas_coherentes/tipos_133.csv'
    c = pd.read_csv(catalogos).set_index('archivo').loc[d.archivo]
    assert c.sha256_senal.tolist() == d.sha256_senal.tolist()
    assert c.sensor.tolist() == d.sensor.tolist() and c.fs.tolist() == d.fs.tolist()
    d['catalogo'] = c.catalogo.tolist()
    alternativas = {q['archivo']: q['alternativas_seleccion']
                    for q in json.loads(evidencia.read_text(encoding='utf-8'))}
    assert len(d) == 133 and d.archivo.is_unique
    # Reproduce exactamente el controlador auditado antes de retirarle reglas.
    reproducido = [controlar(r.tipo_base_auditoria, r.sensor, alternativas.get(r.archivo),
                            usar_d=True, usar_pd=True) for r in d.itertuples()]
    assert reproducido == d.prediccion.tolist()
    pendientes = []
    for nombre, usar_d in [('resultados_sin_control_PD', True), ('resultados_sin_forzar_tipo', False)]:
        salida = AQUI / nombre
        salida.mkdir(exist_ok=True)
        variante = d.copy()
        variante['prediccion'] = [controlar(r.tipo_base_auditoria, r.sensor,
                                           alternativas.get(r.archivo), usar_d=usar_d,
                                           usar_pd=False) for r in d.itertuples()]
        variante['tipo_correcto'] = variante.real.eq(variante.prediccion)
        variante.to_csv(salida / 'tipos_133.csv', index=False)
        protocolo = dict(codigo=firma(Path(__file__)), fuente=firma(fuente),
            catalogos=firma(catalogos),
            evidencia_seleccion=firma(evidencia), control_d=firma(AQUI / 'control_ordenes_giro.py'),
            control_pd=firma(AQUI / 'ensayar_pd_persistente.py'),
            puntuador=firma(AQUI / 'puntuar_tipos_fijos.py'), usar_d=usar_d, usar_pd=False,
            entrenamiento='mismos 52 registros con falla; grupos tipo+tamaño completos excluidos',
            tipos_congelados_antes_de_verificacion=True, cambios_por_archivo=False,
            seleccion_exploratoria=True, validacion_externa=False)
        (salida / 'protocolo.json').write_text(json.dumps(protocolo, ensure_ascii=False, indent=2), encoding='utf-8')
        pendientes.append((variante, salida))
    # Ambos vectores quedan congelados antes de abrir resultados del segmento futuro.
    for variante, salida in pendientes:
        evaluar(variante, salida)


if __name__ == '__main__':
    main()
