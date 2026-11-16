"""Descontar filtraciones exactas sin alterar pesos ni salidas históricas."""
from pathlib import Path
import json
import pandas as pd
from integridad_senales import detectar_filtraciones


if __name__ == '__main__':
    aqui = Path(__file__).parent
    salida = aqui/'auditoria_modelos_actuales'
    inv = pd.read_csv(salida/'inventario.csv')
    huellas = inv.set_index('archivo').sha256_senal.to_dict()
    d = pd.read_csv(aqui/'resultados_modelos_todos/consolidado_133.csv')
    d['evaluable_sin_copia'] = d.evaluable.eq(True)
    d['motivo_exclusion'] = ''
    d.loc[~d.evaluable.eq(True), 'motivo_exclusion'] = 'normal usado para entrenamiento'
    conflictos = []
    for nombre, ruta in [('12k', aqui/'particion_desarrollo.csv'),
                         ('48k', aqui/'resultados_modelo_48k/particion.csv')]:
        split = pd.read_csv(ruta)
        entrenamiento = set(split.loc[split.particion.eq('desarrollo'), 'archivo'])
        prueba = set(split.loc[split.particion.eq('reserva'), 'archivo'])
        for archivo, copias in detectar_filtraciones(huellas, entrenamiento, prueba).items():
            conflictos.append(dict(modelo=nombre, archivo_prueba=archivo,
                                   copias_entrenamiento=copias))
            # Solo se descuenta del consolidado si fue evaluado con ese modelo.
            mask = d.modelo.eq(nombre) & d.archivo.eq(archivo)
            d.loc[mask, 'evaluable_sin_copia'] = False
            d.loc[mask, 'motivo_exclusion'] = 'señal idéntica en entrenamiento: '+', '.join(copias)
    validos = d[d.evaluable_sin_copia]
    assert validos.archivo.map(huellas).is_unique
    assert len(validos) == 129 and int(validos.tipo_correcto.sum()) == 119
    d.to_csv(salida/'consolidado_auditado_133.csv', index=False)
    filas = []
    for fs, g in validos.groupby('fs'):
        filas.append(dict(fs=fs, evaluados=len(g), correctos=int(g.tipo_correcto.sum()),
                          porcentaje=100*g.tipo_correcto.mean()))
    pd.DataFrame(filas).to_csv(salida/'resumen_sin_copias.csv', index=False)
    (salida/'filtraciones.json').write_text(json.dumps(conflictos, indent=2, ensure_ascii=False), encoding='utf-8')
    print(pd.DataFrame(filas).to_string(index=False))
    print(f'Total descriptivo sin copias: {int(validos.tipo_correcto.sum())}/{len(validos)}. Sigue sin ser validación externa.')
