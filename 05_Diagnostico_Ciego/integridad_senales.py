"""Detecta copias de la señal seleccionada, aunque cambie el nombre del archivo."""
import hashlib
import numpy as np
from scipy.io import loadmat
from modelo_ciego import F


def huellas_senales(indice, datos):
    huellas = {}
    for r in indice.itertuples():
        mat = loadmat(datos / r.archivo)
        clave = F.clave_canal(mat, r.sensor_falla, int(r.registro))
        x = np.asarray(mat[clave], dtype='<f8').ravel()
        huellas[r.archivo] = hashlib.sha256(x.tobytes()).hexdigest()
    return huellas


def detectar_filtraciones(huellas, entrenamiento, prueba):
    faltantes = (set(entrenamiento) | set(prueba)) - set(huellas)
    if faltantes:
        raise ValueError(f'Faltan huellas de {len(faltantes)} señales')
    origenes = {}
    for archivo in entrenamiento:
        origenes.setdefault(huellas[archivo], []).append(archivo)
    return {a: sorted(origenes[huellas[a]]) for a in sorted(prueba)
            if huellas[a] in origenes}


def exigir_independencia(huellas, entrenamiento, prueba):
    conflictos = detectar_filtraciones(huellas, entrenamiento, prueba)
    if conflictos:
        raise ValueError('Partición inválida: señales idénticas entre entrenamiento y prueba: '
                         + repr(conflictos))


if __name__ == '__main__':
    h = {'entrenamiento.mat': 'abc', 'copia_otro_nombre.mat': 'abc', 'nuevo.mat': 'xyz'}
    exigir_independencia(h, {'entrenamiento.mat'}, {'nuevo.mat'})
    assert detectar_filtraciones(h, {'entrenamiento.mat'}, {'copia_otro_nombre.mat'}) == {
        'copia_otro_nombre.mat': ['entrenamiento.mat']}
    try:
        exigir_independencia(h, {'entrenamiento.mat'}, {'copia_otro_nombre.mat'})
    except ValueError:
        pass
    else:
        raise AssertionError('No se rechazó la copia de la señal')
    print('OK: copias con nombres distintos rechazadas antes de ajustar el modelo')
