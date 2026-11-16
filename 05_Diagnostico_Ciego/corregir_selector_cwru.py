"""Migración auditable del selector de variables MAT, sin reemplazar señales."""
from pathlib import Path
import ast
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

RAIZ = Path(__file__).parent.parent


if __name__ == '__main__':
    datos = RAIZ/'data2/CWRU'
    indice = pd.read_csv(datos/'indice_cwru.csv')
    for archivo, registro in [('Normal/Normal_2.mat', 99), ('48k_DE/IR021_3.mat', 217)]:
        mat = loadmat(datos/archivo)
        assert f'X{registro:03d}_DE_time' in mat, 'No está el registro esperado en el MAT histórico'
        recuperado = loadmat(datos/f'fuente_verificada/{registro}.mat')
        assert np.array_equal(mat[f'X{registro:03d}_DE_time'], recuperado[f'X{registro:03d}_DE_time'])
        indice.loc[indice.archivo.eq(archivo), 'registro'] = registro
    indice['version_datos'] = 'selector_explicito_cwru_v1'
    indice.to_csv(datos/'indice_canal_verificado.csv', index=False)
    salida = RAIZ/'05_Diagnostico_Ciego/auditoria_selector_mat'
    salida.mkdir(exist_ok=True)
    respaldo = salida/'codigo_anterior'
    respaldo.mkdir(exist_ok=True)
    fuentes = RAIZ/'01_MFD_Kurtogram_Tarea1/Fuentes.py'
    cambios = []
    patron = re.compile(r'next\(k for k in (\w+) if k\.endswith\(f([\x22\x27])_\{(\w+)\.sensor_falla\}_time\2\)\)')
    for ruta in [fuentes, *sorted((RAIZ/'05_Diagnostico_Ciego').glob('*.py'))]:
        if ruta.resolve() == Path(__file__).resolve():
            continue
        original = ruta.read_text(encoding='utf-8')
        nuevo = patron.sub(lambda m: f'F.clave_canal({m[1]}, {m[3]}.sensor_falla, int({m[3]}.registro))', original)
        patron_rpm = re.compile(r'float\(np\.ravel\((\w+)\[(\w+)\[0\]\]\)\[0\]\) if \2 else float\((\w+)\.rpm_nominal\)')
        nuevo = patron_rpm.sub(lambda m: f'F.rpm_registro({m[1]}, int({m[3]}.registro), nominal=float({m[3]}.rpm_nominal))', nuevo)
        if ruta == fuentes:
            nuevo = nuevo.replace('F.clave_canal(', 'clave_canal(').replace('F.rpm_registro(', 'rpm_registro(')
            nuevo = nuevo.replace('clave = [k for k in d if k.endswith(f"_{sensor}_time")][0]',
                                  'clave = clave_canal(d, sensor, int(fila.registro))')
            anterior = 'return pd.read_csv(os.path.join(ruta_datos, "indice_cwru.csv"))'
            reemplazo = ('verificado = os.path.join(ruta_datos, "indice_canal_verificado.csv")\n'
                         '    return pd.read_csv(verificado if os.path.isfile(verificado) else os.path.join(ruta_datos, "indice_cwru.csv"))')
            if anterior in nuevo:
                nuevo = nuevo.replace(anterior, reemplazo, 1)
            if 'def clave_canal(' not in nuevo:
                nuevo += '''\n\ndef clave_canal(mat, sensor, registro=None):
    """Selecciona el ID declarado; rechaza MAT ambiguos sin ID explícito."""
    candidatas = [k for k in mat if k.endswith(f"_{sensor}_time")]
    if registro is not None:
        clave = f"X{int(registro):03d}_{sensor}_time"
        if clave not in mat:
            raise KeyError(f"Registro/canal esperado ausente: {clave}; disponibles: {candidatas}")
        return clave
    if len(candidatas) != 1:
        raise ValueError(f"MAT ambiguo: indique registro. Canales disponibles: {candidatas}")
    return candidatas[0]
'''
            for comilla in ('"', "'"):
                fragmento = f'next(k for k in mat if k.endswith(f{comilla}_{{sensor}}_time{comilla}))'
                nuevo = nuevo.replace(fragmento, 'clave_canal(mat, sensor, int(fila.registro))')
                fragmento = f'[k for k in mat if k.endswith(f{comilla}_{{sensor}}_time{comilla})][0]'
                nuevo = nuevo.replace(fragmento, 'clave_canal(mat, sensor, int(fila.registro))')
        if ruta.name == 'integridad_senales.py' and nuevo != original:
            nuevo = nuevo.replace('from scipy.io import loadmat', 'from scipy.io import loadmat\nfrom modelo_ciego import F', 1)
        if nuevo != original:
            ast.parse(nuevo)
            copia = respaldo/(('Fuentes__' if ruta == fuentes else '')+ruta.name)
            if not copia.exists():
                copia.write_text(original, encoding='utf-8')
            ruta.write_text(nuevo, encoding='utf-8')
            cambios.append(dict(archivo=str(ruta.relative_to(RAIZ)),
                sha256_antes=hashlib.sha256(original.encode()).hexdigest(),
                sha256_despues=hashlib.sha256(nuevo.encode()).hexdigest()))
    if cambios:
        (salida/'cambios.json').write_text(json.dumps(cambios, indent=2), encoding='utf-8')
    print(f'Selector explícito aplicado a {len(cambios)} archivos; código previo preservado.', flush=True)
    import sys
    sys.path.insert(0, str(RAIZ/'05_Diagnostico_Ciego'))
    from modelo_ciego import F
    from integridad_senales import huellas_senales, exigir_independencia
    actual = F.leer_indice(str(datos))
    huellas = huellas_senales(actual, datos)
    assert len(actual) == len(set(huellas.values())) == 133
    for archivo, registro in [('Normal/Normal_2.mat', 99), ('48k_DE/IR021_3.mat', 217)]:
        m = loadmat(datos/archivo)
        assert F.clave_canal(m, 'DE', registro) == f'X{registro:03d}_DE_time'
        try:
            F.clave_canal(m, 'DE')
        except ValueError:
            pass
        else:
            raise AssertionError('MAT ambiguo aceptado sin registro')
    for nombre in ['particion_desarrollo.csv', 'resultados_modelo_48k/particion.csv']:
        s = pd.read_csv(RAIZ/'05_Diagnostico_Ciego'/nombre)
        exigir_independencia(huellas, set(s.loc[s.particion.eq('desarrollo'),'archivo']),
                           set(s.loc[s.particion.eq('reserva'),'archivo']))
    (salida/'verificacion.json').write_text(json.dumps(dict(archivos=133, senales_unicas=133,
        ambiguedad_rechazada=True, particiones_sin_copias=True,
        indice='data2/CWRU/indice_canal_verificado.csv'), indent=2), encoding='utf-8')
    print('OK: 133 archivos, 133 señales únicas; ambas particiones sin copias; ambigüedad rechazada.', flush=True)
