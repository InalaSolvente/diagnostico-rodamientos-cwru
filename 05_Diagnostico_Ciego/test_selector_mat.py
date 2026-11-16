"""Regresión de registros múltiples, RPM coherente y fuente oficial contrastada."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from modelo_ciego import F
from integridad_senales import huellas_senales


if __name__ == '__main__':
    p = Path(__file__).parent.parent/'data2/CWRU'
    idx = F.leer_indice(str(p))
    for archivo, registro in [('Normal/Normal_2.mat', 99), ('48k_DE/IR021_3.mat', 217)]:
        fila = idx.set_index('archivo').loc[archivo]
        assert int(fila.registro) == registro
        mat = loadmat(p/archivo)
        clave = F.clave_canal(mat, 'DE', registro)
        original = loadmat(p/f'fuente_verificada/{registro}.mat')
        assert np.array_equal(mat[clave], original[clave])
        central = F.cargar_registro(registro, ruta_datos=str(p), archivo=archivo)
        assert np.array_equal(central['x'], np.ravel(original[clave]))
        try:
            F.clave_canal(mat, 'DE')
        except ValueError:
            pass
        else:
            raise AssertionError('Se aceptó un archivo ambiguo sin registro')
        rpm = F.rpm_registro(mat, registro, nominal=float(fila.rpm_nominal))
        assert central['rpm'] == rpm
        print(f'{archivo}: {clave}; RPM del registro={rpm}')
    assert F.rpm_registro({'X215RPM':np.array([1752.]),'X217RPM':np.array([1728.])},217)==1728.
    assert F.rpm_registro({'X215RPM':np.array([1752.])},217,nominal=1730.)==1730.
    try:
        F.cargar_registro(217, ruta_datos=str(p))
    except ValueError:
        pass
    else:
        raise AssertionError('Se aceptó un ID interno repetido entre archivos sin desambiguar')
    try:
        F.rpm_registro({'X215RPM':np.array([1752.])},217)
    except ValueError:
        pass
    else:
        raise AssertionError('Se usó la RPM de otro registro')
    huellas = huellas_senales(idx, p)
    assert len(huellas) == len(set(huellas.values())) == 133
    print('OK: 133 señales únicas; registro explícito y RPM del mismo registro; sin modificar las señales históricas')
