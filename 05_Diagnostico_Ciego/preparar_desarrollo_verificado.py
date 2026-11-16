"""Nueva base de desarrollo tras corregir canal y RPM de registros múltiples."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from modelo_ciego import F
from modelo_espectral import extraer, VERSION_DATOS
from ensayar_tipo_seleccion import canonizar_prefijo
from ranking_catalogo import grupo_condicion

AQUI = Path(__file__).parent
SALIDA = AQUI/'resultados_desarrollo_verificado_v2'


if __name__ == '__main__':
    SALIDA.mkdir(exist_ok=True)
    protocolo = dict(version_datos=VERSION_DATOS, segundos_clasificacion=2,
        fs_features=12000, L=4, alcance='52 fallas 12k de desarrollo y dos normales',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        extractor_sha256=hashlib.sha256((AQUI/'modelo_espectral.py').read_bytes()).hexdigest(),
        lector_sha256=hashlib.sha256((AQUI.parent/'01_MFD_Kurtogram_Tarea1/Fuentes.py').read_bytes()).hexdigest(),
        fuente_indice_sha256=hashlib.sha256((AQUI.parent/'data2/CWRU/indice_canal_verificado.csv').read_bytes()).hexdigest())
    ruta_protocolo = SALIDA/'protocolo.json'
    if ruta_protocolo.exists():
        assert json.loads(ruta_protocolo.read_text(encoding='utf-8')) == protocolo
    else:
        ruta_protocolo.write_text(json.dumps(protocolo, indent=2, ensure_ascii=False), encoding='utf-8')
    split = pd.read_csv(AQUI/'particion_desarrollo.csv')
    dev = set(split.loc[split.particion.eq('desarrollo'),'archivo'])
    datos = AQUI.parent/'data2/CWRU'
    idx = F.leer_indice(str(datos))
    idx = idx[idx.archivo.isin(dev)].sort_values('archivo')
    assert len(idx) == 54 and set(idx.archivo) == dev
    filas = []; rasgos = []
    for i, r in enumerate(idx.itertuples()):
        ruta = datos/r.archivo
        mat = loadmat(ruta)
        clave = F.clave_canal(mat, r.sensor_falla, int(r.registro))
        x = np.ravel(mat[clave])
        rpm = F.rpm_registro(mat, int(r.registro), nominal=float(r.rpm_nominal))
        catalogo = {k:float(v*rpm/60) for k,v in F.MULTIPLICADORES[r.sensor_falla].items()}
        prefijo = canonizar_prefijo(x, int(r.fs))
        rasgos.append(extraer(prefijo, 12000, catalogo, L=4)['arbol'])
        huella = hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        filas.append(dict(archivo=r.archivo, registro=int(r.registro), clave_canal=clave,
            real=r.zona, grupo=('normal:'+huella) if r.zona=='N' else grupo_condicion(r.archivo),
            version_datos=VERSION_DATOS, fs_original=int(r.fs), fs_features=12000,
            rpm=rpm, catalogo=json.dumps(catalogo,sort_keys=True),
            sha256_mat=hashlib.sha256(ruta.read_bytes()).hexdigest(), sha256_senal=huella,
            sha256_prefijo=hashlib.sha256(prefijo.astype('<f8').tobytes()).hexdigest()))
        print(f'Datos corregidos {i+1}/54: {clave}', flush=True)
    meta = pd.DataFrame(filas); x = np.asarray(rasgos)
    assert meta.sha256_senal.is_unique and x.shape == (54,1953) and np.isfinite(x).all()
    meta.to_csv(SALIDA/'muestra.csv', index=False)
    np.savez_compressed(SALIDA/'rasgos.npz', x=x)
    print('Nueva base preparada. Las predicciones y métricas históricas no se reutilizan.', flush=True)
