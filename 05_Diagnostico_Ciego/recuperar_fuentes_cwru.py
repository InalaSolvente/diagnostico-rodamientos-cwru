"""Recupera dos adquisiciones desde CWRU sin reemplazar las señales históricas."""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy.io import loadmat

AQUI = Path(__file__).parent
DATOS = AQUI.parent/'data2/CWRU'
CORRECCIONES = {
    'Normal/Normal_2.mat': dict(registro=99,
        pagina='https://engineering.case.edu/bearingdatacenter/normal-baseline-data',
        url='https://engineering.case.edu/sites/default/files/99.mat'),
    '48k_DE/IR021_3.mat': dict(registro=217,
        pagina='https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data',
        url='https://engineering.case.edu/sites/default/files/217.mat'),
}


if __name__ == '__main__':
    destino = DATOS/'fuente_verificada'
    destino.mkdir(exist_ok=True)
    indice = pd.read_csv(DATOS/'indice_cwru.csv')
    indice['archivo_original'] = indice.archivo
    manifest = []
    for original, fuente in CORRECCIONES.items():
        registro = fuente['registro']
        ruta = destino/f'{registro}.mat'
        if not ruta.exists():
            peticion = urllib.request.Request(fuente['url'], headers={'User-Agent': 'CWRU-research-data-verification/1.0'})
            with urllib.request.urlopen(peticion, timeout=45) as respuesta:
                datos = respuesta.read(30_000_001)
                url_final = respuesta.geturl()
            if len(datos) > 30_000_000:
                raise ValueError('Tamaño inesperado')
            temporal = destino/f'{registro}.descarga'
            temporal.write_bytes(datos)
            mat = loadmat(temporal)
            assert f'X{registro:03d}_DE_time' in mat
            temporal.replace(ruta)
        else:
            url_final = fuente['url']
            mat = loadmat(ruta)
        nuevo = np.ravel(mat[f'X{registro:03d}_DE_time']).astype('<f8')
        anterior = loadmat(DATOS/original)
        clave = next(k for k in anterior if k.endswith('_DE_time'))
        viejo = np.ravel(anterior[clave]).astype('<f8')
        assert not np.array_equal(nuevo, viejo), 'La fuente recuperada sigue siendo una copia'
        mask = indice.archivo_original.eq(original)
        assert mask.sum() == 1
        indice.loc[mask, 'archivo'] = f'fuente_verificada/{registro}.mat'
        indice.loc[mask, 'registro'] = registro
        manifest.append(dict(archivo_original=original, archivo_corregido=str(ruta.relative_to(DATOS)),
            **fuente, url_final=url_final, sha256_mat=hashlib.sha256(ruta.read_bytes()).hexdigest(),
            sha256_senal=hashlib.sha256(nuevo.tobytes()).hexdigest(), muestras=len(nuevo),
            sha256_senal_anterior=hashlib.sha256(viejo.tobytes()).hexdigest()))
        print(f'Fuente verificada: {original} -> {registro}.mat; {len(nuevo)} muestras', flush=True)
    assert len(indice) == 133 and indice.archivo.is_unique and indice.archivo_original.is_unique
    indice.to_csv(DATOS/'indice_fuente_verificada.csv', index=False)
    (destino/'procedencia.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Índice separado escrito. Ninguna señal histórica fue reemplazada.', flush=True)
