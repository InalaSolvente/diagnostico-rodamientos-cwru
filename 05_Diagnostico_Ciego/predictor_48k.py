"""Inferencia con el modelo entrenado específicamente en 48 kHz."""
from pathlib import Path
import hashlib
import numpy as np
from alinear_bandas import rasgos_alineados

MODELO_48K=Path(__file__).parent/'resultados_modelo_48k/modelo_48k.npz'


def predecir(x,fs,frecuencias_fabricante,*,modelo=MODELO_48K):
    if not np.isfinite(fs) or fs!=48000:raise ValueError('Este modelo requiere fs=48000 Hz')
    aqui=Path(__file__).parent
    with np.load(modelo,allow_pickle=False) as m:
        for nombre,campo in [('alinear_bandas.py','extractor_alineado_sha256'),('modelo_espectral.py','extractor_espectral_sha256')]:
            if hashlib.sha256((aqui/nombre).read_bytes()).hexdigest()!=str(m[campo]):
                raise ValueError('El extractor cambió; regenere el modelo de 48k')
        x,nodos,L=rasgos_alineados(x,fs,frecuencias_fabricante)
        if x.shape!=m['media'].shape or L!=int(m['L']):raise ValueError('Rasgos incompatibles')
        scores=((x-m['media'])/m['escala'])@m['coef'].T+m['intercepto']
        clases=m['clases'].tolist()
        return dict(tipo=clases[int(scores.argmax())],scores={str(c):float(s) for c,s in zip(clases,scores)},
                    clases_cubiertas=clases,nota='Scores discriminantes; no probabilidades calibradas')
